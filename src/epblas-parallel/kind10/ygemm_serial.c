/*
 * ygemm_serial — kind10 complex GEMM (COMPLEX(KIND=10), x86-64 _Complex
 * long double, 32-byte struct matching gfortran's complex(10) ABI),
 * single-thread. This TU owns ALL of the ygemm math; ygemm_parallel.c
 * only orchestrates threads over these same cores.
 *
 * Strategy: reference algorithm, four orientation bodies selected by
 * (TRANSA, TRANSB).
 *
 * Rationale: on x86-64 there is no SIMD path for long double or its
 * complex form — every multiply goes through the 8-deep x87 register
 * stack. A complex `a * b` already spills several fp80 temporaries,
 * so any register-tile larger than ~1 accumulator over-pressures the
 * stack. We tried OpenBLAS-style MR×NR outer-product tiles and they
 * regressed slightly versus the simpler reference path. Packing also
 * costs more than it saves at the sizes that matter here (a 256×256
 * complex(10) panel is 2 MB, larger than the L2 it would warm). So the
 * win is purely parallelizing the outer j-loop (ygemm_parallel.c) over
 * these stride-1 reference bodies — the same shape as the migrated zgemm.
 *
 * Fortran ABI (ygemm_serial mirrors ygemm_ exactly):
 *   - scalars by pointer; complex scalar = pointer to (re, im) pair
 *   - character args followed by hidden trailing `size_t` lengths
 *   - COMPLEX(KIND=10) ↔ `_Complex long double` (32 bytes on x86-64)
 */

#include "ygemm_kernel.h"
#include <ctype.h>

typedef ygemm_T T;

static const T zero = 0.0L + 0.0iL;
static const T one  = 1.0L + 0.0iL;

static int trans_code(const char *p, size_t len) {
    (void)len;
    return (char)toupper((unsigned char)*p);
}

/* ── beta pre-pass ────────────────────────────────────────────── */

void ygemm_beta_prepass(int M, int N, T beta, T *c, int ldc) {
    for (int j = 0; j < N; ++j) {
        T *cj = &c[(size_t)j * ldc];
        if (beta == zero)      for (int i = 0; i < M; ++i) cj[i]  = zero;
        else if (beta != one)  for (int i = 0; i < M; ++i) cj[i] *= beta;
    }
}

/* ── Orientation cores ────────────────────────────────────────────
 *
 * Rank-1 paths are unrolled on the K (depth) axis, not on I. The
 * migrated Netlib reference compiled by gfortran runs two K iters in
 * parallel — precomputes TEMP_L and TEMP_{L+1}, then per-i accumulates
 * `c[i,j] += TEMP_L * A[i,L] + TEMP_{L+1} * A[i,L+1]`. The two complex
 * FMAs per i form independent chains and mask x87 fmul latency.
 * Unrolling on I instead doesn't help — each row already targets a
 * distinct C location, no extra ILP exposed. See `ygemm_migrated_`
 * disasm at offset ~0xab8.
 */

/* TA='N', TB='N': C[i,j] += sum_l (alpha*B[l,j]) * A[i,l]. */
void ygemm_nn_core(int j_start, int j_end, int M, int K, T alpha,
                   const T *a, int lda, const T *b, int ldb,
                   T *c, int ldc)
{
    for (int j2 = j_start; j2 < j_end; ++j2) {
        T *cj = &c[(size_t)j2 * ldc];
        int l = 0;
        for (; l + 1 < K; l += 2) {
            const T t0 = alpha * b[(size_t)j2 * ldb + l];
            const T t1 = alpha * b[(size_t)j2 * ldb + l + 1];
            const T *al0 = &a[(size_t)l       * lda];
            const T *al1 = &a[(size_t)(l + 1) * lda];
            for (int i2 = 0; i2 < M; ++i2)
                cj[i2] += t0 * al0[i2] + t1 * al1[i2];
        }
        for (; l < K; ++l) {
            const T t = alpha * b[(size_t)j2 * ldb + l];
            const T *al = &a[(size_t)l * lda];
            for (int i2 = 0; i2 < M; ++i2) cj[i2] += t * al[i2];
        }
    }
}

/* TA in {'T','C'}, TB='N': A^op[i,l] = A[l,i] (or conjugated). Dot of A
 * col i and B col j. Single-acc form: gcc already schedules this well;
 * manual unroll with two accs regresses because the original form fits
 * its register allocation. */
void ygemm_tn_core(int j_start, int j_end, int M, int K, T alpha,
                   const T *a, int lda, const T *b, int ldb,
                   T *c, int ldc, int conj_a)
{
    for (int j2 = j_start; j2 < j_end; ++j2) {
        T *cj = &c[(size_t)j2 * ldc];
        const T *bj = &b[(size_t)j2 * ldb];
        for (int i2 = 0; i2 < M; ++i2) {
            const T *ai = &a[(size_t)i2 * lda];
            T acc = zero;
            if (conj_a) for (int l = 0; l < K; ++l) acc += ~ai[l] * bj[l];
            else        for (int l = 0; l < K; ++l) acc +=  ai[l] * bj[l];
            cj[i2] += alpha * acc;
        }
    }
}

/* TA='N', TB in {'T','C'}: B^op[l,j] = B[j,l] (or conj). Rank-1 update
 * over l, K-unrolled. */
void ygemm_nt_core(int j_start, int j_end, int M, int K, T alpha,
                   const T *a, int lda, const T *b, int ldb,
                   T *c, int ldc, int conj_b)
{
    for (int j2 = j_start; j2 < j_end; ++j2) {
        T *cj = &c[(size_t)j2 * ldc];
        int l = 0;
        for (; l + 1 < K; l += 2) {
            const T b0 = b[(size_t)l       * ldb + j2];
            const T b1 = b[(size_t)(l + 1) * ldb + j2];
            const T t0 = alpha * (conj_b ? ~b0 : b0);
            const T t1 = alpha * (conj_b ? ~b1 : b1);
            const T *al0 = &a[(size_t)l       * lda];
            const T *al1 = &a[(size_t)(l + 1) * lda];
            for (int i2 = 0; i2 < M; ++i2)
                cj[i2] += t0 * al0[i2] + t1 * al1[i2];
        }
        for (; l < K; ++l) {
            const T blj = b[(size_t)l * ldb + j2];
            const T t   = alpha * (conj_b ? ~blj : blj);
            const T *al = &a[(size_t)l * lda];
            for (int i2 = 0; i2 < M; ++i2) cj[i2] += t * al[i2];
        }
    }
}

/* Both transposed: A col i × B row j. Dot-product form — single
 * accumulator (same reason as the T*N path; the conditional on
 * conj_a/conj_b inside an unrolled hot loop wrecks codegen). */
void ygemm_tt_core(int j_start, int j_end, int M, int K, T alpha,
                   const T *a, int lda, const T *b, int ldb,
                   T *c, int ldc, int conj_a, int conj_b)
{
    for (int j2 = j_start; j2 < j_end; ++j2) {
        T *cj = &c[(size_t)j2 * ldc];
        for (int i2 = 0; i2 < M; ++i2) {
            const T *ai = &a[(size_t)i2 * lda];
            T acc = zero;
            for (int l = 0; l < K; ++l) {
                const T av = conj_a ? ~ai[l] : ai[l];
                const T bv = b[(size_t)l * ldb + j2];
                acc += av * (conj_b ? ~bv : bv);
            }
            cj[i2] += alpha * acc;
        }
    }
}

/* ── Single-thread entry ──────────────────────────────────────── */

void ygemm_serial(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t transa_len, size_t transb_len)
{
    const int M = *m_, N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const int ta = trans_code(transa, transa_len);
    const int tb = trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    ygemm_beta_prepass(M, N, beta, c, ldc);
    if (alpha == zero || K == 0) return;

    const int conj_a = (ta == 'C');
    const int conj_b = (tb == 'C');

    if (ta == 'N' && tb == 'N') {
        ygemm_nn_core(0, N, M, K, alpha, a, lda, b, ldb, c, ldc);
    } else if ((ta == 'T' || ta == 'C') && tb == 'N') {
        ygemm_tn_core(0, N, M, K, alpha, a, lda, b, ldb, c, ldc, conj_a);
    } else if (ta == 'N' && (tb == 'T' || tb == 'C')) {
        ygemm_nt_core(0, N, M, K, alpha, a, lda, b, ldb, c, ldc, conj_b);
    } else {
        ygemm_tt_core(0, N, M, K, alpha, a, lda, b, ldb, c, ldc, conj_a, conj_b);
    }
}
