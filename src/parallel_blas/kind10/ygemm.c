/*
 * ygemm — kind10 complex GEMM overlay
 *   (COMPLEX(KIND=10), x86-64 _Complex long double, 32-byte struct
 *    matching gfortran's complex(10) ABI).
 *
 * Strategy: reference algorithm + OpenMP-over-j.
 *
 * Rationale: on x86-64 there is no SIMD path for long double or its
 * complex form — every multiply goes through the 8-deep x87 register
 * stack. A complex `a * b` already spills several fp80 temporaries,
 * so any register-tile larger than ~1 accumulator over-pressures the
 * stack. We tried OpenBLAS-style MR×NR outer-product tiles and they
 * regressed slightly versus the simpler reference path. Packing also
 * costs more than it saves at the sizes that matter here (a 256×256
 * complex(10) panel is 2 MB, larger than the L2 it would warm).
 *
 * What does win:
 *   - Parallelism across the j (column-of-C) axis is embarrassingly
 *     parallel and scales linearly until memory bandwidth saturates.
 *   - Stride-1 inner loops over A and B where the orientation permits
 *     — handled by selecting one of four loop bodies based on
 *     (TRANSA, TRANSB). The same as the migrated zgemm in shape;
 *     overlay wins purely by parallelizing the outer j-loop.
 *
 * Fortran ABI:
 *   - subroutine name lowercased + trailing underscore: `ygemm_`
 *   - scalars by pointer; complex scalar = pointer to (re, im) pair
 *   - character args followed by hidden trailing `size_t` lengths
 *   - COMPLEX(KIND=10) ↔ `_Complex long double` (32 bytes on x86-64)
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <complex.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef _Complex long double C10;

static int trans_code(const char *p, size_t len) {
    (void)len;
    return (char)toupper((unsigned char)*p);
}

void ygemm_(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const C10 *alpha_,
    const C10 *a, const int *lda_,
    const C10 *b, const int *ldb_,
    const C10 *beta_,
    C10 *c, const int *ldc_,
    size_t transa_len, size_t transb_len)
{
    const int M = *m_, N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const C10 alpha = *alpha_, beta = *beta_;
    const int ta = trans_code(transa, transa_len);
    const int tb = trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    const C10 zero = 0.0L + 0.0iL;
    const C10 one  = 1.0L + 0.0iL;

    /* Beta pre-pass — handles K==0 / alpha==0 paths in one place. */
    for (int j = 0; j < N; ++j) {
        C10 *cj = &c[(size_t)j * ldc];
        if (beta == zero)      for (int i = 0; i < M; ++i) cj[i]  = zero;
        else if (beta != one)  for (int i = 0; i < M; ++i) cj[i] *= beta;
    }
    if (alpha == zero || K == 0) return;

    /*
     * Four orientation paths, parallel across j. Inner k-loop is
     * stride-1 on A or B (or both) wherever possible. Conjugation
     * on A or B is absorbed via the `~` operator on _Complex.
     */
    const int conj_a = (ta == 'C');
    const int conj_b = (tb == 'C');

    /*
     * Rank-1 paths are unrolled on the K (depth) axis, not on I. The
     * migrated Netlib reference compiled by gfortran runs two K iters
     * in parallel — precomputes TEMP_L and TEMP_{L+1}, then per-i
     * accumulates `c[i,j] += TEMP_L * A[i,L] + TEMP_{L+1} * A[i,L+1]`.
     * The two complex FMAs per i form independent chains and mask
     * x87 fmul latency. Unrolling on I instead doesn't help — each
     * row already targets a distinct C location, no extra ILP
     * exposed. See `ygemm_migrated_` disasm at offset ~0xab8.
     */
    if (ta == 'N' && tb == 'N') {
        /* C[i,j] += sum_l (alpha*B[l,j]) * A[i,l].  Rank-1 update over l,
         * unrolled by 2 on l so each i gets two independent FMA chains. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int j2 = 0; j2 < N; ++j2) {
            C10 *cj = &c[(size_t)j2 * ldc];
            int l = 0;
            for (; l + 1 < K; l += 2) {
                const C10 t0 = alpha * b[(size_t)j2 * ldb + l];
                const C10 t1 = alpha * b[(size_t)j2 * ldb + l + 1];
                const C10 *al0 = &a[(size_t)l       * lda];
                const C10 *al1 = &a[(size_t)(l + 1) * lda];
                for (int i2 = 0; i2 < M; ++i2)
                    cj[i2] += t0 * al0[i2] + t1 * al1[i2];
            }
            for (; l < K; ++l) {
                const C10 t = alpha * b[(size_t)j2 * ldb + l];
                const C10 *al = &a[(size_t)l * lda];
                for (int i2 = 0; i2 < M; ++i2) cj[i2] += t * al[i2];
            }
        }
    } else if ((ta == 'T' || ta == 'C') && tb == 'N') {
        /* A^op[i,l] = A[l,i] (or conjugated). Dot of A col i and B col j.
         * Single-acc form: gcc already schedules this well; manual unroll
         * with two accs regresses because the original form fits its
         * register allocation. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int j2 = 0; j2 < N; ++j2) {
            C10 *cj = &c[(size_t)j2 * ldc];
            const C10 *bj = &b[(size_t)j2 * ldb];
            for (int i2 = 0; i2 < M; ++i2) {
                const C10 *ai = &a[(size_t)i2 * lda];
                C10 acc = zero;
                if (conj_a) for (int l = 0; l < K; ++l) acc += ~ai[l] * bj[l];
                else        for (int l = 0; l < K; ++l) acc +=  ai[l] * bj[l];
                cj[i2] += alpha * acc;
            }
        }
    } else if (ta == 'N' && (tb == 'T' || tb == 'C')) {
        /* B^op[l,j] = B[j,l] (or conj). Rank-1 update over l, K-unrolled. */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int j2 = 0; j2 < N; ++j2) {
            C10 *cj = &c[(size_t)j2 * ldc];
            int l = 0;
            for (; l + 1 < K; l += 2) {
                const C10 b0 = b[(size_t)l       * ldb + j2];
                const C10 b1 = b[(size_t)(l + 1) * ldb + j2];
                const C10 t0 = alpha * (conj_b ? ~b0 : b0);
                const C10 t1 = alpha * (conj_b ? ~b1 : b1);
                const C10 *al0 = &a[(size_t)l       * lda];
                const C10 *al1 = &a[(size_t)(l + 1) * lda];
                for (int i2 = 0; i2 < M; ++i2)
                    cj[i2] += t0 * al0[i2] + t1 * al1[i2];
            }
            for (; l < K; ++l) {
                const C10 blj = b[(size_t)l * ldb + j2];
                const C10 t   = alpha * (conj_b ? ~blj : blj);
                const C10 *al = &a[(size_t)l * lda];
                for (int i2 = 0; i2 < M; ++i2) cj[i2] += t * al[i2];
            }
        }
    } else {
        /* Both transposed. A col i × B row j. Dot-product form — single
         * accumulator (same reason as the T*N path; the conditional on
         * conj_a/conj_b inside an unrolled hot loop wrecks codegen). */
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int j2 = 0; j2 < N; ++j2) {
            C10 *cj = &c[(size_t)j2 * ldc];
            for (int i2 = 0; i2 < M; ++i2) {
                const C10 *ai = &a[(size_t)i2 * lda];
                C10 acc = zero;
                for (int l = 0; l < K; ++l) {
                    const C10 av = conj_a ? ~ai[l] : ai[l];
                    const C10 bv = b[(size_t)l * ldb + j2];
                    acc += av * (conj_b ? ~bv : bv);
                }
                cj[i2] += alpha * acc;
            }
        }
    }
}
