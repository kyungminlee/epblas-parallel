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
#include "../common/blas_char.h"
#include <ctype.h>
#include <stddef.h>

typedef ygemm_TC TC;

static const TC zero = 0.0L + 0.0iL;
static const TC one  = 1.0L + 0.0iL;

/* ── beta pre-pass ────────────────────────────────────────────── */

void ygemm_beta_prepass(ptrdiff_t m, ptrdiff_t n, TC beta, TC *c, ptrdiff_t ldc) {
    for (ptrdiff_t j = 0; j < n; ++j) {
        TC *cj = &c[(size_t)j * ldc];
        if (beta == zero)      for (ptrdiff_t i = 0; i < m; ++i) cj[i]  = zero;
        else if (beta != one)  for (ptrdiff_t i = 0; i < m; ++i) cj[i] *= beta;
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
void ygemm_nn_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, ptrdiff_t k, TC alpha,
                   const TC *a, ptrdiff_t lda, const TC *b, ptrdiff_t ldb,
                   TC *c, ptrdiff_t ldc)
{
    for (ptrdiff_t j2 = j_start; j2 < j_end; ++j2) {
        TC *cj = &c[(size_t)j2 * ldc];
        ptrdiff_t l = 0;
        for (; l + 1 < k; l += 2) {
            const TC t0 = alpha * b[(size_t)j2 * ldb + l];
            const TC t1 = alpha * b[(size_t)j2 * ldb + l + 1];
            const TC *al0 = &a[(size_t)l       * lda];
            const TC *al1 = &a[(size_t)(l + 1) * lda];
            for (ptrdiff_t i2 = 0; i2 < m; ++i2)
                cj[i2] += t0 * al0[i2] + t1 * al1[i2];
        }
        for (; l < k; ++l) {
            const TC t = alpha * b[(size_t)j2 * ldb + l];
            const TC *al = &a[(size_t)l * lda];
            for (ptrdiff_t i2 = 0; i2 < m; ++i2) cj[i2] += t * al[i2];
        }
    }
}

/* TA in {'T','C'}, TB='N': A^op[i,l] = A[l,i] (or conjugated). Dot of A
 * col i and B col j. Single complex accumulator at the x87 floor.
 *
 * The accumulator is decomposed into two scalar `long double` chains
 * (acc_re, acc_im) over the real/imag parts of A and B (the complex
 * arrays reinterpreted as 2N interleaved reals). This is bit-identical
 * to the `_Complex` form — the products and add order are exactly what
 * gfortran's `~a*b` / `a*b` expand to — but it lets gcc keep both fadd
 * chains scheduled without the `fchs` that the `~ai[l]` form forces onto
 * the conjugate critical path (the conj sign folds into the products:
 * conj(a)*b = (ar*br + ai*bi) + i(ar*bi - ai*br)). Measured ~8% faster
 * on the conjugate (TRANS='C') path that yherk/yhemm/ysyrk route their
 * trailing update through; parity on the plain transpose path ygemm uses.
 * (K-unrolling with a single complex acc instead REGRESSES ~16% and is
 * NOT bit-exact — it reorders the dot.) See `ygemm_tn_core` disasm. */
void ygemm_tn_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, ptrdiff_t k, TC alpha,
                   const TC *a, ptrdiff_t lda, const TC *b, ptrdiff_t ldb,
                   TC *c, ptrdiff_t ldc, bool conj_a)
{
    for (ptrdiff_t j2 = j_start; j2 < j_end; ++j2) {
        TC *cj = &c[(size_t)j2 * ldc];
        const long double *bj = (const long double *)&b[(size_t)j2 * ldb];
        for (ptrdiff_t i2 = 0; i2 < m; ++i2) {
            const long double *ai = (const long double *)&a[(size_t)i2 * lda];
            long double acc_re = 0.0L, acc_im = 0.0L;
            if (conj_a) {
                for (ptrdiff_t l = 0; l < k; ++l) {
                    const long double ar = ai[2*l], aim = ai[2*l+1];
                    const long double br = bj[2*l], bim = bj[2*l+1];
                    acc_re += ar * br + aim * bim;
                    acc_im += ar * bim - aim * br;
                }
            } else {
                for (ptrdiff_t l = 0; l < k; ++l) {
                    const long double ar = ai[2*l], aim = ai[2*l+1];
                    const long double br = bj[2*l], bim = bj[2*l+1];
                    acc_re += ar * br - aim * bim;
                    acc_im += ar * bim + aim * br;
                }
            }
            cj[i2] += alpha * (acc_re + acc_im * 1.0iL);
        }
    }
}

/* TA='N', TB in {'T','C'}: B^op[l,j] = B[j,l] (or conj). Rank-1 update
 * over l, K-unrolled. */
void ygemm_nt_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, ptrdiff_t k, TC alpha,
                   const TC *a, ptrdiff_t lda, const TC *b, ptrdiff_t ldb,
                   TC *c, ptrdiff_t ldc, bool conj_b)
{
    for (ptrdiff_t j2 = j_start; j2 < j_end; ++j2) {
        TC *cj = &c[(size_t)j2 * ldc];
        ptrdiff_t l = 0;
        for (; l + 1 < k; l += 2) {
            const TC b0 = b[(size_t)l       * ldb + j2];
            const TC b1 = b[(size_t)(l + 1) * ldb + j2];
            const TC t0 = alpha * (conj_b ? ~b0 : b0);
            const TC t1 = alpha * (conj_b ? ~b1 : b1);
            const TC *al0 = &a[(size_t)l       * lda];
            const TC *al1 = &a[(size_t)(l + 1) * lda];
            for (ptrdiff_t i2 = 0; i2 < m; ++i2)
                cj[i2] += t0 * al0[i2] + t1 * al1[i2];
        }
        for (; l < k; ++l) {
            const TC blj = b[(size_t)l * ldb + j2];
            const TC t   = alpha * (conj_b ? ~blj : blj);
            const TC *al = &a[(size_t)l * lda];
            for (ptrdiff_t i2 = 0; i2 < m; ++i2) cj[i2] += t * al[i2];
        }
    }
}

/* Both transposed: A col i × B row j. Dot-product form — single
 * accumulator (same reason as the T*N path; the conditional on
 * conj_a/conj_b inside an unrolled hot loop wrecks codegen). */
void ygemm_tt_core(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, ptrdiff_t k, TC alpha,
                   const TC *a, ptrdiff_t lda, const TC *b, ptrdiff_t ldb,
                   TC *c, ptrdiff_t ldc, bool conj_a, bool conj_b)
{
    for (ptrdiff_t j2 = j_start; j2 < j_end; ++j2) {
        TC *cj = &c[(size_t)j2 * ldc];
        for (ptrdiff_t i2 = 0; i2 < m; ++i2) {
            const TC *ai = &a[(size_t)i2 * lda];
            TC acc = zero;
            for (ptrdiff_t l = 0; l < k; ++l) {
                const TC av = conj_a ? ~ai[l] : ai[l];
                const TC bv = b[(size_t)l * ldb + j2];
                acc += av * (conj_b ? ~bv : bv);
            }
            cj[i2] += alpha * acc;
        }
    }
}

/* ── Single-thread entry ──────────────────────────────────────── */

void ygemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const TC *alpha_,
    const TC *a, ptrdiff_t lda,
    const TC *b, ptrdiff_t ldb,
    const TC *beta_,
    TC *c, ptrdiff_t ldc)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char ta = blas_trans_complex(transa);
    const char tb = blas_trans_complex(transb);

    if (m <= 0 || n <= 0) return;

    ygemm_beta_prepass(m, n, beta, c, ldc);
    if (alpha == zero || k == 0) return;

    const bool conj_a = (ta == 'C');
    const bool conj_b = (tb == 'C');

    if (ta == 'N' && tb == 'N') {
        ygemm_nn_core(0, n, m, k, alpha, a, lda, b, ldb, c, ldc);
    } else if ((ta == 'T' || ta == 'C') && tb == 'N') {
        ygemm_tn_core(0, n, m, k, alpha, a, lda, b, ldb, c, ldc, conj_a);
    } else if (ta == 'N' && (tb == 'T' || tb == 'C')) {
        ygemm_nt_core(0, n, m, k, alpha, a, lda, b, ldb, c, ldc, conj_b);
    } else {
        ygemm_tt_core(0, n, m, k, alpha, a, lda, b, ldb, c, ldc, conj_a, conj_b);
    }
}
