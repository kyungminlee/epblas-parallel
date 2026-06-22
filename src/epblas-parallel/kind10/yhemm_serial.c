/*
 * yhemm_serial — kind10 complex (_Complex long double) Hermitian
 * matrix-multiply, single-thread. This TU owns ALL of the yhemm math;
 * yhemm_parallel.c only orchestrates the outer panel loop across a team.
 *
 * Same "read A_IK once, use it twice" recipe as ysymm, but Hermitian: the
 * reflection gemm uses 'C' and the scalar diagonal block keeps the diagonal
 * real. The trailing updates run through ygemm_serial (NOT ygemm_): when a
 * panel worker runs inside the team yhemm_parallel.c opened, a nested ygemm
 * team would trip the libgomp barrier wedge (project-etrsm-omp4-wedge).
 */

#include "yhemm_kernel.h"
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef yhemm_T T;

/* nb is fixed at 32 (the original hemm_nb_pick clamps to [32, 32]). */
ptrdiff_t yhemm_nb(void) { return 32; }

extern void ygemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const T *alpha,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    const T *beta,
    T *c, ptrdiff_t ldc);

static inline T cconj(T z) { return ~z; }

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* Fused Hermitian-Left inner sweep over k in [k_lo, k_hi):
 *     cj[k]  += temp1 * ai[k]
 *     temp2  += bj[k] * conj(ai[k])
 * returning temp2. Driven as scalar long double re/im chains over the complex
 * data reinterpreted as 2n reals: each ai[k]/bj[k] element is loaded ONCE and
 * shared by both the cj axpy and the temp2 conj-dot, so the cj[k] store can't
 * force a reload of ai/bj — the gcc aliasing-reload (ai[k] and bj[k] each
 * fetched twice per iter) that cost par ~7% vs gfortran's zhemm on LL. The
 * conj sign is folded into the products, dropping the per-element fchs.
 * Bit-identical to the _Complex form (same decomposition as yherk UC/LC). */
static inline T hemm_L_conj_sweep(T *restrict cj, const T *restrict bj,
                                  const T *restrict ai, T temp1,
                                  ptrdiff_t k_lo, ptrdiff_t k_hi)
{
    long double *restrict cR = (long double *)cj;
    const long double *restrict bR = (const long double *)bj;
    const long double *restrict aR = (const long double *)ai;
    const long double t1r = __real__ temp1, t1i = __imag__ temp1;
    long double s2r = 0.0L, s2i = 0.0L;
    for (ptrdiff_t k = k_lo; k < k_hi; ++k) {
        const long double akr = aR[2 * k], aki = aR[2 * k + 1];
        const long double bkr = bR[2 * k], bki = bR[2 * k + 1];
        cR[2 * k]     += t1r * akr - t1i * aki;
        cR[2 * k + 1] += t1r * aki + t1i * akr;
        s2r += bkr * akr + bki * aki;
        s2i += bki * akr - bkr * aki;
    }
    return s2r + s2i * 1.0iL;
}

/* Scalar Hermitian diagonal block, SIDE='L' (see original yhemm for the
 * Netlib stride-1 access rationale). */
static void hemm_diag_add_L(ptrdiff_t ic, ptrdiff_t ib, ptrdiff_t jc, ptrdiff_t jb, T alpha,
                            const T *restrict a, ptrdiff_t lda,
                            const T *restrict b, ptrdiff_t ldb,
                            T *restrict c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        const T *bj = b + (size_t)j * ldb;
        if (UPLO == 'L') {
            for (ptrdiff_t i = ic + ib - 1; i >= ic; --i) {
                const T temp1 = alpha * bj[i];
                const T *ai = &A_(0, i);
                const T temp2 = hemm_L_conj_sweep(cj, bj, ai, temp1, i + 1, ic + ib);
                cj[i] += temp1 * __real__ ai[i] + alpha * temp2;
            }
        } else {
            for (ptrdiff_t i = ic; i < ic + ib; ++i) {
                const T temp1 = alpha * bj[i];
                const T *ai = &A_(0, i);
                const T temp2 = hemm_L_conj_sweep(cj, bj, ai, temp1, ic, i);
                cj[i] += temp1 * __real__ ai[i] + alpha * temp2;
            }
        }
    }
}

/* Scalar Hermitian diagonal block, SIDE='R'. */
static void hemm_diag_add_R(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t ic, ptrdiff_t ib, T alpha,
                            const T *restrict a, ptrdiff_t lda,
                            const T *restrict b, ptrdiff_t ldb,
                            T *restrict c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        {
            const T t = alpha * (__real__ A_(j, j));
            for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, j);
        }
        if (UPLO == 'L') {
            for (ptrdiff_t k = jc; k < j; ++k) {
                const T t = alpha * cconj(A_(j, k));
                if (t != ZERO) for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
            for (ptrdiff_t k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * A_(k, j);
                if (t != ZERO) for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
        } else {
            for (ptrdiff_t k = jc; k < j; ++k) {
                const T t = alpha * A_(k, j);
                if (t != ZERO) for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
            for (ptrdiff_t k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * cconj(A_(j, k));
                if (t != ZERO) for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
        }
    }
}

void yhemm_beta_only(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T beta, T *c, ptrdiff_t ldc)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (ptrdiff_t i = 0; i < m; ++i) cj[i] = ZERO;
        else              for (ptrdiff_t i = 0; i < m; ++i) cj[i] *= beta;
    }
}

void yhemm_L_singleblock(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m,
                         T alpha, T beta,
                         const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                         T *c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        T *cj = c + (size_t)j * ldc;
        const T *bj = b + (size_t)j * ldb;
        if (UPLO == 'L') {
            for (ptrdiff_t i = m - 1; i >= 0; --i) {
                const T temp1 = alpha * bj[i];
                const T *ai = &A_(0, i);
                const T temp2 = hemm_L_conj_sweep(cj, bj, ai, temp1, i + 1, m);
                const T diag = temp1 * __real__ ai[i] + alpha * temp2;
                if (beta == ZERO)     cj[i] = diag;
                else if (beta == ONE) cj[i] += diag;
                else                  cj[i] = beta * cj[i] + diag;
            }
        } else {
            for (ptrdiff_t i = 0; i < m; ++i) {
                const T temp1 = alpha * bj[i];
                const T *ai = &A_(0, i);
                const T temp2 = hemm_L_conj_sweep(cj, bj, ai, temp1, 0, i);
                const T diag = temp1 * __real__ ai[i] + alpha * temp2;
                if (beta == ZERO)     cj[i] = diag;
                else if (beta == ONE) cj[i] += diag;
                else                  cj[i] = beta * cj[i] + diag;
            }
        }
    }
}

void yhemm_L_panel(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t m, T alpha, T beta,
                   const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                   T *c, ptrdiff_t ldc, char UPLO, ptrdiff_t nb)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (ptrdiff_t i = 0; i < m; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (ptrdiff_t i = 0; i < m; ++i) cj[i] *= beta;
    }

    for (ptrdiff_t ic = 0; ic < m; ic += nb) {
        const ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;

        if (UPLO == 'L') {
            for (ptrdiff_t kc = 0; kc < ic; kc += nb) {
                const ptrdiff_t kb = (ic - kc < nb) ? (ic - kc) : nb;
                ygemm_serial('C', 'N', kb, jb, ib, &alpha,
                             &A_(ic, kc), lda, &B_(ic, jc), ldb, &ONE,
                             &C_(kc, jc), ldc);
                ygemm_serial('N', 'N', ib, jb, kb, &alpha,
                             &A_(ic, kc), lda, &B_(kc, jc), ldb, &ONE,
                             &C_(ic, jc), ldc);
            }
        } else {
            for (ptrdiff_t kc = ic + ib; kc < m; kc += nb) {
                const ptrdiff_t kb = (m - kc < nb) ? (m - kc) : nb;
                ygemm_serial('C', 'N', kb, jb, ib, &alpha,
                             &A_(ic, kc), lda, &B_(ic, jc), ldb, &ONE,
                             &C_(kc, jc), ldc);
                ygemm_serial('N', 'N', ib, jb, kb, &alpha,
                             &A_(ic, kc), lda, &B_(kc, jc), ldb, &ONE,
                             &C_(ic, jc), ldc);
            }
        }

        hemm_diag_add_L(ic, ib, jc, jb, alpha, a, lda, b, ldb, c, ldc, UPLO);
    }
}

void yhemm_R_panel(ptrdiff_t ic, ptrdiff_t ib, ptrdiff_t n, T alpha, T beta,
                   const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                   T *c, ptrdiff_t ldc, char UPLO, ptrdiff_t nb)
{
    for (ptrdiff_t j = 0; j < n; ++j) {
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] *= beta;
    }

    for (ptrdiff_t jc = 0; jc < n; jc += nb) {
        const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;

        if (UPLO == 'L') {
            for (ptrdiff_t kc = jc + jb; kc < n; kc += nb) {
                const ptrdiff_t kb = (n - kc < nb) ? (n - kc) : nb;
                ygemm_serial('N', 'N', ib, jb, kb, &alpha,
                             &B_(ic, kc), ldb, &A_(kc, jc), lda, &ONE,
                             &C_(ic, jc), ldc);
                ygemm_serial('N', 'C', ib, kb, jb, &alpha,
                             &B_(ic, jc), ldb, &A_(kc, jc), lda, &ONE,
                             &C_(ic, kc), ldc);
            }
        } else {
            for (ptrdiff_t kc = 0; kc < jc; kc += nb) {
                const ptrdiff_t kb = (jc - kc < nb) ? (jc - kc) : nb;
                ygemm_serial('N', 'N', ib, jb, kb, &alpha,
                             &B_(ic, kc), ldb, &A_(kc, jc), lda, &ONE,
                             &C_(ic, jc), ldc);
                ygemm_serial('N', 'C', ib, kb, jb, &alpha,
                             &B_(ic, jc), ldb, &A_(kc, jc), lda, &ONE,
                             &C_(ic, kc), ldc);
            }
        }

        hemm_diag_add_R(jc, jb, ic, ib, alpha, a, lda, b, ldb, c, ldc, UPLO);
    }
}

void yhemm_serial(
    char side, char uplo,
    ptrdiff_t m, ptrdiff_t n,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    const T *beta_,
    T *c, ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = blas_up(side);
    const char UPLO = blas_up(uplo);

    if (m == 0 || n == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
        yhemm_beta_only(0, n, m, beta, c, ldc);
        return;
    }

    const ptrdiff_t nb = yhemm_nb();

    if (SIDE == 'L' && m <= nb) {
        yhemm_L_singleblock(0, n, m, alpha, beta, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }

    if (SIDE == 'L') {
        for (ptrdiff_t jc = 0; jc < n; jc += nb) {
            const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
            yhemm_L_panel(jc, jb, m, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    } else {
        for (ptrdiff_t ic = 0; ic < m; ic += nb) {
            const ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            yhemm_R_panel(ic, ib, n, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    }
}

#undef A_
#undef B_
#undef C_
