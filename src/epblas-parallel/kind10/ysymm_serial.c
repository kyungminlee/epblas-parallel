/*
 * ysymm_serial — kind10 complex (_Complex long double) symmetric
 * matrix-multiply, single-thread. This TU owns ALL of the ysymm math;
 * ysymm_parallel.c only orchestrates the outer panel loop across a team.
 *
 * Blocked "read A_IK once, use it twice" recipe (see ysymm_kernel.h). The
 * trailing updates run through ygemm_serial (NOT ygemm_): when a panel
 * worker runs inside the team ysymm_parallel.c opened, a nested ygemm team
 * would trip the libgomp barrier wedge.
 */

#include "ysymm_kernel.h"
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef ysymm_T T;

/* nb is fixed at 32 (the original symm_nb_pick clamps to [32, 32]). */
ptrdiff_t ysymm_nb(void) { return 32; }

extern void ygemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const T *alpha,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    const T *beta,
    T *c, ptrdiff_t ldc);

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* Scalar diagonal-block symm for SIDE='L' (see original ysymm for the
 * Netlib stride-1 access rationale). */
static void symm_diag_add_L(ptrdiff_t ic, ptrdiff_t ib, ptrdiff_t jc, ptrdiff_t jb, T alpha,
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
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (ptrdiff_t k = i + 1; k < ic + ib; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * ai[k];
                }
                cj[i] += temp1 * ai[i] + alpha * temp2;
            }
        } else {
            for (ptrdiff_t i = ic; i < ic + ib; ++i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (ptrdiff_t k = ic; k < i; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * ai[k];
                }
                cj[i] += temp1 * ai[i] + alpha * temp2;
            }
        }
    }
}

/* Scalar diagonal-block symm for SIDE='R'. */
static void symm_diag_add_R(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t ic, ptrdiff_t ib, T alpha,
                            const T *restrict a, ptrdiff_t lda,
                            const T *restrict b, ptrdiff_t ldb,
                            T *restrict c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        {
            const T t = alpha * A_(j, j);
            for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, j);
        }
        if (UPLO == 'L') {
            for (ptrdiff_t k = jc; k < j; ++k) {
                const T t = alpha * A_(j, k);
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
                const T t = alpha * A_(j, k);
                if (t != ZERO) for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
        }
    }
}

void ysymm_beta_only(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, T beta, T *c, ptrdiff_t ldc)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (ptrdiff_t i = 0; i < m; ++i) cj[i] = ZERO;
        else              for (ptrdiff_t i = 0; i < m; ++i) cj[i] *= beta;
    }
}

void ysymm_L_singleblock(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m,
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
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (ptrdiff_t k = i + 1; k < m; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * ai[k];
                }
                const T diag = temp1 * ai[i] + alpha * temp2;
                if (beta == ZERO)     cj[i] = diag;
                else if (beta == ONE) cj[i] += diag;
                else                  cj[i] = beta * cj[i] + diag;
            }
        } else {
            for (ptrdiff_t i = 0; i < m; ++i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (ptrdiff_t k = 0; k < i; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * ai[k];
                }
                const T diag = temp1 * ai[i] + alpha * temp2;
                if (beta == ZERO)     cj[i] = diag;
                else if (beta == ONE) cj[i] += diag;
                else                  cj[i] = beta * cj[i] + diag;
            }
        }
    }
}

void ysymm_L_panel(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t m, T alpha, T beta,
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
                ygemm_serial('T', 'N', kb, jb, ib, &alpha,
                             &A_(ic, kc), lda, &B_(ic, jc), ldb, &ONE,
                             &C_(kc, jc), ldc);
                ygemm_serial('N', 'N', ib, jb, kb, &alpha,
                             &A_(ic, kc), lda, &B_(kc, jc), ldb, &ONE,
                             &C_(ic, jc), ldc);
            }
        } else {
            for (ptrdiff_t kc = ic + ib; kc < m; kc += nb) {
                const ptrdiff_t kb = (m - kc < nb) ? (m - kc) : nb;
                ygemm_serial('T', 'N', kb, jb, ib, &alpha,
                             &A_(ic, kc), lda, &B_(ic, jc), ldb, &ONE,
                             &C_(kc, jc), ldc);
                ygemm_serial('N', 'N', ib, jb, kb, &alpha,
                             &A_(ic, kc), lda, &B_(kc, jc), ldb, &ONE,
                             &C_(ic, jc), ldc);
            }
        }

        symm_diag_add_L(ic, ib, jc, jb, alpha, a, lda, b, ldb, c, ldc, UPLO);
    }
}

void ysymm_R_panel(ptrdiff_t ic, ptrdiff_t ib, ptrdiff_t n, T alpha, T beta,
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
                ygemm_serial('N', 'T', ib, kb, jb, &alpha,
                             &B_(ic, jc), ldb, &A_(kc, jc), lda, &ONE,
                             &C_(ic, kc), ldc);
            }
        } else {
            for (ptrdiff_t kc = 0; kc < jc; kc += nb) {
                const ptrdiff_t kb = (jc - kc < nb) ? (jc - kc) : nb;
                ygemm_serial('N', 'N', ib, jb, kb, &alpha,
                             &B_(ic, kc), ldb, &A_(kc, jc), lda, &ONE,
                             &C_(ic, jc), ldc);
                ygemm_serial('N', 'T', ib, kb, jb, &alpha,
                             &B_(ic, jc), ldb, &A_(kc, jc), lda, &ONE,
                             &C_(ic, kc), ldc);
            }
        }

        symm_diag_add_R(jc, jb, ic, ib, alpha, a, lda, b, ldb, c, ldc, UPLO);
    }
}

void ysymm_serial(
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
        ysymm_beta_only(0, n, m, beta, c, ldc);
        return;
    }

    const ptrdiff_t nb = ysymm_nb();

    if (SIDE == 'L' && m <= nb) {
        ysymm_L_singleblock(0, n, m, alpha, beta, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }

    if (SIDE == 'L') {
        for (ptrdiff_t jc = 0; jc < n; jc += nb) {
            const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
            ysymm_L_panel(jc, jb, m, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    } else {
        for (ptrdiff_t ic = 0; ic < m; ic += nb) {
            const ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            ysymm_R_panel(ic, ib, n, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    }
}

#undef A_
#undef B_
#undef C_
