/*
 * ysyrk_serial — kind10 complex (_Complex long double) symmetric rank-k,
 * single-thread. This TU owns ALL of the ysyrk math; ysyrk_parallel.c only
 * orchestrates the diagonal-block loop across a team.
 *
 * TRANS ∈ {N, T}. Complex syrk does NOT conjugate (see yherk). Blocked:
 * scalar diagonal + ygemm trailing. The trailing update runs through
 * ygemm_serial (NOT ygemm_): when ysyrk_block runs inside the team
 * ysyrk_parallel.c opened, a nested ygemm team would trip the libgomp
 * barrier wedge.
 */

#include "ysyrk_kernel.h"
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef ysyrk_TC TC;

ptrdiff_t ysyrk_nb(void) { return 32; }

extern void ygemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const TC *alpha,
    const TC *a, ptrdiff_t lda,
    const TC *b, ptrdiff_t ldb,
    const TC *beta,
    TC *c, ptrdiff_t ldc);

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const TC ZERO = 0.0L + 0.0Li;
static const TC ONE  = 1.0L + 0.0Li;

static void syrk_diag_add(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t k, TC alpha,
                          const TC *restrict a, ptrdiff_t lda,
                          TC *restrict c, ptrdiff_t ldc,
                          char UPLO, char TRANS)
{
    if (TRANS == 'N') {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            for (ptrdiff_t l = 0; l < k; ++l) {
                const TC ajl = A_(j, l);
                if (ajl != ZERO) {
                    const TC t = alpha * ajl;
                    for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] += t * A_(i, l);
                }
            }
        }
    } else {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            const TC *Aj = a + (size_t)j * lda;
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                const TC *Ai = a + (size_t)i * lda;
                TC s = ZERO;
                for (ptrdiff_t l = 0; l < k; ++l) s += Ai[l] * Aj[l];
                cj[i] += alpha * s;
            }
        }
    }
}

void ysyrk_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, TC beta,
                      TC *c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
        TC *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = ZERO;
        else              for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }
}

void ysyrk_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t n, ptrdiff_t k, TC alpha, TC beta,
                 const TC *a, ptrdiff_t lda, TC *c, ptrdiff_t ldc, char UPLO, char TRANS)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
        TC *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }

    syrk_diag_add(jc, jb, k, alpha, a, lda, c, ldc, UPLO, TRANS);

    if (UPLO == 'L') {
        const ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            const ptrdiff_t j0 = jc + jb;
            if (TRANS == 'N') {
                ygemm_serial('N', 'T', trailing, jb, k, &alpha,
                             &A_(j0, 0), lda, &A_(jc, 0), lda,
                             &ONE, &C_(j0, jc), ldc);
            } else {
                ygemm_serial('T', 'N', trailing, jb, k, &alpha,
                             &A_(0, j0), lda, &A_(0, jc), lda,
                             &ONE, &C_(j0, jc), ldc);
            }
        }
    } else {
        if (jc > 0) {
            if (TRANS == 'N') {
                ygemm_serial('N', 'T', jc, jb, k, &alpha,
                             &A_(0, 0), lda, &A_(jc, 0), lda,
                             &ONE, &C_(0, jc), ldc);
            } else {
                ygemm_serial('T', 'N', jc, jb, k, &alpha,
                             &A_(0, 0), lda, &A_(0, jc), lda,
                             &ONE, &C_(0, jc), ldc);
            }
        }
    }
}

void ysyrk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TC *alpha_,
    const TC *a, ptrdiff_t lda,
    const TC *beta_,
    TC *c, ptrdiff_t ldc)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const char TRANS   = blas_up(trans);

    if (n == 0) return;

    if (alpha == ZERO || k == 0) {
        if (beta == ONE) return;
        ysyrk_beta_scale(0, n, n, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = ysyrk_nb();
    for (ptrdiff_t jc = 0; jc < n; jc += nb) {
        const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        ysyrk_block(jc, jb, n, k, alpha, beta, a, lda, c, ldc, UPLO, TRANS);
    }
}

#undef A_
#undef C_
