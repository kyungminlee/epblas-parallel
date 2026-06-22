/*
 * xsyrk_serial — kind16 complex (__complex128) symmetric rank-k,
 * single-thread. This TU owns ALL of the xsyrk math; xsyrk_parallel.c only
 * orchestrates the diagonal-block loop across a team.
 *
 * TRANS ∈ {N, T}. Complex syrk does NOT conjugate (see xherk). Blocked:
 * scalar diagonal + xgemm trailing. The trailing update runs through
 * xgemm_serial (NOT xgemm_): when xsyrk_block runs inside the team
 * xsyrk_parallel.c opened, a nested xgemm team would open a region inside a
 * region. Mirrors the kind10 ysyrk overlay.
 */

#include "xsyrk_kernel.h"
#include "../common/blas_char.h"
#include "xgemm_kernel.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef xsyrk_T T;

char xsyrk_uplo(char c) {
    return blas_up(c);
}

char xsyrk_trans(char c) {
    return blas_up(c);
}

ptrdiff_t xsyrk_nb(void) { return 32; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

static void syrk_diag_add(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t k, T alpha,
                          const T *restrict a, ptrdiff_t lda,
                          T *restrict c, ptrdiff_t ldc,
                          char UPLO, char TR)
{
    if (TR == 'N') {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            for (ptrdiff_t l = 0; l < k; ++l) {
                const T ajl = A_(j, l);
                if (ajl != ZERO) {
                    const T t = alpha * ajl;
                    for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] += t * A_(i, l);
                }
            }
        }
    } else {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            const T *Aj = a + (size_t)j * lda;
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + (size_t)i * lda;
                T s = ZERO;
                for (ptrdiff_t l = 0; l < k; ++l) s += Ai[l] * Aj[l];
                cj[i] += alpha * s;
            }
        }
    }
}

void xsyrk_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, T beta,
                      T *c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = ZERO;
        else              for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }
}

void xsyrk_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t n, ptrdiff_t k, T alpha, T beta,
                 const T *a, ptrdiff_t lda, T *c, ptrdiff_t ldc, char UPLO, char TR)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }

    syrk_diag_add(jc, jb, k, alpha, a, lda, c, ldc, UPLO, TR);

    if (UPLO == 'L') {
        const ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            const ptrdiff_t j0 = jc + jb;
            if (TR == 'N') {
                xgemm_serial('N', 'T', trailing, jb, k, &alpha,
                             &A_(j0, 0), lda, &A_(jc, 0), lda,
                             &ONE, &C_(j0, jc), ldc);
            } else {
                xgemm_serial('T', 'N', trailing, jb, k, &alpha,
                             &A_(0, j0), lda, &A_(0, jc), lda,
                             &ONE, &C_(j0, jc), ldc);
            }
        }
    } else {
        if (jc > 0) {
            if (TR == 'N') {
                xgemm_serial('N', 'T', jc, jb, k, &alpha,
                             &A_(0, 0), lda, &A_(jc, 0), lda,
                             &ONE, &C_(0, jc), ldc);
            } else {
                xgemm_serial('T', 'N', jc, jb, k, &alpha,
                             &A_(0, 0), lda, &A_(0, jc), lda,
                             &ONE, &C_(0, jc), ldc);
            }
        }
    }
}

void xsyrk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    const T *beta_,
    T *c, ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const char TR   = blas_up(trans);

    if (n == 0) return;

    if (alpha == ZERO || k == 0) {
        if (beta == ONE) return;
        xsyrk_beta_scale(0, n, n, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = xsyrk_nb();
    for (ptrdiff_t jc = 0; jc < n; jc += nb) {
        const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        xsyrk_block(jc, jb, n, k, alpha, beta, a, lda, c, ldc, UPLO, TR);
    }
}

#undef A_
#undef C_
