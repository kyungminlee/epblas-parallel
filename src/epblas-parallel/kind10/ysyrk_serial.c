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
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef ysyrk_T T;

ptrdiff_t ysyrk_nb(void) { return 32; }

extern void ygemm_serial(
    const char *transa, const char *transb,
    const ptrdiff_t *m, const ptrdiff_t *n, const ptrdiff_t *k,
    const T *alpha,
    const T *a, const ptrdiff_t *lda,
    const T *b, const ptrdiff_t *ldb,
    const T *beta,
    T *c, const ptrdiff_t *ldc,
    size_t transa_len, size_t transb_len);

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

static void syrk_diag_add(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t K, T alpha,
                          const T *restrict a, ptrdiff_t lda,
                          T *restrict c, ptrdiff_t ldc,
                          char UPLO, char TR)
{
    if (TR == 'N') {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            for (ptrdiff_t l = 0; l < K; ++l) {
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
                for (ptrdiff_t l = 0; l < K; ++l) s += Ai[l] * Aj[l];
                cj[i] += alpha * s;
            }
        }
    }
}

void ysyrk_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t N, T beta,
                      T *c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = ZERO;
        else              for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }
}

void ysyrk_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t N, ptrdiff_t K, T alpha, T beta,
                 const T *a, ptrdiff_t lda, T *c, ptrdiff_t ldc, char UPLO, char TR)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }

    syrk_diag_add(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);

    if (UPLO == 'L') {
        const ptrdiff_t trailing = N - jc - jb;
        if (trailing > 0) {
            const ptrdiff_t j0 = jc + jb;
            if (TR == 'N') {
                ygemm_serial(NN, TN, &trailing, &jb, &K, &alpha,
                             &A_(j0, 0), &lda, &A_(jc, 0), &lda,
                             &ONE, &C_(j0, jc), &ldc, 1, 1);
            } else {
                ygemm_serial(TN, NN, &trailing, &jb, &K, &alpha,
                             &A_(0, j0), &lda, &A_(0, jc), &lda,
                             &ONE, &C_(j0, jc), &ldc, 1, 1);
            }
        }
    } else {
        if (jc > 0) {
            if (TR == 'N') {
                ygemm_serial(NN, TN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &A_(jc, 0), &lda,
                             &ONE, &C_(0, jc), &ldc, 1, 1);
            } else {
                ygemm_serial(TN, NN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &A_(0, jc), &lda,
                             &ONE, &C_(0, jc), &ldc, 1, 1);
            }
        }
    }
}

void ysyrk_serial(
    const char *uplo, const char *trans,
    const ptrdiff_t *n_, const ptrdiff_t *k_,
    const T *alpha_,
    const T *a, const ptrdiff_t *lda_,
    const T *beta_,
    T *c, const ptrdiff_t *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const ptrdiff_t N = *n_, K = *k_;
    const ptrdiff_t lda = *lda_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const char TR   = (char)toupper((unsigned char)*trans);

    if (N == 0) return;

    if (alpha == ZERO || K == 0) {
        if (beta == ONE) return;
        ysyrk_beta_scale(0, N, N, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = ysyrk_nb();
    for (ptrdiff_t jc = 0; jc < N; jc += nb) {
        const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        ysyrk_block(jc, jb, N, K, alpha, beta, a, lda, c, ldc, UPLO, TR);
    }
}

#undef A_
#undef C_
