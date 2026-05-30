/*
 * ysyrk_serial — kind10 complex (_Complex long double) symmetric rank-k,
 * single-thread. This TU owns ALL of the ysyrk math; ysyrk_parallel.c only
 * orchestrates the diagonal-block loop across a team.
 *
 * TRANS ∈ {N, T}. Complex syrk does NOT conjugate (see yherk). Blocked:
 * scalar diagonal + ygemm trailing. The trailing update runs through
 * ygemm_serial (NOT ygemm_): when ysyrk_block runs inside the team
 * ysyrk_parallel.c opened, a nested ygemm team would trip the libgomp
 * barrier wedge (memory project-etrsm-omp4-wedge).
 */

#include "ysyrk_kernel.h"
#include <stdlib.h>
#include <ctype.h>

typedef ysyrk_T T;

static int g_ysyrk_nb = 0;
int ysyrk_nb(void) {
    if (g_ysyrk_nb == 0) {
        g_ysyrk_nb = 32;
        const char *s = getenv("YSYRK_NB");
        if (s && *s) { int v = atoi(s); if (v > 0) g_ysyrk_nb = v; }
    }
    return g_ysyrk_nb;
}

extern void ygemm_serial(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

static void syrk_diag_add(int jc, int jb, int K, T alpha,
                          const T *restrict a, int lda,
                          T *restrict c, int ldc,
                          char UPLO, char TR)
{
    if (TR == 'N') {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            for (int l = 0; l < K; ++l) {
                const T ajl = A_(j, l);
                if (ajl != ZERO) {
                    const T t = alpha * ajl;
                    for (int i = i_lo; i < i_hi; ++i) cj[i] += t * A_(i, l);
                }
            }
        }
    } else {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            const T *Aj = a + (size_t)j * lda;
            for (int i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + (size_t)i * lda;
                T s = ZERO;
                for (int l = 0; l < K; ++l) s += Ai[l] * Aj[l];
                cj[i] += alpha * s;
            }
        }
    }
}

void ysyrk_beta_scale(int j_start, int j_end, int N, T beta,
                      T *c, int ldc, char UPLO)
{
    for (int j = j_start; j < j_end; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (int i = i_lo; i < i_hi; ++i) cj[i] = ZERO;
        else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }
}

void ysyrk_block(int jc, int jb, int N, int K, T alpha, T beta,
                 const T *a, int lda, T *c, int ldc, char UPLO, char TR)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    for (int j = jc; j < jc + jb; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (int i = i_lo; i < i_hi; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }

    syrk_diag_add(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);

    if (UPLO == 'L') {
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            const int j0 = jc + jb;
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
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const char TR   = (char)toupper((unsigned char)*trans);

    if (N == 0) return;

    if (alpha == ZERO || K == 0) {
        if (beta == ONE) return;
        ysyrk_beta_scale(0, N, N, beta, c, ldc, UPLO);
        return;
    }

    const int nb = ysyrk_nb();
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        ysyrk_block(jc, jb, N, K, alpha, beta, a, lda, c, ldc, UPLO, TR);
    }
}

#undef A_
#undef C_
