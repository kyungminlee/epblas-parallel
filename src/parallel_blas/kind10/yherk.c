/*
 * yherk — kind10 complex (`_Complex long double`) Hermitian rank-k.
 *   C := alpha · A · Aᴴ + beta · C          (TRANS='N')
 *   C := alpha · Aᴴ · A + beta · C          (TRANS='C')
 *
 * alpha and beta are REAL. Diagonal of C stays real on output.
 * Blocked: scalar diagonal-block kernel (handles diagonal-real
 * accumulation) + ygemm trailing with conjugate transpose.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YHERK_OMP_MIN 32

typedef _Complex long double TC;
typedef long double          TR;

static int g_yherk_nb = 32;

__attribute__((constructor))
static void yherk_init(void) {
    const char *s = getenv("YHERK_NB");
    if (s && *s) {
        int v = atoi(s);
        if (v > 0) g_yherk_nb = v;
    }
}
static int herk_nb(void) { return g_yherk_nb; }

extern void ygemm_(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const TC *alpha,
    const TC *a, const int *lda,
    const TC *b, const int *ldb,
    const TC *beta,
    TC *c, const int *ldc,
    size_t transa_len, size_t transb_len);

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static inline TC cconj(TC z) { return ~z; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* Diagonal jb×jb block rank-k add, keeping diagonal entries real.
 * No beta scaling (caller pre-scales). */
static void herk_diag_add(int jc, int jb, int K, TR alpha,
                          const TC *restrict a, int lda,
                          TC *restrict c, int ldc,
                          char UPLO, char TR_c)
{
    if (TR_c == 'N') {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            for (int l = 0; l < K; ++l) {
                const TC ajl = A_(j, l);
                if (ajl != (TC)0.0L) {
                    const TC t = alpha * cconj(ajl);
                    for (int i = i_lo; i < i_hi; ++i) {
                        if (i == j) cj[i] += __real__ (t * A_(i, l));
                        else        cj[i] += t * A_(i, l);
                    }
                }
            }
        }
    } else {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            const TC *Aj = a + (size_t)j * lda;
            for (int i = i_lo; i < i_hi; ++i) {
                const TC *Ai = a + (size_t)i * lda;
                TC s = 0.0L + 0.0Li;
                for (int l = 0; l < K; ++l) s += cconj(Ai[l]) * Aj[l];
                if (i == j) cj[i] += alpha * __real__ s;
                else        cj[i] += alpha * s;
            }
        }
    }
}

void yherk_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const TR *alpha_,
    const TC *restrict a, const int *lda_,
    const TR *beta_,
    TC *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    const char TR_c = up(trans);

    if (N == 0) return;

    const TR rzero = 0.0L, rone = 1.0L;
    const TC czero = 0.0L + 0.0Li;
    const TC cone  = 1.0L + 0.0Li;
    const TC alpha_c = alpha + 0.0Li;
    const char NN[1] = {'N'};
    const char CN[1] = {'C'};

    if (alpha == rzero || K == 0) {
        if (beta == rone) {
            for (int j = 0; j < N; ++j) c[(size_t)j * ldc + j] = __real__ c[(size_t)j * ldc + j];
            return;
        }
#ifdef _OPENMP
        const int use_omp = (N >= YHERK_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            TC *cj = c + (size_t)j * ldc;
            if (beta == rzero) for (int i = i_lo; i < i_hi; ++i) cj[i] = czero;
            else {
                for (int i = i_lo; i < i_hi; ++i) {
                    if (i == j) cj[i] = beta * __real__ cj[i];
                    else        cj[i] = beta * cj[i];
                }
            }
        }
        return;
    }

    const int nb = herk_nb();

#ifdef _OPENMP
    const int use_omp = (N >= YHERK_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;

        /* Beta-scale block columns: diag stays real. */
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            TC *cj = c + (size_t)j * ldc;
            if (beta == rzero) {
                for (int i = i_lo; i < i_hi; ++i) cj[i] = czero;
            } else if (beta != rone) {
                for (int i = i_lo; i < i_hi; ++i) {
                    if (i == j) cj[i] = beta * __real__ cj[i];
                    else        cj[i] = beta * cj[i];
                }
            } else {
                cj[j] = __real__ cj[j];
            }
        }

        herk_diag_add(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR_c);

        if (UPLO == 'L') {
            const int trailing = N - jc - jb;
            if (trailing > 0) {
                const int j0 = jc + jb;
                if (TR_c == 'N') {
                    /* C(j0..N, jc..jc+jb) += alpha · A(j0..N, :) · A(jc..jc+jb, :)ᴴ */
                    ygemm_(NN, CN, &trailing, &jb, &K, &alpha_c,
                           &A_(j0, 0), &lda, &A_(jc, 0), &lda,
                           &cone, &C_(j0, jc), &ldc, 1, 1);
                } else {
                    /* C(j0..N, jc..jc+jb) += alpha · A(:, j0..N)ᴴ · A(:, jc..jc+jb) */
                    ygemm_(CN, NN, &trailing, &jb, &K, &alpha_c,
                           &A_(0, j0), &lda, &A_(0, jc), &lda,
                           &cone, &C_(j0, jc), &ldc, 1, 1);
                }
            }
        } else {
            if (jc > 0) {
                if (TR_c == 'N') {
                    ygemm_(NN, CN, &jc, &jb, &K, &alpha_c,
                           &A_(0, 0), &lda, &A_(jc, 0), &lda,
                           &cone, &C_(0, jc), &ldc, 1, 1);
                } else {
                    ygemm_(CN, NN, &jc, &jb, &K, &alpha_c,
                           &A_(0, 0), &lda, &A_(0, jc), &lda,
                           &cone, &C_(0, jc), &ldc, 1, 1);
                }
            }
        }
    }
}

#undef A_
#undef C_
