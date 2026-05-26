/*
 * yher2k — kind10 complex (`_Complex long double`) Hermitian rank-2k.
 *
 *   C := alpha · A · Bᴴ + conj(alpha) · B · Aᴴ + beta · C  (TRANS='N')
 *   C := alpha · Aᴴ · B + conj(alpha) · Bᴴ · A + beta · C  (TRANS='C')
 *
 * alpha is COMPLEX, beta is REAL.
 * C is Hermitian; diagonal of C stays real on output.
 * Only the UPLO triangle is touched.
 *
 * Same blocking recipe as yherk: column-panel jc, scalar diag with
 * real-diagonal accumulation, two ygemm trailing calls per
 * off-diagonal block (one for each half of the rank-2 update).
 * nb=32 (complex-side default).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YHER2K_OMP_MIN 32

typedef _Complex long double TC;
typedef long double          TR;

static int g_yher2k_nb = 32;

__attribute__((constructor))
static void yher2k_init(void) {
    const char *s = getenv("YHER2K_NB");
    if (s && *s) {
        int v = atoi(s);
        if (v > 0) g_yher2k_nb = v;
    }
}
static int her2k_nb(void) { return g_yher2k_nb; }

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
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* Scalar Hermitian rank-2 diagonal block. Caller pre-scales β. */
static void her2k_diag_add(int jc, int jb, int K, TC alpha,
                           const TC *restrict a, int lda,
                           const TC *restrict b, int ldb,
                           TC *restrict c, int ldc,
                           char UPLO, char TR_c)
{
    if (TR_c == 'N') {
        /* C(I,J) += alpha · A(I,l)·conj(B(J,l)) + conj(alpha)·B(I,l)·conj(A(J,l)),
         * summed over l. Diagonal (i==j) result is real. */
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            for (int l = 0; l < K; ++l) {
                const TC t1 = alpha           * cconj(B_(j, l));
                const TC t2 = cconj(alpha)    * cconj(A_(j, l));
                for (int i = i_lo; i < i_hi; ++i) {
                    if (i == j) cj[i] += __real__ (A_(i, l) * t1 + B_(i, l) * t2);
                    else        cj[i] += A_(i, l) * t1 + B_(i, l) * t2;
                }
            }
        }
    } else {
        /* C(I,J) += alpha · conj(A(l,I))·B(l,J) + conj(alpha)·conj(B(l,I))·A(l,J) */
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            const TC *Aj = a + (size_t)j * lda;
            const TC *Bj = b + (size_t)j * ldb;
            for (int i = i_lo; i < i_hi; ++i) {
                const TC *Ai = a + (size_t)i * lda;
                const TC *Bi = b + (size_t)i * ldb;
                TC s1 = 0.0L + 0.0Li;
                TC s2 = 0.0L + 0.0Li;
                for (int l = 0; l < K; ++l) {
                    s1 += cconj(Ai[l]) * Bj[l];
                    s2 += cconj(Bi[l]) * Aj[l];
                }
                if (i == j) cj[i] += __real__ (alpha * s1 + cconj(alpha) * s2);
                else        cj[i] += alpha * s1 + cconj(alpha) * s2;
            }
        }
    }
}

void yher2k_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const TC *alpha_,
    const TC *restrict a, const int *lda_,
    const TC *restrict b, const int *ldb_,
    const TR *beta_,
    TC *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const TC alpha = *alpha_;
    const TR beta  = *beta_;
    const char UPLO = up(uplo);
    const char TR_c = up(trans);

    if (N == 0) return;

    const TR rzero = 0.0L, rone = 1.0L;
    const TC czero = 0.0L + 0.0Li;
    const TC cone  = 1.0L + 0.0Li;
    const TC alpha_conj = cconj(alpha);
    const char NN[1] = {'N'};
    const char CN[1] = {'C'};

    if ((alpha == czero) || K == 0) {
        if (beta == rone) {
            /* Keep diagonal real (defensive: caller may have stored
             * something with non-zero imag). */
            for (int j = 0; j < N; ++j) c[(size_t)j * ldc + j] = __real__ c[(size_t)j * ldc + j];
            return;
        }
#ifdef _OPENMP
        const int use_omp = (N >= YHER2K_OMP_MIN && blas_omp_max_threads() > 1);
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

    const int nb = her2k_nb();

#ifdef _OPENMP
    const int use_omp = (N >= YHER2K_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;

        /* β-scale block columns; diag stays real. */
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

        her2k_diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR_c);

        if (UPLO == 'L') {
            const int trailing = N - jc - jb;
            if (trailing > 0) {
                const int j0 = jc + jb;
                if (TR_c == 'N') {
                    /* C(j0:, jc..) += α · A(j0:, :) · B(jc.., :)ᴴ */
                    ygemm_(NN, CN, &trailing, &jb, &K, &alpha,
                           &A_(j0, 0), &lda, &B_(jc, 0), &ldb,
                           &cone, &C_(j0, jc), &ldc, 1, 1);
                    /* C(j0:, jc..) += conj(α) · B(j0:, :) · A(jc.., :)ᴴ */
                    ygemm_(NN, CN, &trailing, &jb, &K, &alpha_conj,
                           &B_(j0, 0), &ldb, &A_(jc, 0), &lda,
                           &cone, &C_(j0, jc), &ldc, 1, 1);
                } else {
                    ygemm_(CN, NN, &trailing, &jb, &K, &alpha,
                           &A_(0, j0), &lda, &B_(0, jc), &ldb,
                           &cone, &C_(j0, jc), &ldc, 1, 1);
                    ygemm_(CN, NN, &trailing, &jb, &K, &alpha_conj,
                           &B_(0, j0), &ldb, &A_(0, jc), &lda,
                           &cone, &C_(j0, jc), &ldc, 1, 1);
                }
            }
        } else {
            if (jc > 0) {
                if (TR_c == 'N') {
                    ygemm_(NN, CN, &jc, &jb, &K, &alpha,
                           &A_(0, 0), &lda, &B_(jc, 0), &ldb,
                           &cone, &C_(0, jc), &ldc, 1, 1);
                    ygemm_(NN, CN, &jc, &jb, &K, &alpha_conj,
                           &B_(0, 0), &ldb, &A_(jc, 0), &lda,
                           &cone, &C_(0, jc), &ldc, 1, 1);
                } else {
                    ygemm_(CN, NN, &jc, &jb, &K, &alpha,
                           &A_(0, 0), &lda, &B_(0, jc), &ldb,
                           &cone, &C_(0, jc), &ldc, 1, 1);
                    ygemm_(CN, NN, &jc, &jb, &K, &alpha_conj,
                           &B_(0, 0), &ldb, &A_(0, jc), &lda,
                           &cone, &C_(0, jc), &ldc, 1, 1);
                }
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
