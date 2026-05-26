/*
 * ysyr2k — kind10 complex (`_Complex long double`) symmetric rank-2k.
 *
 *   C := alpha · (A · Bᵀ + B · Aᵀ) + beta · C         (TRANS='N')
 *   C := alpha · (Aᵀ · B + Bᵀ · A) + beta · C         (TRANS='T')
 *
 * C is N×N complex symmetric (NOT Hermitian — see yher2k for that).
 * Only the UPLO triangle is touched. TRANS='C' is rejected by BLAS
 * spec but we accept it as alias for 'T' for safety.
 *
 * Same recipe as esyr2k: column-panel blocked, scalar rank-2k diag,
 * two ygemm trailing calls per off-diagonal block. nb=32 to match
 * the ysyrk/ysymm complex-side choice.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YSYR2K_OMP_MIN 32

typedef _Complex long double T;

static int g_ysyr2k_nb = 32;

__attribute__((constructor))
static void ysyr2k_init(void) {
    const char *s = getenv("YSYR2K_NB");
    if (s && *s) {
        int v = atoi(s);
        if (v > 0) g_ysyr2k_nb = v;
    }
}
static int syr2k_nb(void) { return g_ysyr2k_nb; }

extern void ygemm_(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static void syr2k_diag_add(int jc, int jb, int K, T alpha,
                           const T *restrict a, int lda,
                           const T *restrict b, int ldb,
                           T *restrict c, int ldc,
                           char UPLO, char TR)
{
    if (TR == 'N') {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            for (int l = 0; l < K; ++l) {
                const T t1 = alpha * A_(j, l);
                const T t2 = alpha * B_(j, l);
                for (int i = i_lo; i < i_hi; ++i) {
                    cj[i] += B_(i, l) * t1 + A_(i, l) * t2;
                }
            }
        }
    } else {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            const T *Aj = a + (size_t)j * lda;
            const T *Bj = b + (size_t)j * ldb;
            for (int i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + (size_t)i * lda;
                const T *Bi = b + (size_t)i * ldb;
                T s = ZERO;
                for (int l = 0; l < K; ++l) s += Ai[l] * Bj[l] + Bi[l] * Aj[l];
                cj[i] += alpha * s;
            }
        }
    }
}

void ysyr2k_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict b, const int *ldb_,
    const T *beta_,
    T *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    if (alpha == ZERO || K == 0) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const int use_omp = (N >= YSYR2K_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            if (beta == ZERO) for (int i = i_lo; i < i_hi; ++i) cj[i] = ZERO;
            else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
        }
        return;
    }

    const int nb = syr2k_nb();

#ifdef _OPENMP
    const int use_omp = (N >= YSYR2K_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;

        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            if (beta == ZERO)      for (int i = i_lo; i < i_hi; ++i) cj[i]  = ZERO;
            else if (beta != ONE)  for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
        }

        syr2k_diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR);

        if (UPLO == 'L') {
            const int trailing = N - jc - jb;
            if (trailing > 0) {
                const int j0 = jc + jb;
                if (TR == 'N') {
                    ygemm_(NN, TN, &trailing, &jb, &K, &alpha,
                           &A_(j0, 0), &lda, &B_(jc, 0), &ldb, &ONE,
                           &C_(j0, jc), &ldc, 1, 1);
                    ygemm_(NN, TN, &trailing, &jb, &K, &alpha,
                           &B_(j0, 0), &ldb, &A_(jc, 0), &lda, &ONE,
                           &C_(j0, jc), &ldc, 1, 1);
                } else {
                    ygemm_(TN, NN, &trailing, &jb, &K, &alpha,
                           &A_(0, j0), &lda, &B_(0, jc), &ldb, &ONE,
                           &C_(j0, jc), &ldc, 1, 1);
                    ygemm_(TN, NN, &trailing, &jb, &K, &alpha,
                           &B_(0, j0), &ldb, &A_(0, jc), &lda, &ONE,
                           &C_(j0, jc), &ldc, 1, 1);
                }
            }
        } else {
            if (jc > 0) {
                if (TR == 'N') {
                    ygemm_(NN, TN, &jc, &jb, &K, &alpha,
                           &A_(0, 0), &lda, &B_(jc, 0), &ldb, &ONE,
                           &C_(0, jc), &ldc, 1, 1);
                    ygemm_(NN, TN, &jc, &jb, &K, &alpha,
                           &B_(0, 0), &ldb, &A_(jc, 0), &lda, &ONE,
                           &C_(0, jc), &ldc, 1, 1);
                } else {
                    ygemm_(TN, NN, &jc, &jb, &K, &alpha,
                           &A_(0, 0), &lda, &B_(0, jc), &ldb, &ONE,
                           &C_(0, jc), &ldc, 1, 1);
                    ygemm_(TN, NN, &jc, &jb, &K, &alpha,
                           &B_(0, 0), &ldb, &A_(0, jc), &lda, &ONE,
                           &C_(0, jc), &ldc, 1, 1);
                }
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
