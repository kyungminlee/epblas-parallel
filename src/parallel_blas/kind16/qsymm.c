/*
 * qsymm — kind16 (REAL(KIND=16)) symmetric matrix multiply.
 *
 *   C := alpha · A · B + beta · C          (SIDE='L', A is M×M sym)
 *   C := alpha · B · A + beta · C          (SIDE='R', A is N×N sym)
 *
 * UPLO selects which triangle of A is stored. The other half is the
 * reflection (A(i,k) = A(k,i)).
 *
 * Unblocked Netlib reference with omp-parallel-for over columns of C.
 * Each column j is independent — partition N across threads.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QSYMM_OMP_MIN 32

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void qsymm_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t side_len, size_t uplo_len)
{
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = up(side);
    const char UPLO = up(uplo);

    if (M == 0 || N == 0) return;

    const T zero = 0.0Q, one = 1.0Q;

    if (alpha == zero) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp = (N >= QSYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = 0; i < M; ++i) cj[i]  = zero;
            else              for (int i = 0; i < M; ++i) cj[i] *= beta;
        }
        return;
    }

#ifdef _OPENMP
    const int use_omp = (N >= QSYMM_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
        T *cj = c + (size_t)j * ldc;

        /* beta scale C[:,j] up front. */
        if (beta == zero)      for (int i = 0; i < M; ++i) cj[i]  = zero;
        else if (beta != one)  for (int i = 0; i < M; ++i) cj[i] *= beta;

        if (SIDE == 'L') {
            if (UPLO == 'L') {
                /* C(:,j) += alpha · A_sym · B(:,j), A_sym lower stored. */
                for (int i = 0; i < M; ++i) {
                    T temp1 = alpha * B_(i, j);
                    T temp2 = zero;
                    for (int k = 0; k < i; ++k) {
                        cj[k]  += temp1 * A_(i, k);
                        temp2  += B_(k, j) * A_(i, k);
                    }
                    cj[i] += temp1 * A_(i, i) + alpha * temp2;
                }
            } else {  /* UPLO == 'U' */
                for (int i = M - 1; i >= 0; --i) {
                    T temp1 = alpha * B_(i, j);
                    T temp2 = zero;
                    for (int k = i + 1; k < M; ++k) {
                        cj[k]  += temp1 * A_(i, k);
                        temp2  += B_(k, j) * A_(i, k);
                    }
                    cj[i] += temp1 * A_(i, i) + alpha * temp2;
                }
            }
        } else {  /* SIDE = 'R': C(:,j) += alpha · sum_k A_sym(k,j) · B(:,k) */
            /* Diagonal */
            {
                const T t = alpha * A_(j, j);
                for (int i = 0; i < M; ++i) cj[i] += t * B_(i, j);
            }
            if (UPLO == 'L') {
                for (int k = 0; k < j; ++k) {
                    /* A_sym(k,j) = A_stored(j,k) when k<j */
                    const T ajk = A_(j, k);
                    if (ajk != zero) {
                        const T t = alpha * ajk;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
                for (int k = j + 1; k < N; ++k) {
                    /* A_sym(k,j) = A_stored(k,j) when k>j */
                    const T akj = A_(k, j);
                    if (akj != zero) {
                        const T t = alpha * akj;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
            } else {  /* UPLO = 'U' */
                for (int k = 0; k < j; ++k) {
                    /* A_sym(k,j) = A_stored(k,j) when k<j */
                    const T akj = A_(k, j);
                    if (akj != zero) {
                        const T t = alpha * akj;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
                for (int k = j + 1; k < N; ++k) {
                    /* A_sym(k,j) = A_stored(j,k) when k>j */
                    const T ajk = A_(j, k);
                    if (ajk != zero) {
                        const T t = alpha * ajk;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
