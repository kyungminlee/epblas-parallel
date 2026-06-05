/*
 * qsymm_serial.c — kind16 (REAL(KIND=16) / __float128) symmetric matrix
 * multiply, serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * uplo-char decode and the per-column compute core (declared in
 * qsymm_kernel.h), plus the public `qsymm_serial_` Fortran entry. No OpenMP
 * anywhere on this call path — safe to invoke from inside another function's
 * `#pragma omp parallel` region.
 *
 * Both qsymm_serial_ and the parallel qsymm_ drive numerics through
 * qsymm_core, so the two paths are bitwise-identical.
 */

#include "qsymm_kernel.h"
#include <ctype.h>
#include <quadmath.h>

typedef qsymm_T T;

char qsymm_uplo(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void qsymm_core(
    int j0, int j1,
    char SIDE, char UPLO,
    int M, int N,
    T alpha, T beta,
    const T *a, int lda,
    const T *b, int ldb,
    T *c, int ldc)
{
    const T zero = 0.0Q, one = 1.0Q;

    for (int j = j0; j < j1; ++j) {
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

void qsymm_serial_(
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
    const char SIDE = qsymm_uplo(side);
    const char UPLO = qsymm_uplo(uplo);

    if (M == 0 || N == 0) return;

    const T zero = 0.0Q, one = 1.0Q;

    if (alpha == zero) {
        if (beta == one) return;
        for (int j = 0; j < N; ++j) {
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = 0; i < M; ++i) cj[i]  = zero;
            else              for (int i = 0; i < M; ++i) cj[i] *= beta;
        }
        return;
    }

    qsymm_core(0, N, SIDE, UPLO, M, N, alpha, beta, a, lda, b, ldb, c, ldc);
}

#undef A_
#undef B_
#undef C_
