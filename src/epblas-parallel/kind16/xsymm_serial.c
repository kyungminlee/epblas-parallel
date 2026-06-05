/*
 * xsymm_serial.c — kind16 complex (`__complex128`) symmetric matrix multiply,
 * serial core. NOT Hermitian — no conjugate.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * uplo-char decode and the per-column compute core (declared in
 * xsymm_kernel.h), plus the public `xsymm_serial_` Fortran entry. No OpenMP
 * anywhere on this call path — safe to invoke from inside another function's
 * `#pragma omp parallel` region.
 *
 * Both xsymm_serial_ and the parallel xsymm_ drive numerics through
 * xsymm_core, so the two paths are bitwise-identical.
 */

#include "xsymm_kernel.h"
#include <ctype.h>
#include <quadmath.h>

typedef xsymm_T T;

char xsymm_uplo(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void xsymm_core(
    int j0, int j1,
    char SIDE, char UPLO,
    int M, int N,
    T alpha, T beta,
    const T *a, int lda,
    const T *b, int ldb,
    T *c, int ldc)
{
    for (int j = j0; j < j1; ++j) {
        T *cj = c + (size_t)j * ldc;

        if (beta == ZERO)      for (int i = 0; i < M; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (int i = 0; i < M; ++i) cj[i] *= beta;

        if (SIDE == 'L') {
            if (UPLO == 'L') {
                for (int i = 0; i < M; ++i) {
                    T temp1 = alpha * B_(i, j);
                    T temp2 = ZERO;
                    for (int k = 0; k < i; ++k) {
                        cj[k]  += temp1 * A_(i, k);
                        temp2  += B_(k, j) * A_(i, k);
                    }
                    cj[i] += temp1 * A_(i, i) + alpha * temp2;
                }
            } else {
                for (int i = M - 1; i >= 0; --i) {
                    T temp1 = alpha * B_(i, j);
                    T temp2 = ZERO;
                    for (int k = i + 1; k < M; ++k) {
                        cj[k]  += temp1 * A_(i, k);
                        temp2  += B_(k, j) * A_(i, k);
                    }
                    cj[i] += temp1 * A_(i, i) + alpha * temp2;
                }
            }
        } else {  /* SIDE = 'R' */
            {
                const T t = alpha * A_(j, j);
                for (int i = 0; i < M; ++i) cj[i] += t * B_(i, j);
            }
            if (UPLO == 'L') {
                for (int k = 0; k < j; ++k) {
                    const T ajk = A_(j, k);
                    if (ajk != ZERO) {
                        const T t = alpha * ajk;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
                for (int k = j + 1; k < N; ++k) {
                    const T akj = A_(k, j);
                    if (akj != ZERO) {
                        const T t = alpha * akj;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
            } else {
                for (int k = 0; k < j; ++k) {
                    const T akj = A_(k, j);
                    if (akj != ZERO) {
                        const T t = alpha * akj;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
                for (int k = j + 1; k < N; ++k) {
                    const T ajk = A_(j, k);
                    if (ajk != ZERO) {
                        const T t = alpha * ajk;
                        for (int i = 0; i < M; ++i) cj[i] += t * B_(i, k);
                    }
                }
            }
        }
    }
}

void xsymm_serial_(
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
    const char SIDE = xsymm_uplo(side);
    const char UPLO = xsymm_uplo(uplo);

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
        for (int j = 0; j < N; ++j) {
            T *cj = c + (size_t)j * ldc;
            if (beta == ZERO) for (int i = 0; i < M; ++i) cj[i] = ZERO;
            else              for (int i = 0; i < M; ++i) cj[i] *= beta;
        }
        return;
    }

    xsymm_core(0, N, SIDE, UPLO, M, N, alpha, beta, a, lda, b, ldb, c, ldc);
}

#undef A_
#undef B_
#undef C_
