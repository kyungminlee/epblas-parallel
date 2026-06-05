/*
 * qgemm_serial.c — kind16 (REAL(KIND=16) / __float128) GEMM, serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * trans-char decode and the per-tile compute kernel (declared in
 * qgemm_kernel.h), plus the public `qgemm_serial_` Fortran entry. No
 * OpenMP anywhere on this call path — safe to invoke from inside another
 * function's `#pragma omp parallel` region; callers are responsible for
 * partitioning if they want thread parallelism.
 *
 * Both qgemm_serial_ and the parallel qgemm_ drive numerics through
 * qgemm_tile_compute, so the two paths are bitwise-identical.
 */

#include "qgemm_kernel.h"
#include <ctype.h>
#include <quadmath.h>

typedef qgemm_T T;

int qgemm_trans_code(const char *p, size_t len) {
    (void)len;
    char c = (char)toupper((unsigned char)*p);
    return (c == 'C') ? 'T' : c;
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void qgemm_tile_compute(
    int i0, int i1, int j0, int j1, int K,
    int trans_a, int trans_b,
    T alpha, T beta,
    const T *a, int lda,
    const T *b, int ldb,
    T *c, int ldc)
{
    const T zero = 0.0Q;
    const T one  = 1.0Q;

    /* Beta pass. */
    for (int j = j0; j < j1; ++j) {
        T *cj = &C_(0, j);
        if (beta == zero)      for (int i = i0; i < i1; ++i) cj[i]  = zero;
        else if (beta != one)  for (int i = i0; i < i1; ++i) cj[i] *= beta;
    }

    if (alpha == zero || K == 0) return;

    if (!trans_a) {
        /* Rank-1 (axpy) form: TEMP = alpha · op(B)[k,j], then
         * C[:,j] += TEMP · A[:,k]. Hoist trans_b out of the k loop. */
        if (!trans_b) {
            for (int j = j0; j < j1; ++j) {
                T *cj = &C_(0, j);
                for (int k = 0; k < K; ++k) {
                    const T bkj = B_(k, j);
                    if (bkj != zero) {
                        const T t = alpha * bkj;
                        const T *ak = &A_(0, k);
                        for (int i = i0; i < i1; ++i) cj[i] += t * ak[i];
                    }
                }
            }
        } else {
            for (int j = j0; j < j1; ++j) {
                T *cj = &C_(0, j);
                for (int k = 0; k < K; ++k) {
                    const T bjk = B_(j, k);
                    if (bjk != zero) {
                        const T t = alpha * bjk;
                        const T *ak = &A_(0, k);
                        for (int i = i0; i < i1; ++i) cj[i] += t * ak[i];
                    }
                }
            }
        }
    } else {
        /* Inner-product (DDOT) form. */
        if (!trans_b) {
            for (int j = j0; j < j1; ++j) {
                T *cj = &C_(0, j);
                for (int i = i0; i < i1; ++i) {
                    T s = zero;
                    for (int k = 0; k < K; ++k) s += A_(k, i) * B_(k, j);
                    cj[i] += alpha * s;
                }
            }
        } else {
            for (int j = j0; j < j1; ++j) {
                T *cj = &C_(0, j);
                for (int i = i0; i < i1; ++i) {
                    T s = zero;
                    for (int k = 0; k < K; ++k) s += A_(k, i) * B_(j, k);
                    cj[i] += alpha * s;
                }
            }
        }
    }
}

void qgemm_serial_(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t transa_len, size_t transb_len)
{
    const int M = *m_, N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const int ta = qgemm_trans_code(transa, transa_len);
    const int tb = qgemm_trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    const int trans_a = (ta != 'N');
    const int trans_b = (tb != 'N');

    qgemm_tile_compute(0, M, 0, N, K,
                       trans_a, trans_b,
                       alpha, beta, a, lda, b, ldb, c, ldc);
}

#undef A_
#undef B_
#undef C_
