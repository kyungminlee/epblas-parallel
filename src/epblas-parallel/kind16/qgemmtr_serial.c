/*
 * qgemmtr_serial.c — kind16 (REAL(KIND=16) / __float128) triangular GEMM
 * update, serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * trans-char decode and the two per-column cores (a beta-only pass and the
 * full compute pass, declared in qgemmtr_kernel.h), plus the public
 * `qgemmtr_serial_` Fortran entry. No OpenMP anywhere on this call path —
 * safe to invoke from inside another function's `#pragma omp parallel` region.
 *
 * Both qgemmtr_serial_ and the parallel qgemmtr_ drive numerics through these
 * cores, so the two paths are bitwise-identical.
 */

#include "qgemmtr_kernel.h"
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>

typedef qgemmtr_T T;

char qgemmtr_trans_code(const char *p) {
    char c = blas_up(*p);
    return (c == 'C') ? 'T' : c;
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void qgemmtr_beta_core(
    ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t N, bool upper,
    T beta,
    T *c, ptrdiff_t ldc)
{
    const T zero = 0.0Q;
    for (ptrdiff_t j = j0; j < j1; ++j) {
        const ptrdiff_t is = upper ? 0 : j;
        const ptrdiff_t ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);
        if (beta == zero)      for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
        else                   for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
    }
}

void qgemmtr_compute_core(
    ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t N, bool upper, ptrdiff_t K,
    char ta, char tb,
    T alpha, T beta,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    T *c, ptrdiff_t ldc)
{
    const T zero = 0.0Q, one = 1.0Q;

    for (ptrdiff_t j = j0; j < j1; ++j) {
        const ptrdiff_t is = upper ? 0 : j;
        const ptrdiff_t ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);

        if (ta == 'N') {
            if (beta == zero)      for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
            else if (beta != one)  for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
            if (tb == 'N') {
                for (ptrdiff_t k = 0; k < K; ++k) {
                    const T bkj = B_(k, j);
                    if (bkj != zero) {
                        const T t = alpha * bkj;
                        const T *ak = &A_(0, k);
                        for (ptrdiff_t i = is; i < ie; ++i) cj[i] += t * ak[i];
                    }
                }
            } else {
                for (ptrdiff_t k = 0; k < K; ++k) {
                    const T bjk = B_(j, k);
                    if (bjk != zero) {
                        const T t = alpha * bjk;
                        const T *ak = &A_(0, k);
                        for (ptrdiff_t i = is; i < ie; ++i) cj[i] += t * ak[i];
                    }
                }
            }
        } else {
            if (tb == 'N') {
                for (ptrdiff_t i = is; i < ie; ++i) {
                    T s = zero;
                    for (ptrdiff_t k = 0; k < K; ++k) s += A_(k, i) * B_(k, j);
                    cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
                }
            } else {
                for (ptrdiff_t i = is; i < ie; ++i) {
                    T s = zero;
                    for (ptrdiff_t k = 0; k < K; ++k) s += A_(k, i) * B_(j, k);
                    cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
                }
            }
        }
    }
}

void qgemmtr_serial(char uplo, char transa, char transb,
                    ptrdiff_t N, ptrdiff_t K,
                    const T *alpha_,
                    const T *a, ptrdiff_t lda,
                    const T *b, ptrdiff_t ldb,
                    const T *beta_,
                    T *c, ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const bool upper = (blas_up(uplo) == 'U');
    const char ta = qgemmtr_trans_code(&transa);
    const char tb = qgemmtr_trans_code(&transb);

    if (N <= 0) return;
    const T zero = 0.0Q, one = 1.0Q;

    if (alpha == zero || K == 0) {
        if (beta == one) return;
        qgemmtr_beta_core(0, N, N, upper, beta, c, ldc);
        return;
    }

    qgemmtr_compute_core(0, N, N, upper, K, ta, tb,
                         alpha, beta, a, lda, b, ldb, c, ldc);
}

#undef A_
#undef B_
#undef C_
