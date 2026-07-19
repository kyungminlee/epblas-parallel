/*
 * qgemmtr_serial.c — kind16 (REAL(KIND=16) / __float128) triangular GEMM
 * update, serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * trans-char decode and the two per-column cores (a beta-only pass and the
 * full compute pass, declared in qgemmtr_kernel.h), plus the public
 * by-value `qgemmtr_serial` entry. No OpenMP anywhere on this call path —
 * safe to invoke from inside another function's `#pragma omp parallel` region.
 *
 * Both qgemmtr_serial and the parallel qgemmtr_ drive numerics through these
 * cores, so the two paths are bitwise-identical.
 */

#include "qgemmtr_kernel.h"
#include "../common/blas_char.h"
#include <ctype.h>
#include <stdbool.h>
#include <quadmath.h>

typedef qgemmtr_TR TR;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void qgemmtr_beta_core(
    ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t n, bool upper,
    TR beta,
    TR *c, ptrdiff_t ldc)
{
    const TR zero = 0.0Q;
    for (ptrdiff_t j = j0; j < j1; ++j) {
        const ptrdiff_t is = upper ? 0 : j;
        const ptrdiff_t ie = upper ? (j + 1) : n;
        TR *cj = &C_(0, j);
        if (beta == zero)      for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
        else                   for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
    }
}

void qgemmtr_compute_core(
    ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t n, bool upper, ptrdiff_t k,
    char ta, char tb,
    TR alpha, TR beta,
    const TR *a, ptrdiff_t lda,
    const TR *b, ptrdiff_t ldb,
    TR *c, ptrdiff_t ldc)
{
    const TR zero = 0.0Q, one = 1.0Q;

    for (ptrdiff_t j = j0; j < j1; ++j) {
        const ptrdiff_t is = upper ? 0 : j;
        const ptrdiff_t ie = upper ? (j + 1) : n;
        TR *cj = &C_(0, j);

        if (ta == 'N') {
            if (beta == zero)      for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
            else if (beta != one)  for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
            if (tb == 'N') {
                for (ptrdiff_t p = 0; p < k; ++p) {
                    const TR bkj = B_(p, j);
                    if (bkj != zero) {
                        const TR t = alpha * bkj;
                        const TR *ak = &A_(0, p);
                        for (ptrdiff_t i = is; i < ie; ++i) cj[i] += t * ak[i];
                    }
                }
            } else {
                for (ptrdiff_t p = 0; p < k; ++p) {
                    const TR bjk = B_(j, p);
                    if (bjk != zero) {
                        const TR t = alpha * bjk;
                        const TR *ak = &A_(0, p);
                        for (ptrdiff_t i = is; i < ie; ++i) cj[i] += t * ak[i];
                    }
                }
            }
        } else {
            if (tb == 'N') {
                for (ptrdiff_t i = is; i < ie; ++i) {
                    TR s = zero;
                    for (ptrdiff_t p = 0; p < k; ++p) s += A_(p, i) * B_(p, j);
                    cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
                }
            } else {
                for (ptrdiff_t i = is; i < ie; ++i) {
                    TR s = zero;
                    for (ptrdiff_t p = 0; p < k; ++p) s += A_(p, i) * B_(j, p);
                    cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
                }
            }
        }
    }
}

void qgemmtr_serial(char uplo, char transa, char transb,
                    ptrdiff_t n, ptrdiff_t k,
                    const TR *alpha_,
                    const TR *a, ptrdiff_t lda,
                    const TR *b, ptrdiff_t ldb,
                    const TR *beta_,
                    TR *c, ptrdiff_t ldc)
{
    const TR alpha = *alpha_, beta = *beta_;
    const bool upper = (blas_up(uplo) == 'U');
    const char ta = blas_trans_real(transa);
    const char tb = blas_trans_real(transb);

    if (n <= 0) return;
    const TR zero = 0.0Q, one = 1.0Q;

    if (alpha == zero || k == 0) {
        if (beta == one) return;
        qgemmtr_beta_core(0, n, n, upper, beta, c, ldc);
        return;
    }

    qgemmtr_compute_core(0, n, n, upper, k, ta, tb,
                         alpha, beta, a, lda, b, ldb, c, ldc);
}

#undef A_
#undef B_
#undef C_
