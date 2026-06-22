/*
 * xgemmtr_serial.c — kind16 (COMPLEX(KIND=16) / __complex128) triangular GEMM
 * update, serial core.
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the
 * trans-char decode and the two per-column cores (a beta-only pass and the
 * full compute pass, declared in xgemmtr_kernel.h), plus the public
 * `xgemmtr_serial_` Fortran entry. No OpenMP anywhere on this call path —
 * safe to invoke from inside another function's `#pragma omp parallel` region.
 *
 * Both xgemmtr_serial_ and the parallel xgemmtr_ drive numerics through these
 * cores, so the two paths are bitwise-identical.
 */

#include "xgemmtr_kernel.h"
#include "../common/blas_char.h"
#include <ctype.h>
#include <stddef.h>
#include <quadmath.h>

typedef xgemmtr_T T;

char xgemmtr_trans_code(char c) {
    return blas_up(c);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

void xgemmtr_beta_core(
    ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t N, bool upper,
    T beta,
    T *c, ptrdiff_t ldc)
{
    const T zero = 0.0Q + 0.0Qi;
    for (ptrdiff_t j = j0; j < j1; ++j) {
        const ptrdiff_t is = upper ? 0 : j;
        const ptrdiff_t ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);
        if (beta == zero)      for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
        else                   for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
    }
}

void xgemmtr_compute_core(
    ptrdiff_t j0, ptrdiff_t j1, ptrdiff_t N, bool upper, ptrdiff_t K,
    bool trans_a, bool trans_b, bool conj_a, bool conj_b,
    T alpha, T beta,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    T *c, ptrdiff_t ldc)
{
    const T zero = 0.0Q + 0.0Qi;
    const T one  = 1.0Q + 0.0Qi;

    for (ptrdiff_t j = j0; j < j1; ++j) {
        const ptrdiff_t is = upper ? 0 : j;
        const ptrdiff_t ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);

        if (!trans_a) {
            /* axpy form */
            if (beta == zero)      for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
            else if (beta != one)  for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
            for (ptrdiff_t l = 0; l < K; ++l) {
                T bl;
                if (!trans_b)      bl = B_(l, j);
                else if (!conj_b)  bl = B_(j, l);
                else               bl = conjq(B_(j, l));
                if (bl != zero) {
                    const T t = alpha * bl;
                    const T *al = &A_(0, l);
                    for (ptrdiff_t i = is; i < ie; ++i) cj[i] += t * al[i];
                }
            }
        } else {
            /* inner-product form */
            for (ptrdiff_t i = is; i < ie; ++i) {
                T s = zero;
                if (!trans_b) {
                    if (!conj_a) for (ptrdiff_t l = 0; l < K; ++l) s += A_(l, i)        * B_(l, j);
                    else         for (ptrdiff_t l = 0; l < K; ++l) s += conjq(A_(l, i)) * B_(l, j);
                } else if (!conj_b) {
                    if (!conj_a) for (ptrdiff_t l = 0; l < K; ++l) s += A_(l, i)        * B_(j, l);
                    else         for (ptrdiff_t l = 0; l < K; ++l) s += conjq(A_(l, i)) * B_(j, l);
                } else {
                    if (!conj_a) for (ptrdiff_t l = 0; l < K; ++l) s += A_(l, i)        * conjq(B_(j, l));
                    else         for (ptrdiff_t l = 0; l < K; ++l) s += conjq(A_(l, i)) * conjq(B_(j, l));
                }
                cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
            }
        }
    }
}

void xgemmtr_serial(char uplo, char transa, char transb,
                    ptrdiff_t N, ptrdiff_t K,
                    const T *alpha_,
                    const T *a, ptrdiff_t lda,
                    const T *b, ptrdiff_t ldb,
                    const T *beta_,
                    T *c, ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const bool upper = (blas_up(uplo) == 'U');
    const char ta = xgemmtr_trans_code(transa);
    const char tb = xgemmtr_trans_code(transb);

    if (N <= 0) return;
    const T zero = 0.0Q + 0.0Qi;
    const T one  = 1.0Q + 0.0Qi;

    const bool conj_a = (ta == 'C');
    const bool conj_b = (tb == 'C');
    const bool trans_a = (ta != 'N');
    const bool trans_b = (tb != 'N');

    if (alpha == zero || K == 0) {
        if (beta == one) return;
        xgemmtr_beta_core(0, N, N, upper, beta, c, ldc);
        return;
    }

    xgemmtr_compute_core(0, N, N, upper, K,
                         trans_a, trans_b, conj_a, conj_b,
                         alpha, beta, a, lda, b, ldb, c, ldc);
}

#undef A_
#undef B_
#undef C_
