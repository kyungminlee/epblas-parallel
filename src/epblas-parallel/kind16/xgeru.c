/*
 * xgeru — kind16 complex unconjugated rank-1.
 *   A := alpha · x · yᵀ + A
 */

#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define XGERU_OMP_MIN 64

typedef __complex128 TC;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xgeru_core(
    ptrdiff_t m, ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    const TC *restrict y, ptrdiff_t incy,
    TC *restrict a, ptrdiff_t lda)
{
    const TC alpha = *alpha_;
    const TC zero = 0.0Q + 0.0Qi;

    if (m == 0 || n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (n >= XGERU_OMP_MIN && blas_omp_should_thread());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TC yj = y[j];
            if (yj != zero) {
                const TC t = alpha * yj;
                TC *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < m; ++i) aj[i] += t * x[i];
            }
        }
    } else {
        /* Strided x/y. Columns of A are disjoint → OMP-over-j race-free and
         * bit-exact (jy recomputed as jy0 + j*incy, same as carried add). */
        const ptrdiff_t jy0 = (incy < 0) ? -(n - 1) * incy : 0;
        const ptrdiff_t ix0 = (incx < 0) ? -(m - 1) * incx : 0;
#ifdef _OPENMP
        const bool use_omp = (n >= XGERU_OMP_MIN && blas_omp_should_thread());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TC yj = y[jy0 + (ptrdiff_t)j * incy];
            if (yj != zero) {
                const TC t = alpha * yj;
                ptrdiff_t ix = ix0;
                TC *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < m; ++i) {
                    aj[i] += t * x[ix];
                    ix += incx;
                }
            }
        }
    }
}

#undef A_

EPBLAS_FACADE_GER(xgeru, TC)
