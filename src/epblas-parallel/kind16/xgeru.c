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

typedef __complex128 T;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xgeru_core(
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *restrict x, ptrdiff_t incx,
    const T *restrict y, ptrdiff_t incy,
    T *restrict a, ptrdiff_t lda)
{
    const T alpha = *alpha_;
    const T zero = 0.0Q + 0.0Qi;

    if (M == 0 || N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (N >= XGERU_OMP_MIN && blas_omp_should_thread());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j) {
            const T yj = y[j];
            if (yj != zero) {
                const T t = alpha * yj;
                T *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < M; ++i) aj[i] += t * x[i];
            }
        }
    } else {
        /* Strided x/y. Columns of A are disjoint → OMP-over-j race-free and
         * bit-exact (jy recomputed as jy0 + j*incy, same as carried add). */
        const ptrdiff_t jy0 = (incy < 0) ? -(N - 1) * incy : 0;
        const ptrdiff_t ix0 = (incx < 0) ? -(M - 1) * incx : 0;
#ifdef _OPENMP
        const bool use_omp = (N >= XGERU_OMP_MIN && blas_omp_should_thread());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j) {
            const T yj = y[jy0 + (ptrdiff_t)j * incy];
            if (yj != zero) {
                const T t = alpha * yj;
                ptrdiff_t ix = ix0;
                T *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < M; ++i) {
                    aj[i] += t * x[ix];
                    ix += incx;
                }
            }
        }
    }
}

#undef A_

EPBLAS_FACADE_GER(xgeru, T)
