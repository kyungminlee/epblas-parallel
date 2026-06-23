/*
 * qger — kind16 (__float128) rank-1 update.
 *   A := alpha · x · yᵀ + A
 */

#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define QGER_OMP_MIN 64

typedef __float128 TR;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void qger_core(
    ptrdiff_t m, ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    const TR *restrict y, ptrdiff_t incy,
    TR *restrict a, ptrdiff_t lda)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0Q;

    if (m == 0 || n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (n >= QGER_OMP_MIN && blas_omp_should_thread());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR yj = y[j];
            if (yj != zero) {
                const TR t = alpha * yj;
                TR *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < m; ++i) aj[i] += t * x[i];
            }
        }
    } else {
        /* Strided x/y. Columns of A are disjoint → OMP-over-j race-free and
         * bit-exact (jy recomputed as jy0 + j*incy, same as carried add). */
        const ptrdiff_t jy0 = (incy < 0) ? -(n - 1) * incy : 0;
        const ptrdiff_t ix0 = (incx < 0) ? -(m - 1) * incx : 0;
#ifdef _OPENMP
        const bool use_omp = (n >= QGER_OMP_MIN && blas_omp_should_thread());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR yj = y[jy0 + j * incy];
            if (yj != zero) {
                const TR t = alpha * yj;
                ptrdiff_t ix = ix0;
                TR *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < m; ++i) {
                    aj[i] += t * x[ix];
                    ix += incx;
                }
            }
        }
    }
}


EPBLAS_FACADE_GER(qger, TR)

#undef A_
