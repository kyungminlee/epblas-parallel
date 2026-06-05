/* xqscal — kind16: X := α·X with α real __float128, X complex. */
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __complex128 T;
typedef __float128 R;

#ifdef _OPENMP
/* Threaded elementwise real-scale of complex X — quad is compute-bound, so it
 * threads (see qaxpy.c). Index-from-i covers every stride; serial preserved. */
#define XQSCAL_OMP_MIN 128
__attribute__((noinline)) static int xqscal_omp(int n, R alpha, T *x, int incx)
{
    if (n <= XQSCAL_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    int ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; ++i) {
        int ix = ix0 + i * incx;
        __real__ x[ix] *= alpha;
        __imag__ x[ix] *= alpha;
    }
    return 1;
}
#endif

void xqscal_(const int *n_, const R *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const R alpha = *alpha_;
    if (n <= 0 || alpha == 1.0Q) return;
#ifdef _OPENMP
    if (xqscal_omp(n, alpha, x, incx)) return;
#endif
    if (incx == 1) {
        T *end = x + n;
        for (T *p = x; p < end; ++p) {
            __real__ *p *= alpha;
            __imag__ *p *= alpha;
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) {
            __real__ x[ix] *= alpha;
            __imag__ x[ix] *= alpha;
            ix += incx;
        }
    }
}
