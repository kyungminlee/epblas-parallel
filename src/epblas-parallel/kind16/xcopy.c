/* xcopy — kind16 complex: Y := X. */
#include <string.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __complex128 T;

#ifdef _OPENMP
/* Threaded copy — spreads the quad-wide stream across cores for memory
 * bandwidth (see qcopy.c). schedule(static) → contiguous per-thread blocks.
 * Complex quad is 32B/elem so it leaves cache sooner (crossover ~3K). */
#define XCOPY_OMP_MIN 2048
__attribute__((noinline)) static int xcopy_omp(int n, const T *x, int incx,
                                               T *y, int incy)
{
    if (n <= XCOPY_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    int ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    int iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; ++i) y[iy0 + i * incy] = x[ix0 + i * incx];
    return 1;
}
#endif

void xcopy_(const int *n_, const T *x, const int *incx_, T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
#ifdef _OPENMP
    if (xcopy_omp(n, x, incx, y, incy)) return;
#endif
    if (incx == 1 && incy == 1) memcpy(y, x, (size_t)n * sizeof(T));
    else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}
