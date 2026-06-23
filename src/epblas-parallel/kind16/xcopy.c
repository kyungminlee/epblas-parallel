/* xcopy — kind16 complex: Y := X. */
#include <stddef.h>
#include <string.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __complex128 TC;

#ifdef _OPENMP
/* Threaded copy — spreads the quad-wide stream across cores for memory
 * bandwidth (see qcopy.c). schedule(static) → contiguous per-thread blocks.
 * Complex quad is 32B/elem so it leaves cache sooner (crossover ~3K). */
#define XCOPY_OMP_MIN 2048
__attribute__((noinline)) static bool xcopy_omp(ptrdiff_t n, const TC *x, ptrdiff_t incx,
                                               TC *y, ptrdiff_t incy)
{
    if (n <= XCOPY_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    ptrdiff_t iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (ptrdiff_t i = 0; i < n; ++i) y[iy0 + i * incy] = x[ix0 + i * incx];
    return 1;
}
#endif

static void xcopy_core(ptrdiff_t n, const TC *x, ptrdiff_t incx, TC *y, ptrdiff_t incy)
{
    if (n <= 0) return;
#ifdef _OPENMP
    if (xcopy_omp(n, x, incx, y, incy)) return;
#endif
    if (incx == 1 && incy == 1) memcpy(y, x, (size_t)n * sizeof(TC));
    else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_COPY(xcopy, TC)
