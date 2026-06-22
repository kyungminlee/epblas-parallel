/* xswap — kind16 complex: swap X ↔ Y. */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __complex128 T;

#ifdef _OPENMP
/* Threaded swap — two quad streams read+write; threading spreads them across
 * cores for memory bandwidth (see qcopy.c). Each iteration is independent.
 * Complex quad is 32B/elem so it leaves cache sooner (crossover ~3K). */
#define XSWAP_OMP_MIN 2048
__attribute__((noinline)) static int xswap_omp(ptrdiff_t n, T *x, ptrdiff_t incx,
                                               T *y, ptrdiff_t incy)
{
    if (n <= XSWAP_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    ptrdiff_t iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (ptrdiff_t i = 0; i < n; ++i) {
        ptrdiff_t ix = ix0 + i * incx, iy = iy0 + i * incy;
        T t = x[ix]; x[ix] = y[iy]; y[iy] = t;
    }
    return 1;
}
#endif

static void xswap_core(ptrdiff_t n, T *x, ptrdiff_t incx, T *y, ptrdiff_t incy)
{
    if (n <= 0) return;
#ifdef _OPENMP
    if (xswap_omp(n, x, incx, y, incy)) return;
#endif
    if (incx == 1 && incy == 1) {
        const ptrdiff_t m = n % 3;
        for (ptrdiff_t i = 0; i < m; ++i) { T t = x[i]; x[i] = y[i]; y[i] = t; }
        for (ptrdiff_t i = m; i < n; i += 3) {
            T t0 = x[i    ]; x[i    ] = y[i    ]; y[i    ] = t0;
            T t1 = x[i + 1]; x[i + 1] = y[i + 1]; y[i + 1] = t1;
            T t2 = x[i + 2]; x[i + 2] = y[i + 2]; y[i + 2] = t2;
        }
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { T t = x[ix]; x[ix] = y[iy]; y[iy] = t; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_SWAP(xswap, T)
