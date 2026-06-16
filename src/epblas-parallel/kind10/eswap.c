#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
/* eswap — kind10 real: swap X ↔ Y. */
typedef long double T;

/* Unit-stride kernel, shared by the serial entry and the per-thread OMP slices.
 * 3-way unrolled to amortize loop overhead over the fp80 load/store pairs. */
static void eswap_unit(ptrdiff_t n, T *x, T *y)
{
    const ptrdiff_t m = n % 3;
    for (ptrdiff_t i = 0; i < m; ++i) { T t = x[i]; x[i] = y[i]; y[i] = t; }
    for (ptrdiff_t i = m; i < n; i += 3) {
        T t0 = x[i    ]; x[i    ] = y[i    ]; y[i    ] = t0;
        T t1 = x[i + 1]; x[i + 1] = y[i + 1]; y[i + 1] = t1;
        T t2 = x[i + 2]; x[i + 2] = y[i + 2]; y[i + 2] = t2;
    }
}

#ifdef _OPENMP
/* Threaded unit-stride SWAP — same cache-bandwidth rationale as eaxpy_omp
 * (see eaxpy.c). swap is the heaviest RMW (2 reads + 2 writes/elem). Threshold
 * is set by par4<=ob4 (ob keeps swap serial at small N, so par's 4-thread time
 * must beat ob's *serial* time). Measured under iomp5: par4/ob4 1.34@1024,
 * 1.12@2048, 1.03@3072, then 0.96@4096 and 0.77@6144 — break-even ~4096, stay
 * serial through 3072. */
#define ESWAP_OMP_MIN 3072
static int eswap_omp(ptrdiff_t n, T *x, T *y)
{
    if (n <= ESWAP_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) eswap_unit(hi - lo, x + lo, y + lo);
    }
    return 1;
}
#endif

void eswap_(const int *n_, T *x, const int *incx_, T *y, const int *incy_)
{
    const ptrdiff_t n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (eswap_omp(n, x, y)) return;
#endif
        eswap_unit(n, x, y);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { T t = x[ix]; x[ix] = y[iy]; y[iy] = t; ix += incx; iy += incy; }
    }
}
