#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
/* yaxpy — kind10 complex: Y := α·X + Y. */
typedef _Complex long double T;

/* Unit-stride kernel, shared by the serial entry and the per-thread OMP slices.
 * Pointer-walking form (the gfortran ZAXPY shape) keeps the loop tight. */
static void yaxpy_unit(ptrdiff_t n, T alpha, const T *x, T *y)
{
    const T *xe = x + n;
    T *yp = y;
    for (const T *xp = x; xp < xe; ++xp, ++yp) *yp += alpha * (*xp);
}

#ifdef _OPENMP
/* Threaded unit-stride complex AXPY — same cache-bandwidth rationale as eaxpy_omp
 * (see eaxpy.c). A complex element is 2x the bytes of a real one, so the cache
 * regime (and the break-even) shifts to ~half the element count: measured
 * proto4/par1 ~0.92 at N=192, ~0.76 at 256, and <1.0 out to 4M (~0.65), so no
 * upper bound. Break-even ~N=180; 256 keeps margin. */
#define YAXPY_OMP_MIN 256
static int yaxpy_omp(ptrdiff_t n, T alpha, const T *x, T *y)
{
    if (n <= YAXPY_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) yaxpy_unit(hi - lo, alpha, x + lo, y + lo);
    }
    return 1;
}
#endif

static void yaxpy_core(ptrdiff_t n, const T *alpha_,
                       const T *x, ptrdiff_t incx,
                       T *y, ptrdiff_t incy)
{
    const T alpha = *alpha_;
    if (n <= 0) return;
    if (alpha == (T)0.0L) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (yaxpy_omp(n, alpha, x, y)) return;
#endif
        yaxpy_unit(n, alpha, x, y);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { y[iy] += alpha * x[ix]; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_AXPY(yaxpy, T, T)
