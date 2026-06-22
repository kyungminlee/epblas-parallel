/* ydotu — kind10 complex: returns Σ X·Y (no conjugate).
 *
 * Single accumulator: see ydotc for rationale (x87 stack pressure
 * from 2-acc unroll, Addendum 1 §kind10 complex). */
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include <stddef.h>
#endif
#include "../common/epblas_facade.h"
typedef _Complex long double T;

static T ydotu_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx, const T *y, ptrdiff_t incy)
{
    T s = 0.0L;
    if (incx == 1 && incy == 1) {
        const T *xe = x + n;
        for (const T *xp = x, *yp = y; xp < xe; ++xp, ++yp) s += *xp * *yp;
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { s += x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s;
}

#ifdef _OPENMP
/* Threaded partial-reduction for unit-stride X·Y. Mirrors ydotc; manual
 * complex partial[] (no OMP reduction for `_Complex long double`).
 * RECALIBRATED 2026-06-07 (was 10000): same iomp5 stale-gate fix as ydotc — the
 * complex fp80 MAC threads from far lower n. Measured par4/par1 (taskset 0-3,
 * min-of-12): n=400 0.83, n=600 0.64, n=1000 0.48, n=4000 0.30. 1024 adds wins
 * in [1024,10000) where ob is serial; par still threads >=10000 to match ob. */
#define YDOTU_OMP_MIN 1024
#define YDOTU_MAX_CPUS 64
__attribute__((noinline)) static ptrdiff_t ydotu_omp(ptrdiff_t n, const T *x, const T *y, T *out)
{
    if (n <= YDOTU_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > YDOTU_MAX_CPUS) nthreads = YDOTU_MAX_CPUS;
    T partial[YDOTU_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) partial[tid] = ydotu_kernel(hi - lo, x + lo, 1, y + lo, 1);
    }
    T s = 0.0L;
    for (ptrdiff_t i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

static T ydotu_core(ptrdiff_t n, const T *x, ptrdiff_t incx,
                    const T *y, ptrdiff_t incy)
{
    if (n <= 0) return (T)0.0L;
#ifdef _OPENMP
    if (incx == 1 && incy == 1) {
        T s;
        if (ydotu_omp(n, x, y, &s)) return s;
    }
#endif
    return ydotu_kernel(n, x, incx, y, incy);
}

EPBLAS_FACADE_DOT(ydotu, T, T)
