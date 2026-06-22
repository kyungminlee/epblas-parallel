#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
/* yswap — kind10 complex: swap X ↔ Y. */
typedef _Complex long double T;

/* Unit-stride kernel, shared by the serial entry and the per-thread OMP slices.
 * 3-way unrolled to amortize loop overhead over the complex load/store pairs. */
static void yswap_unit(ptrdiff_t n, T *x, T *y)
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
/* Threaded unit-stride complex SWAP — same cache-bandwidth rationale as
 * eaxpy_omp (see eaxpy.c). Heaviest RMW (2 reads + 2 writes/elem); a complex
 * element is 2x bytes, so the bandwidth regime — and thus the fork/join
 * break-even — is reached at ~half eswap's element count (eswap floor 3072,
 * 16B/elem; yswap 32B/elem). The threshold is set by par4<=par1 AND par4<=ob4
 * (ob keeps swap serial at small N). Re-measured under iomp5 at reps=60 (the old
 * "break-even ~150" was stale): par4/par1 is 1.077@1024 (5558 vs 5159 serial —
 * threading LOSES), 1.021@1536 (7895 vs 7730 — still loses), then 0.93@2048
 * (9675 vs 10397 — wins) and 0.59@4096; par4/ob4 tracks it (1.041@1024,
 * 0.989@1536, 0.90@2048). Break-even ~N=1800, so stay serial through 1536 and
 * thread from 2048 — par's serial is already fastest below break-even (par/ref
 * ~0.94), so threading there would only regress it. */
#define YSWAP_OMP_MIN 1536
static int yswap_omp(ptrdiff_t n, T *x, T *y)
{
    if (n <= YSWAP_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) yswap_unit(hi - lo, x + lo, y + lo);
    }
    return 1;
}
#endif

static void yswap_core(ptrdiff_t n, T *x, ptrdiff_t incx, T *y, ptrdiff_t incy)
{
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (yswap_omp(n, x, y)) return;
#endif
        yswap_unit(n, x, y);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { T t = x[ix]; x[ix] = y[iy]; y[iy] = t; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_SWAP(yswap, T)
