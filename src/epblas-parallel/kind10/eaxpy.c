#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
/* eaxpy — kind10 real: Y := α·X + Y. */
typedef long double T;

/* Unit-stride kernel, shared by the serial entry and the per-thread slices of
 * the OMP path. 8-way unroll: per-element x87 op counts are identical at any
 * unroll factor (2 fldt, 1 fmul, 1 faddp, 1 fstpt — no SIMD for fp80), so the
 * only lever is loop-overhead amortization; unrolling by 8 (the epblas-openblas
 * daxpy shape) halves the per-element increment/compare/branch cost vs a 4-way
 * body and recovers ~3% over the L1-resident range.
 *
 * Remainder LAST (n1 = n & -8 main loop from 0, scalar tail), matching ob's
 * daxpy. The earlier remainder-FIRST form (n % 8 head, main loop from m) made
 * GCC compile a branchy Duff's-device ladder at the function entry — paid on
 * every call — and started the aligned main loop at the unaligned offset m,
 * leaving ~2% on the table at the L2-band size (N~64k) vs ob. Each += is
 * independent, so the head→tail remainder move is bit-identical. */
static void eaxpy_unit(ptrdiff_t n, T alpha, const T *x, T *y)
{
    ptrdiff_t i, n1 = n & -8;
    for (i = 0; i < n1; i += 8) {
        y[i    ] += alpha * x[i    ];
        y[i + 1] += alpha * x[i + 1];
        y[i + 2] += alpha * x[i + 2];
        y[i + 3] += alpha * x[i + 3];
        y[i + 4] += alpha * x[i + 4];
        y[i + 5] += alpha * x[i + 5];
        y[i + 6] += alpha * x[i + 6];
        y[i + 7] += alpha * x[i + 7];
    }
    for (; i < n; ++i) y[i] += alpha * x[i];
}

#ifdef _OPENMP
/* Threaded unit-stride AXPY. fp80 is cheap per element, so this op is cache/
 * memory-bandwidth bound, not compute bound like quad. It was long left serial
 * on the premise "write-bound RMW doesn't thread" — but that only holds in the
 * >=1M main-memory regime. In the cache-resident regime (~N=768..512K) per-core
 * L1/L2 bandwidth scales with threads: OpenBLAS threads daxpy ~3.8x there, so an
 * unthreaded par lost up to ~3.9x at OMP=4. Threading par's own tuned kernel on
 * index slices recovers it — net win out to 4M, so no upper bound is needed.
 * Threshold is set by par4<=ob4 (the real bar), NOT par4<par1: ob keeps this op
 * SERIAL at small N, so par's 4-thread time must beat ob's *serial* time, which
 * needs the work to dwarf the fork/join cost. Measured under iomp5: par4/ob4 is
 * 1.57@1024, 1.19@2048, 1.05@3072, then 0.99@4096 and 0.85@6144 — break-even
 * ~4096, so stay serial through 3072. Strided cases stay serial (rare). */
#define EAXPY_OMP_MIN 3072
static int eaxpy_omp(ptrdiff_t n, T alpha, const T *x, T *y)
{
    if (n <= EAXPY_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) eaxpy_unit(hi - lo, alpha, x + lo, y + lo);
    }
    return 1;
}
#endif

static void eaxpy_core(ptrdiff_t n, const T *alpha_,
                       const T *x, ptrdiff_t incx,
                       T *y, ptrdiff_t incy)
{
    const T alpha = *alpha_;
    if (n <= 0 || alpha == 0.0L) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (eaxpy_omp(n, alpha, x, y)) return;
#endif
        eaxpy_unit(n, alpha, x, y);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { y[iy] += alpha * x[ix]; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_AXPY(eaxpy, T, T)
