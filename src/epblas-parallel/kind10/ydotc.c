/* ydotc — kind10 complex: returns Σ conj(X)·Y.
 *
 * Single accumulator: 2-acc unroll was tried (commit add00f58) to expose
 * ILP and mask x87 fmul latency, but each `_Complex long double` multiply
 * needs ~6 fp80 slots — 2 accs + temp slots overflow the 8-deep x87
 * stack and force fxch/spill (Addendum 1 §kind10 complex). With single
 * acc and pointer-walk, gcc's scheduler produces an inner loop close to
 * gfortran's reference codegen for ZDOTC.
 */
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include <stddef.h>
#endif
#include "../common/epblas_facade.h"
typedef _Complex long double T;

static T ydotc_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx, const T *y, ptrdiff_t incy)
{
    T s = 0.0L;
    if (incx == 1 && incy == 1) {
        const T *xe = x + n;
        for (const T *xp = x, *yp = y; xp < xe; ++xp, ++yp) s += ~*xp * *yp;
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { s += ~x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s;
}

#ifdef _OPENMP
/* Threaded partial-reduction for unit-stride conj(X)·Y. A manual complex
 * partial[] array is required — `#pragma omp reduction(+:)` does not accept
 * `_Complex long double`. noinline isolates the region bookkeeping from the
 * serial kernel's already-saturated x87 stack.
 * RECALIBRATED 2026-06-07 (was 10000): 10000 mirrored OpenBLAS zdotc's gate so
 * par wouldn't trail ob ~3.5x at n>=16384. But the complex fp80 MAC is ~4x the
 * work of a real dot, so under iomp5 (hot-team reuse) this threads from far
 * lower n — measured par4/par1 (taskset 0-3, min-of-12): n=400 0.81, n=600
 * 0.61, n=1000 0.50, n=4000 0.31. Lowering to 1024 only ADDS wins in
 * [1024,10000) where ob stays serial (par4 now beats ob4 there); par still
 * threads >=10000 to match ob. n=300 is the break-even (0.97), so 1024 keeps
 * a safe margin past it — reduction timings are contention-sensitive. */
#define YDOTC_OMP_MIN 1024
#define YDOTC_MAX_CPUS 64
__attribute__((noinline)) static ptrdiff_t ydotc_omp(ptrdiff_t n, const T *x, const T *y, T *out)
{
    if (n <= YDOTC_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > YDOTC_MAX_CPUS) nthreads = YDOTC_MAX_CPUS;
    T partial[YDOTC_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) partial[tid] = ydotc_kernel(hi - lo, x + lo, 1, y + lo, 1);
    }
    T s = 0.0L;
    for (ptrdiff_t i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

static T ydotc_core(ptrdiff_t n, const T *x, ptrdiff_t incx,
                    const T *y, ptrdiff_t incy)
{
    if (n <= 0) return (T)0.0L;
#ifdef _OPENMP
    if (incx == 1 && incy == 1) {
        T s;
        if (ydotc_omp(n, x, y, &s)) return s;
    }
#endif
    return ydotc_kernel(n, x, incx, y, incy);
}

EPBLAS_FACADE_DOT(ydotc, T, T)
