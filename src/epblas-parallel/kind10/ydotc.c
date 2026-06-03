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
#endif
typedef _Complex long double T;

static T ydotc_kernel(int n, const T *x, int incx, const T *y, int incy)
{
    T s = 0.0L;
    if (incx == 1 && incy == 1) {
        const T *xe = x + n;
        for (const T *xp = x, *yp = y; xp < xe; ++xp, ++yp) s += ~*xp * *yp;
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { s += ~x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s;
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride conj(X)·Y. ob threads at
 * n>10000; par mirrors so it doesn't trail ob ~3.5x at n≥16384. A manual
 * complex partial[] array is required — `#pragma omp reduction(+:)` does not
 * accept `_Complex long double`. noinline isolates the region bookkeeping
 * from the serial kernel's already-saturated x87 stack. */
#define YDOTC_OMP_MIN 10000
#define YDOTC_MAX_CPUS 64
__attribute__((noinline)) static int ydotc_omp(int n, const T *x, const T *y, T *out)
{
    if (n <= YDOTC_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > YDOTC_MAX_CPUS) nthreads = YDOTC_MAX_CPUS;
    T partial[YDOTC_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) partial[tid] = ydotc_kernel(hi - lo, x + lo, 1, y + lo, 1);
    }
    T s = 0.0L;
    for (int i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

T ydotc_(const int *n_, const T *x, const int *incx_,
         const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return (T)0.0L;
#ifdef _OPENMP
    if (incx == 1 && incy == 1) {
        T s;
        if (ydotc_omp(n, x, y, &s)) return s;
    }
#endif
    return ydotc_kernel(n, x, incx, y, incy);
}
