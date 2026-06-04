/* ydotu — kind10 complex: returns Σ X·Y (no conjugate).
 *
 * Single accumulator: see ydotc for rationale (x87 stack pressure
 * from 2-acc unroll, Addendum 1 §kind10 complex). */
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include <stddef.h>
#endif
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
/* Threaded partial-reduction for large unit-stride X·Y. Mirrors ydotc; manual
 * complex partial[] (no OMP reduction for `_Complex long double`). */
#define YDOTU_OMP_MIN 10000
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
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) partial[tid] = ydotu_kernel(hi - lo, x + lo, 1, y + lo, 1);
    }
    T s = 0.0L;
    for (ptrdiff_t i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

T ydotu_(const int *n_, const T *x, const int *incx_,
         const T *y, const int *incy_)
{
    const ptrdiff_t n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return (T)0.0L;
#ifdef _OPENMP
    if (incx == 1 && incy == 1) {
        T s;
        if (ydotu_omp(n, x, y, &s)) return s;
    }
#endif
    return ydotu_kernel(n, x, incx, y, incy);
}
