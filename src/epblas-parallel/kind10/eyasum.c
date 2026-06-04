/* eyasum — kind10: Σ (|re(X)| + |im(X)|) for complex X. */
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include <stddef.h>
#endif
typedef _Complex long double T;
typedef long double R;

/* 3-accumulator unroll of |re|+|im| over a contiguous logical range. */
static R eyasum_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx)
{
    R s = 0.0L, s1 = 0.0L, s2 = 0.0L;
    if (incx == 1) {
        ptrdiff_t i = 0;
        for (; i + 2 < n; i += 3) {
            s  += fabsl(__real__ x[i    ]) + fabsl(__imag__ x[i    ]);
            s1 += fabsl(__real__ x[i + 1]) + fabsl(__imag__ x[i + 1]);
            s2 += fabsl(__real__ x[i + 2]) + fabsl(__imag__ x[i + 2]);
        }
        s += s1 + s2;
        for (; i < n; ++i) s += fabsl(__real__ x[i]) + fabsl(__imag__ x[i]);
    } else {
        for (ptrdiff_t i = 0, ix = 0; i < n; ++i, ix += incx)
            s += fabsl(__real__ x[ix]) + fabsl(__imag__ x[ix]);
    }
    return s;
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride X. ob threads at n>10000;
 * par mirrors so it doesn't trail ob ~2.2x at n≥16384. */
#define EYASUM_OMP_MIN 10000
#define EYASUM_MAX_CPUS 64
__attribute__((noinline)) static ptrdiff_t eyasum_omp(ptrdiff_t n, const T *x, R *out)
{
    if (n <= EYASUM_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > EYASUM_MAX_CPUS) nthreads = EYASUM_MAX_CPUS;
    R partial[EYASUM_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) partial[tid] = eyasum_kernel(hi - lo, x + lo, 1);
    }
    R s = 0.0L;
    for (ptrdiff_t i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

R eyasum_(const int *n_, const T *x, const int *incx_)
{
    const ptrdiff_t n = *n_, incx = *incx_;
    if (n < 1 || incx < 1) return 0.0L;
#ifdef _OPENMP
    if (incx == 1) {
        R s;
        if (eyasum_omp(n, x, &s)) return s;
    }
#endif
    return eyasum_kernel(n, x, incx);
}
