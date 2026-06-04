/* edot — kind10 real: returns Σ X·Y. */
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include <stddef.h>
#endif
typedef long double T;

/* Σ x·y over a contiguous logical range. 5-accumulator unroll matches NETLIB
 * DDOT, masks the ~3-cycle x87 fadd latency. Unit-stride fast path; strided
 * walk handles arbitrary (possibly negative) increments for the serial case. */
static T edot_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx, const T *y, ptrdiff_t incy)
{
    T s = 0.0L;
    if (incx == 1 && incy == 1) {
        T s0 = 0.0L, s1 = 0.0L, s2 = 0.0L, s3 = 0.0L, s4 = 0.0L;
        ptrdiff_t i = 0;
        for (; i + 4 < n; i += 5) {
            s0 += x[i    ] * y[i    ];
            s1 += x[i + 1] * y[i + 1];
            s2 += x[i + 2] * y[i + 2];
            s3 += x[i + 3] * y[i + 3];
            s4 += x[i + 4] * y[i + 4];
        }
        s = s0 + s1 + s2 + s3 + s4;
        for (; i < n; ++i) s += x[i] * y[i];
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { s += x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s;
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride X·Y. ob threads at n>10000;
 * par mirrors so it doesn't trail ob ~2.7x at n≥16384. noinline keeps the
 * region bookkeeping off the serial kernel's x87 allocation. */
#define EDOT_OMP_MIN 10000
#define EDOT_MAX_CPUS 64
__attribute__((noinline)) static ptrdiff_t edot_omp(ptrdiff_t n, const T *x, const T *y, T *out)
{
    if (n <= EDOT_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > EDOT_MAX_CPUS) nthreads = EDOT_MAX_CPUS;
    T partial[EDOT_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) partial[tid] = edot_kernel(hi - lo, x + lo, 1, y + lo, 1);
    }
    T s = 0.0L;
    for (ptrdiff_t i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

T edot_(const int *n_, const T *x, const int *incx_,
        const T *y, const int *incy_)
{
    const ptrdiff_t n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return 0.0L;
#ifdef _OPENMP
    if (incx == 1 && incy == 1) {
        T s;
        if (edot_omp(n, x, y, &s)) return s;
    }
#endif
    return edot_kernel(n, x, incx, y, incy);
}
