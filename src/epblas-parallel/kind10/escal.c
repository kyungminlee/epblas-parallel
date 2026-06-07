#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
/* escal — kind10 real: X := α · X. x87 fp80, no SIMD path. */
typedef long double T;

/* Unit-stride kernel, shared by the serial entry and the per-thread OMP slices.
 * 5-way unrolled to match the NETLIB DSCAL reference: each x87 `fmulp` has
 * ~3-cycle latency; unrolling lets the OOO core dispatch independent
 * fldt/fmul/fstpt sequences in parallel since all five mults share alpha but
 * touch distinct x[i]. Plain `for(i) x[i] *= alpha` lost ~25% vs migrated at
 * every size above N=128 without it. */
static void escal_unit(ptrdiff_t n, T alpha, T *x)
{
    const ptrdiff_t m = n % 5;
    for (ptrdiff_t i = 0; i < m; ++i) x[i] *= alpha;
    for (ptrdiff_t i = m; i < n; i += 5) {
        x[i    ] *= alpha;
        x[i + 1] *= alpha;
        x[i + 2] *= alpha;
        x[i + 3] *= alpha;
        x[i + 4] *= alpha;
    }
}

#ifdef _OPENMP
/* Threaded unit-stride SCAL — same cache-bandwidth rationale as eaxpy_omp
 * (see eaxpy.c): cache-resident regime threads ~3.5x where OpenBLAS does too;
 * net win out to 4M, so no upper bound. Break-even ~N=768; 1024 keeps margin. */
#define ESCAL_OMP_MIN 1024
static int escal_omp(ptrdiff_t n, T alpha, T *x)
{
    if (n <= ESCAL_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) escal_unit(hi - lo, alpha, x + lo);
    }
    return 1;
}
#endif

void escal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const ptrdiff_t n = *n_, incx = *incx_;
    const T alpha = *alpha_;
    if (n <= 0 || alpha == 1.0L) return;
    if (incx == 1) {
#ifdef _OPENMP
        if (escal_omp(n, alpha, x)) return;
#endif
        escal_unit(n, alpha, x);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { x[ix] *= alpha; ix += incx; }
    }
}
