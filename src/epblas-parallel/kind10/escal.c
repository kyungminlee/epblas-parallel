#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
/* escal — kind10 real: X := α · X. x87 fp80, no SIMD path. */
typedef long double T;

/* Unit-stride kernel, shared by the serial entry and the per-thread OMP slices.
 * 8-way unrolled: each x87 `fmulp` has ~3-cycle latency; unrolling lets the OOO
 * core dispatch independent fldt/fmul/fstpt sequences in parallel since all
 * mults share alpha but touch distinct x[i]. Plain `for(i) x[i] *= alpha` lost
 * ~25% vs the references. This is OpenBLAS dscal's shape (8-way, remainder
 * last); the narrower 5-way netlib form left ~3% on the table vs ob at the
 * L2-band size (N~64k), where the per-element loop overhead is still exposed —
 * matching the 8-way head closes it. Each *= is independent, so the wider
 * unroll and the head→tail remainder move are bit-identical. */
static void escal_unit(ptrdiff_t n, T alpha, T *x)
{
    ptrdiff_t i, n1 = n & -8;
    for (i = 0; i < n1; i += 8) {
        x[i    ] *= alpha; x[i + 1] *= alpha;
        x[i + 2] *= alpha; x[i + 3] *= alpha;
        x[i + 4] *= alpha; x[i + 5] *= alpha;
        x[i + 6] *= alpha; x[i + 7] *= alpha;
    }
    for (; i < n; ++i) x[i] *= alpha;
}

#ifdef _OPENMP
/* Threaded unit-stride SCAL — same cache-bandwidth rationale as eaxpy_omp
 * (see eaxpy.c): cache-resident regime threads where OpenBLAS does too; net win
 * out to 4M, so no upper bound. SCAL is the lightest RMW (1 read+1 write, 1 mul/
 * elem) so it has the highest break-even of the family: threshold set by
 * par4<=ob4 (ob keeps scal serial at small N). Measured under iomp5: par4/ob4
 * 1.68@1536, 1.23@3072, 1.08@6144, then 1.006@8192 — break-even ~8192, stay
 * serial through 6144. */
#define ESCAL_OMP_MIN 6144
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
