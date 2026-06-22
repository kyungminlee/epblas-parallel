/* edot — kind10 real: returns Σ X·Y. */
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include <stddef.h>
#endif
#include "../common/epblas_facade.h"
typedef long double T;

/* Σ x·y over a contiguous logical range. FOUR independent accumulators hide
 * the ~3-cycle x87 fadd latency; four is the most chains that fit the 8-deep
 * x87 stack alongside the two load temps without spilling (a 5th chain pushed
 * the stack to 7/8 and cost ~10% vs the openblas clone at L1-resident N=1024).
 * The 8-way unroll (two elements per chain) amortizes the per-iteration
 * increment/compare/branch over more work, recovering the small-N overhead and
 * keeping the large-N win. Unit-stride fast path; strided walk handles
 * arbitrary (possibly negative) increments for the serial case. */
static T edot_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx, const T *y, ptrdiff_t incy)
{
    T s = 0.0L;
    if (incx == 1 && incy == 1) {
        T s0 = 0.0L, s1 = 0.0L, s2 = 0.0L, s3 = 0.0L;
        ptrdiff_t i = 0;
        for (; i + 7 < n; i += 8) {
            s0 += x[i    ] * y[i    ];
            s1 += x[i + 1] * y[i + 1];
            s2 += x[i + 2] * y[i + 2];
            s3 += x[i + 3] * y[i + 3];
            s0 += x[i + 4] * y[i + 4];
            s1 += x[i + 5] * y[i + 5];
            s2 += x[i + 6] * y[i + 6];
            s3 += x[i + 7] * y[i + 7];
        }
        s = (s0 + s1) + (s2 + s3);
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

static T edot_core(ptrdiff_t n, const T *x, ptrdiff_t incx,
                   const T *y, ptrdiff_t incy)
{
    if (n <= 0) return 0.0L;
#ifdef _OPENMP
    if (incx == 1 && incy == 1) {
        T s;
        if (edot_omp(n, x, y, &s)) return s;
    }
#endif
    return edot_kernel(n, x, incx, y, incy);
}

EPBLAS_FACADE_DOT(edot, T, T)
