/* ieamax — kind10 real: 1-based argmax(|X|). */
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include <stddef.h>
#endif
#include "../common/epblas_facade.h"
typedef long double TR;

/* Scan a contiguous unit-stride range [0,n); return the 0-based index of the
 * first element with maximal |x| and store that magnitude in *bv_out.
 * Strictly-greater update keeps the lowest index on ties. */
static ptrdiff_t ieamax_kernel(ptrdiff_t n, const TR *x, TR *bv_out)
{
    ptrdiff_t best = 0;
    TR bv = fabsl(x[0]);
    /* Walk a separate load pointer instead of x[i]: if the load address shares
     * the index i with `best = i`, GCC ties them to one IV and recomputes i*16
     * (mov+shl) per element (1.76x slower). A distinct pointer lets GCC
     * strength-reduce the load to a pure pointer increment — matching the
     * serial core loop's codegen. Bit-identical (same scan, same tie-break). */
    const TR *p = x;
    for (ptrdiff_t i = 1; i < n; ++i) {
        ++p;
        TR v = fabsl(*p);
        if (v > bv) { bv = v; best = i; }
    }
    *bv_out = bv;
    return best;
}

#ifdef _OPENMP
/* Threaded argmax for large unit-stride X. ob threads at n>10000; par mirrors
 * so it doesn't trail ob ~3x at n≥16384. Ascending-tid serial reduce with
 * strictly-greater keeps the lowest global index on ties (IxAMAX first-occurrence). */
#define IEAMAX_OMP_MIN 10000
#define IEAMAX_MAX_CPUS 64
__attribute__((noinline)) static ptrdiff_t ieamax_omp(ptrdiff_t n, const TR *x, ptrdiff_t *out)
{
    if (n <= IEAMAX_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > IEAMAX_MAX_CPUS) nthreads = IEAMAX_MAX_CPUS;
    TR pval[IEAMAX_MAX_CPUS];
    ptrdiff_t pidx[IEAMAX_MAX_CPUS];   /* global 0-based index of each thread's local best */
    for (ptrdiff_t i = 0; i < nthreads; ++i) { pval[i] = -1.0L; pidx[i] = -1; }
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) {
            TR bv;
            ptrdiff_t li = ieamax_kernel(hi - lo, x + lo, &bv);
            pval[tid] = bv;
            pidx[tid] = lo + li;
        }
    }
    TR bv = -1.0L;
    ptrdiff_t best = 0;
    for (ptrdiff_t i = 0; i < nthreads; ++i) {
        if (pidx[i] >= 0 && pval[i] > bv) { bv = pval[i]; best = pidx[i]; }
    }
    *out = best + 1;   /* 1-based */
    return 1;
}
#endif

static ptrdiff_t ieamax_core(ptrdiff_t n, const TR *x, ptrdiff_t incx)
{
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        ptrdiff_t idx;
        if (ieamax_omp(n, x, &idx)) return idx;
    }
#endif
    ptrdiff_t best = 1; TR bv = fabsl(x[0]); ptrdiff_t ix = incx;
    for (ptrdiff_t i = 2; i <= n; ++i) { TR v = fabsl(x[ix]); if (v > bv) { bv = v; best = i; } ix += incx; }
    return best;
}

EPBLAS_FACADE_IAMAX(ieamax, TR)
