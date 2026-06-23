/* iqamax — kind16 real: 1-based argmax(|X|). */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
/* fabsq via __builtin_fabsf128 — single `pand` instead of a libquadmath function call. */
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __float128 TR;

/* Scan a contiguous unit-stride range [0,n); return the 0-based index of the
 * first element with maximal |x| and store that magnitude in *bv_out.
 * Strictly-greater update keeps the lowest index on ties. */
static ptrdiff_t iqamax_kernel(ptrdiff_t n, const TR *x, TR *bv_out)
{
    ptrdiff_t best = 0;
    TR bv = fabsq(x[0]);
    for (ptrdiff_t i = 1; i < n; ++i) {
        TR v = fabsq(x[i]);
        if (v > bv) { bv = v; best = i; }
    }
    *bv_out = bv;
    return best;
}

#ifdef _OPENMP
/* Threaded argmax for large unit-stride X. Ascending-tid serial reduce with
 * strictly-greater keeps the lowest global index on ties (first-occurrence),
 * matching the serial scan. See qasum.c for threshold/noinline rationale. */
#define IQAMAX_OMP_MIN 128
#define IQAMAX_MAX_CPUS 64
__attribute__((noinline)) static ptrdiff_t iqamax_omp(ptrdiff_t n, const TR *x, ptrdiff_t *out)
{
    if (n <= IQAMAX_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > IQAMAX_MAX_CPUS) nthreads = IQAMAX_MAX_CPUS;
    TR pval[IQAMAX_MAX_CPUS];
    ptrdiff_t pidx[IQAMAX_MAX_CPUS];   /* global 0-based index of each thread's local best */
    for (ptrdiff_t i = 0; i < nthreads; ++i) { pval[i] = -1.0Q; pidx[i] = -1; }
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) {
            TR bv;
            ptrdiff_t li = iqamax_kernel(hi - lo, x + lo, &bv);
            pval[tid] = bv;
            pidx[tid] = lo + li;
        }
    }
    TR bv = -1.0Q;
    ptrdiff_t best = 0;
    for (ptrdiff_t i = 0; i < nthreads; ++i) {
        if (pidx[i] >= 0 && pval[i] > bv) { bv = pval[i]; best = pidx[i]; }
    }
    *out = best + 1;   /* 1-based */
    return 1;
}
#endif

static ptrdiff_t iqamax_core(ptrdiff_t n, const TR *x, ptrdiff_t incx)
{
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        ptrdiff_t idx;
        if (iqamax_omp(n, x, &idx)) return idx;
    }
#endif
    ptrdiff_t best = 1; TR bv = fabsq(x[0]); ptrdiff_t ix = incx;
    for (ptrdiff_t i = 2; i <= n; ++i) { TR v = fabsq(x[ix]); if (v > bv) { bv = v; best = i; } ix += incx; }
    return best;
}

EPBLAS_FACADE_IAMAX(iqamax, TR)
