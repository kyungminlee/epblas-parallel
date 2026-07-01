/* iyamax — kind10 complex: 1-based argmax(|re|+|im|). */
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include <stddef.h>
#endif
#include "../common/epblas_facade.h"
typedef _Complex long double TC;
typedef long double R;

/* Scan a contiguous unit-stride range [0,n); return the 0-based index of the
 * first element with maximal |re|+|im| and store that magnitude in *bv_out.
 * Strictly-greater update keeps the lowest index on ties. */
static ptrdiff_t iyamax_kernel(ptrdiff_t n, const TC *x, R *bv_out)
{
    ptrdiff_t best = 0;
    R bv = fabsl(__real__ x[0]) + fabsl(__imag__ x[0]);
    /* Separate load pointer so GCC strength-reduces the load to a pointer
     * increment instead of recomputing i*sizeof per element (i is otherwise
     * tied to `best = i`). Bit-identical. See ieamax for the disasm rationale. */
    const TC *p = x;
    for (ptrdiff_t i = 1; i < n; ++i) {
        ++p;
        R v = fabsl(__real__ *p) + fabsl(__imag__ *p);
        if (v > bv) { bv = v; best = i; }
    }
    *bv_out = bv;
    return best;
}

#ifdef _OPENMP
/* Threaded argmax for large unit-stride X. ob threads at n>10000; par mirrors
 * so it doesn't trail ob ~3x at n≥16384. Each thread scans a contiguous chunk;
 * the serial reduce walks threads in ascending tid (= ascending element order)
 * with strictly-greater so the lowest global index wins ties — matching the
 * reference IxAMAX "first occurrence" semantics. */
#define IYAMAX_OMP_MIN 10000
#define IYAMAX_MAX_CPUS 64
__attribute__((noinline)) static ptrdiff_t iyamax_omp(ptrdiff_t n, const TC *x, ptrdiff_t *out)
{
    if (n <= IYAMAX_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > IYAMAX_MAX_CPUS) nthreads = IYAMAX_MAX_CPUS;
    R pval[IYAMAX_MAX_CPUS];
    ptrdiff_t pidx[IYAMAX_MAX_CPUS];   /* global 0-based index of each thread's local best */
    for (ptrdiff_t i = 0; i < nthreads; ++i) { pval[i] = -1.0L; pidx[i] = -1; }
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) {
            R bv;
            ptrdiff_t li = iyamax_kernel(hi - lo, x + lo, &bv);
            pval[tid] = bv;
            pidx[tid] = lo + li;
        }
    }
    R bv = -1.0L;
    ptrdiff_t best = 0;
    for (ptrdiff_t i = 0; i < nthreads; ++i) {
        if (pidx[i] >= 0 && pval[i] > bv) { bv = pval[i]; best = pidx[i]; }
    }
    *out = best + 1;   /* 1-based */
    return 1;
}
#endif

static ptrdiff_t iyamax_core(ptrdiff_t n, const TC *x, ptrdiff_t incx)
{
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        ptrdiff_t idx;
        if (iyamax_omp(n, x, &idx)) return idx;
    }
#endif
    /* Serial strided/unit scan (1-based). */
    ptrdiff_t best = 1;
    R bv = fabsl(__real__ x[0]) + fabsl(__imag__ x[0]);
    ptrdiff_t ix = incx;
    for (ptrdiff_t i = 2; i <= n; ++i) {
        R v = fabsl(__real__ x[ix]) + fabsl(__imag__ x[ix]);
        if (v > bv) { bv = v; best = i; }
        ix += incx;
    }
    return best;
}

EPBLAS_FACADE_IAMAX(iyamax, TC)
