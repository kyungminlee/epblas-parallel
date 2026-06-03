/* iyamax — kind10 complex: 1-based argmax(|re|+|im|). */
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef _Complex long double T;
typedef long double R;

/* Scan a contiguous unit-stride range [0,n); return the 0-based index of the
 * first element with maximal |re|+|im| and store that magnitude in *bv_out.
 * Strictly-greater update keeps the lowest index on ties. */
static int iyamax_kernel(int n, const T *x, R *bv_out)
{
    int best = 0;
    R bv = fabsl(__real__ x[0]) + fabsl(__imag__ x[0]);
    for (int i = 1; i < n; ++i) {
        R v = fabsl(__real__ x[i]) + fabsl(__imag__ x[i]);
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
__attribute__((noinline)) static int iyamax_omp(int n, const T *x, int *out)
{
    if (n <= IYAMAX_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > IYAMAX_MAX_CPUS) nthreads = IYAMAX_MAX_CPUS;
    R pval[IYAMAX_MAX_CPUS];
    int pidx[IYAMAX_MAX_CPUS];   /* global 0-based index of each thread's local best */
    for (int i = 0; i < nthreads; ++i) { pval[i] = -1.0L; pidx[i] = -1; }
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) {
            R bv;
            int li = iyamax_kernel(hi - lo, x + lo, &bv);
            pval[tid] = bv;
            pidx[tid] = lo + li;
        }
    }
    R bv = -1.0L;
    int best = 0;
    for (int i = 0; i < nthreads; ++i) {
        if (pidx[i] >= 0 && pval[i] > bv) { bv = pval[i]; best = pidx[i]; }
    }
    *out = best + 1;   /* 1-based */
    return 1;
}
#endif

int iyamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        int idx;
        if (iyamax_omp(n, x, &idx)) return idx;
    }
#endif
    /* Serial strided/unit scan (1-based). */
    int best = 1;
    R bv = fabsl(__real__ x[0]) + fabsl(__imag__ x[0]);
    int ix = incx;
    for (int i = 2; i <= n; ++i) {
        R v = fabsl(__real__ x[ix]) + fabsl(__imag__ x[ix]);
        if (v > bv) { bv = v; best = i; }
        ix += incx;
    }
    return best;
}
