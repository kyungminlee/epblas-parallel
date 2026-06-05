/* iqamax — kind16 real: 1-based argmax(|X|). */
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
/* fabsq via __builtin_fabsf128 — single `pand` instead of a libquadmath function call. */
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __float128 T;

/* Scan a contiguous unit-stride range [0,n); return the 0-based index of the
 * first element with maximal |x| and store that magnitude in *bv_out.
 * Strictly-greater update keeps the lowest index on ties. */
static int iqamax_kernel(int n, const T *x, T *bv_out)
{
    int best = 0;
    T bv = fabsq(x[0]);
    for (int i = 1; i < n; ++i) {
        T v = fabsq(x[i]);
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
__attribute__((noinline)) static int iqamax_omp(int n, const T *x, int *out)
{
    if (n <= IQAMAX_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > IQAMAX_MAX_CPUS) nthreads = IQAMAX_MAX_CPUS;
    T pval[IQAMAX_MAX_CPUS];
    int pidx[IQAMAX_MAX_CPUS];   /* global 0-based index of each thread's local best */
    for (int i = 0; i < nthreads; ++i) { pval[i] = -1.0Q; pidx[i] = -1; }
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) {
            T bv;
            int li = iqamax_kernel(hi - lo, x + lo, &bv);
            pval[tid] = bv;
            pidx[tid] = lo + li;
        }
    }
    T bv = -1.0Q;
    int best = 0;
    for (int i = 0; i < nthreads; ++i) {
        if (pidx[i] >= 0 && pval[i] > bv) { bv = pval[i]; best = pidx[i]; }
    }
    *out = best + 1;   /* 1-based */
    return 1;
}
#endif

int iqamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        int idx;
        if (iqamax_omp(n, x, &idx)) return idx;
    }
#endif
    int best = 1; T bv = fabsq(x[0]); int ix = incx;
    for (int i = 2; i <= n; ++i) { T v = fabsq(x[ix]); if (v > bv) { bv = v; best = i; } ix += incx; }
    return best;
}
