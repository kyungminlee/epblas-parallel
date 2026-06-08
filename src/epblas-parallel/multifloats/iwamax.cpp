/* iwamax — multifloats complex DD: 1-based argmax(|re|+|im|). */
#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
/* Inline magnitude — see imamax.cpp: the public fabsdd() is out-of-line, so
 * cabs1 emitted two PLT calls per element. a < 0 ? -a : a is header-inline and
 * yields the identical canonical magnitude. */
inline R t_abs(R a) { return a < 0 ? -a : a; }
inline bool dd_gt(R a, R b) {
    return a.limbs[0] > b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] > b.limbs[1]);
}
inline R cabs1(T const &z) { return t_abs(z.re) + t_abs(z.im); }

/* Contiguous unit-stride scan: 0-based index of first max-|re|+|im| element. */
inline ptrdiff_t iwamax_scan(ptrdiff_t n, const T *x, R *bv_out)
{
    ptrdiff_t best = 0;
    R bv = cabs1(x[0]);
    for (ptrdiff_t i = 1; i < n; ++i) {
        R v = cabs1(x[i]);
        if (v > bv) { bv = v; best = i; }   /* lexicographic >, matches ob clone */
    }
    *bv_out = bv;
    return best;
}
}

#ifdef _OPENMP
/* Threaded argmax for large unit-stride X. Each thread scans its contiguous
 * slice keeping the first (lowest-index) maximum, then partials merge in
 * ascending-tid order with a strict-greater test — so the lowest global index
 * wins any tie, bit-identical to the serial left-to-right scan. */
#define IWAMAX_OMP_MIN 8192
#define IWAMAX_MAX_CPUS 64
__attribute__((noinline)) static int iwamax_omp(int n, const T *x, int *out)
{
    if (n <= IWAMAX_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > IWAMAX_MAX_CPUS) nthreads = IWAMAX_MAX_CPUS;
    int   idx[IWAMAX_MAX_CPUS];
    R     val[IWAMAX_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        int b = 0;
        R bv{0.0, 0.0};
        if (lo < hi) {
            R sv;
            ptrdiff_t li = iwamax_scan(hi - lo, x + lo, &sv);
            b = lo + (int)li + 1;   /* 1-based global index */
            bv = sv;
        }
        idx[tid] = b; val[tid] = bv;
    }
    int best = 0;
    R bestv{0.0, 0.0};
    for (int t = 0; t < nthreads; ++t) {
        if (idx[t] == 0) continue;
        if (best == 0 || dd_gt(val[t], bestv)) { best = idx[t]; bestv = val[t]; }
    }
    *out = best;
    return 1;
}
#endif

extern "C" int iwamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        int r;
        if (iwamax_omp(n, x, &r)) return r;
    }
#endif
    if (incx == 1) {
        R bv;
        return (int)iwamax_scan(n, x, &bv) + 1;
    }
    int best = 1;
    R bestv = cabs1(x[0]);
    int ix = incx;
    for (int i = 2; i <= n; ++i) {
        R av = cabs1(x[ix]);
        if (dd_gt(av, bestv)) { bestv = av; best = i; }
        ix += incx;
    }
    return best;
}
