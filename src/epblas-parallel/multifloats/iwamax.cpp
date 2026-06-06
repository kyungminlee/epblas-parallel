/* iwamax — multifloats complex DD: 1-based argmax(|re|+|im|). */
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
inline bool dd_gt(R a, R b) {
    return a.limbs[0] > b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] > b.limbs[1]);
}
inline R cabs1(T const &z) { return fabsdd(z.re) + fabsdd(z.im); }
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
            b = lo + 1;           /* 1-based global index */
            bv = cabs1(x[lo]);
            for (int i = lo + 1; i < hi; ++i) {
                R av = cabs1(x[i]);
                if (dd_gt(av, bv)) { bv = av; b = i + 1; }
            }
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
