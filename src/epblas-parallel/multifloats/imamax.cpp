/* imamax — multifloats real DD: 1-based argmax(|X|). */
#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
/* Inline magnitude — the public fabsdd() is an out-of-line library call, so
 * using it per element emits a PLT call in the hot loop (~2x slower than ob's
 * inlined abs). a < 0 ? -a : a uses only header-inline constexpr ops and yields
 * the identical canonical magnitude for the comparison below. */
inline T t_abs(T a) { return a < 0 ? -a : a; }

inline bool dd_gt(T a, T b) {
    return a.limbs[0] > b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] > b.limbs[1]);
}

/* Scan a contiguous unit-stride range [0,n); return the 0-based index of the
 * first element with maximal |x| and store that magnitude in *bv_out.
 * Strictly-greater update keeps the lowest index on ties. */
__attribute__((noinline)) ptrdiff_t imamax_scan(ptrdiff_t n, const T *x, T *bv_out)
{
    ptrdiff_t best = 0;
    T bv = t_abs(x[0]);
    for (ptrdiff_t i = 1; i < n; ++i) {
        T v = t_abs(x[i]);
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
#define IMAMAX_OMP_MIN 8192
#define IMAMAX_MAX_CPUS 64
__attribute__((noinline)) static int imamax_omp(int n, const T *x, int *out)
{
    if (n <= IMAMAX_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > IMAMAX_MAX_CPUS) nthreads = IMAMAX_MAX_CPUS;
    int   idx[IMAMAX_MAX_CPUS];
    T     val[IMAMAX_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        int b = 0;
        T bv{0.0, 0.0};
        if (lo < hi) {
            T sv;
            ptrdiff_t li = imamax_scan(hi - lo, x + lo, &sv);
            b = lo + (int)li + 1;   /* 1-based global index */
            bv = sv;
        }
        idx[tid] = b; val[tid] = bv;
    }
    int best = 0;
    T bestv{0.0, 0.0};
    for (int t = 0; t < nthreads; ++t) {
        if (idx[t] == 0) continue;
        if (best == 0 || dd_gt(val[t], bestv)) { best = idx[t]; bestv = val[t]; }
    }
    *out = best;
    return 1;
}
#endif

extern "C" int imamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        int r;
        if (imamax_omp(n, x, &r)) return r;
    }
#endif
    if (incx == 1) {
        T bv;
        return (int)imamax_scan(n, x, &bv) + 1;
    }
    int best = 1;
    T bestv = t_abs(x[0]);
    int ix = incx;
    for (int i = 2; i <= n; ++i) {
        T av = t_abs(x[ix]);
        if (dd_gt(av, bestv)) { bestv = av; best = i; }
        ix += incx;
    }
    return best;
}
