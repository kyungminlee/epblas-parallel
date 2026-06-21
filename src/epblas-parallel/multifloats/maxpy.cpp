/*
 * maxpy — multifloats real DD: Y := α · X + Y.
 *
 * 4-wide AVX2 SIMD path for INCX=INCY=1, scalar fallback otherwise.
 */
#include <cstddef>
#include <multifloats.h>
#include "mf_pred.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#endif
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
namespace {

#ifdef MBLAS_SIMD_DD
using simd_exact::load_dd4;
using simd_exact::store_dd4;
#endif
}  // namespace

/* Y := α·X + Y over a contiguous unit-stride range — serial kernel, unchanged.
 * Carved out so the OpenMP path can run it per disjoint sub-range. */
static void maxpy_unit(int n, T alpha, const T *x, T *y)
{
#ifdef MBLAS_SIMD_DD
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d xh, xl, yh, yl;
        load_dd4(&x[i], xh, xl);
        load_dd4(&y[i], yh, yl);
        __m256d ph, pl;
        simd_fast::mul(ah, al, xh, xl, ph, pl);
        __m256d nh, nl;
        simd_fast::add(yh, yl, ph, pl, nh, nl);
        store_dd4(&y[i], nh, nl);
    }
    for (int i = n4; i < n; ++i) y[i] = y[i] + alpha * x[i];
#else
    for (int i = 0; i < n; ++i) y[i] = y[i] + alpha * x[i];
#endif
}

#ifdef _OPENMP
/* Threaded elementwise AXPY. Outputs are disjoint per slice, so each thread
 * runs the SIMD kernel over its own [lo,hi) range — no reduction. DD math is
 * compute-bound, so this threads profitably above the crossover. */
#define MAXPY_OMP_MIN 2048
__attribute__((noinline)) static int maxpy_omp(int n, T alpha, const T *x, T *y)
{
    if (n <= MAXPY_OMP_MIN || !blas_omp_available() || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) maxpy_unit(hi - lo, alpha, x + lo, y + lo);
    }
    return 1;
}
#endif

extern "C" void maxpy_(const int *n_, const T *alpha_,
                       const T *x, const int *incx_,
                       T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    if (n <= 0 || eq0(alpha)) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (maxpy_omp(n, alpha, x, y)) return;
#endif
        maxpy_unit(n, alpha, x, y);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] = y[iy] + alpha * x[ix]; ix += incx; iy += incy; }
    }
}
