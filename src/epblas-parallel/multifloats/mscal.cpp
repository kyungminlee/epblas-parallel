/*
 * mscal — multifloats real DD vector scale: X := α · X.
 *
 * 4-wide AVX2 SIMD path for INCX==1, scalar fallback for strided.
 */
#include <cstddef>
#include <multifloats.h>
#include "mf_pred.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq1;
namespace {

#ifdef MBLAS_SIMD_DD
using simd_exact::load_dd4;
using simd_exact::store_dd4;
#endif
}  // namespace

/* X := α·X over a contiguous unit-stride range — serial kernel, unchanged. */
static void mscal_unit(int n, T alpha, T *x)
{
#ifdef MBLAS_SIMD_DD
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d xh, xl;
        load_dd4(&x[i], xh, xl);
        __m256d nh, nl;
        simd_fast::mul(xh, xl, ah, al, nh, nl);
        store_dd4(&x[i], nh, nl);
    }
    for (int i = n4; i < n; ++i) x[i] = x[i] * alpha;
#else
    for (int i = 0; i < n; ++i) x[i] = x[i] * alpha;
#endif
}

#ifdef _OPENMP
/* Threaded scale: disjoint output slices, each running the SIMD kernel. */
#define MSCAL_OMP_MIN 2048
__attribute__((noinline)) static int mscal_omp(int n, T alpha, T *x)
{
    if (n <= MSCAL_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) mscal_unit(hi - lo, alpha, x + lo);
    }
    return 1;
}
#endif

extern "C" void mscal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const T alpha = *alpha_;
    if (n <= 0 || eq1(alpha)) return;

    if (incx == 1) {
#ifdef _OPENMP
        if (mscal_omp(n, alpha, x)) return;
#endif
        mscal_unit(n, alpha, x);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) { x[ix] = x[ix] * alpha; ix += incx; }
    }
}
