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
#include "mf_omp.h"
#endif
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::eq1;
namespace {

#ifdef MBLAS_SIMD_DD
using simd_exact::load_dd4;
using simd_exact::store_dd4;
#endif
}  // namespace

#ifdef MBLAS_SIMD_DD
/* AVX2+FMA kernel body — compiled under target("avx2,fma") so it builds even
 * when the library's baseline -march is pre-Haswell; reached only behind the
 * mf_have_avx2_fma() runtime probe below. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static void mscal_unit_simd(std::ptrdiff_t n, TR alpha, TR *x)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xh, xl;
        load_dd4(&x[i], xh, xl);
        __m256d nh, nl;
        simd_fast::mul(xh, xl, ah, al, nh, nl);
        store_dd4(&x[i], nh, nl);
    }
    for (std::ptrdiff_t i = n4; i < n; ++i) x[i] = x[i] * alpha;
}
#pragma GCC pop_options
#endif

/* X := α·X over a contiguous unit-stride range — serial kernel. Runtime
 * dispatch: SIMD on Haswell+, scalar (always compiled) on Sandybridge/Ivy. */
static void mscal_unit(std::ptrdiff_t n, TR alpha, TR *x)
{
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) { mscal_unit_simd(n, alpha, x); return; }
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = x[i] * alpha;
}

#ifdef _OPENMP
/* Threaded scale: disjoint output slices, each running the SIMD kernel. */
#define MSCAL_OMP_MIN 2048
__attribute__((noinline)) static std::ptrdiff_t mscal_omp(std::ptrdiff_t n, TR alpha, TR *x)
{
    if (n <= MSCAL_OMP_MIN || !blas_omp_should_thread())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) mscal_unit(hi - lo, alpha, x + lo);
    }
    return 1;
}
#endif

static void mscal_core(std::ptrdiff_t n, const TR *alpha_, TR *x, std::ptrdiff_t incx)
{
    const TR alpha = *alpha_;
    if (n <= 0 || eq1(alpha)) return;

    if (incx == 1) {
#ifdef _OPENMP
        if (mscal_omp(n, alpha, x)) return;
#endif
        mscal_unit(n, alpha, x);
    } else {
        std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) { x[ix] = x[ix] * alpha; ix += incx; }
    }
}

extern "C" { EPBLAS_FACADE_SCAL(mscal, TR, TR) }
