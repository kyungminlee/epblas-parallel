/* mnrm2 — multifloats real DD: ||X||₂ via two-pass scaled.
 *
 * Pass 1 (SIMD vmaxpd): find scale = max(|x[i].hi|).
 * Pass 2 (SIMD wide-acc): Σ (x[i]/scale)² then scale·√sum.
 */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#define MNRM2_OMP_MIN 8192
#endif
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"

namespace {
/* canonical EFTs — mf_simd_fast.h */
using simd_fast::twoprod;
using simd_fast::fast2sum;
using simd_fast::twosum;
using simd_exact::load_dd4;
using simd_fast::horizontal_dd;  /* Bailey 2-limb finalizer — mf_simd_fast.h */
}
#endif

/* Pass-1 unit kernel: max|x.hi| over a contiguous (incx==1) slice. Max is
 * exact (no rounding), so the combined global scale is order-independent and
 * BIT-EXACT regardless of how the slice is split. */
#ifdef MBLAS_SIMD_DD
/* AVX2+FMA pass-1 body — compiled under target("avx2,fma") so it builds under a
 * pre-Haswell baseline -march; reached only behind mf_have_avx2_fma() below. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static double mnrm2_maxabs_unit_simd(std::ptrdiff_t n, const TR *x)
{
    double scale_hi = 0.0;
    __m256d mx = _mm256_setzero_pd();
    const __m256d absmask = _mm256_castsi256_pd(
        _mm256_set1_epi64x(static_cast<long long>(0x7FFFFFFFFFFFFFFFULL)));
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xh, xl; load_dd4(&x[i], xh, xl); (void)xl;
        __m256d ax = _mm256_and_pd(xh, absmask);
        mx = _mm256_max_pd(mx, ax);
    }
    alignas(32) double mxa[4];
    _mm256_store_pd(mxa, mx);
    for (std::ptrdiff_t k = 0; k < 4; ++k) if (mxa[k] > scale_hi) scale_hi = mxa[k];
    for (std::ptrdiff_t i = n4; i < n; ++i) {
        TR ax = fabsdd(x[i]);
        if (ax.limbs[0] > scale_hi) scale_hi = ax.limbs[0];
    }
    return scale_hi;
}
#pragma GCC pop_options
#endif

static double mnrm2_maxabs_unit(std::ptrdiff_t n, const TR *x)
{
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) return mnrm2_maxabs_unit_simd(n, x);
#endif
    double scale_hi = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        TR ax = fabsdd(x[i]);
        if (ax.limbs[0] > scale_hi) scale_hi = ax.limbs[0];
    }
    return scale_hi;
}

/* Pass-2 unit kernel: Σ (x/scale)² over a contiguous (incx==1) slice. */
#ifdef MBLAS_SIMD_DD
/* AVX2+FMA pass-2 body — target("avx2,fma"); reached only behind the probe. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static TR mnrm2_ssq_unit_simd(std::ptrdiff_t n, const TR *x, TR scale)
{
    TR s{0.0, 0.0};
    /* Precompute inverse so the inner loop avoids dd_div. */
    TR inv = TR{1.0, 0.0} / scale;
    __m256d invh = _mm256_set1_pd(inv.limbs[0]);
    __m256d invl = _mm256_set1_pd(inv.limbs[1]);
    __m256d a0 = _mm256_setzero_pd();
    __m256d a1 = _mm256_setzero_pd();
    __m256d a2 = _mm256_setzero_pd();
    constexpr std::ptrdiff_t k = 64;
    std::ptrdiff_t counter = k;
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xh, xl; load_dd4(&x[i], xh, xl);
        /* t = x · inv  (DD mul, drop xl*invl) */
        __m256d th, tl;
        twoprod(xh, invh, th, tl);
        tl = _mm256_add_pd(tl,
                _mm256_add_pd(_mm256_mul_pd(xh, invl), _mm256_mul_pd(xl, invh)));
        /* p = t · t */
        __m256d ph, pl;
        twoprod(th, th, ph, pl);
        pl = _mm256_add_pd(pl,
                _mm256_add_pd(_mm256_mul_pd(th, tl), _mm256_mul_pd(tl, th)));
        __m256d e0, e1, e2;
        twosum(a0, ph, a0, e0);
        twosum(a1, pl, a1, e1);
        twosum(a1, e0, a1, e2);
        a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
        if (--counter == 0) {
            __m256d t, e;
            fast2sum(a1, a2, t, e);
            a1 = t; a2 = e;
            fast2sum(a0, a1, a0, a1);
            a1 = _mm256_add_pd(a1, a2);
            fast2sum(a0, a1, a0, a1);
            a2 = _mm256_setzero_pd();
            counter = k;
        }
    }
    __m256d t = _mm256_add_pd(a1, a2);
    s = horizontal_dd(a0, t);
    for (std::ptrdiff_t i = n4; i < n; ++i) { TR u = x[i] / scale; s = s + u * u; }
    return s;
}
#pragma GCC pop_options
#endif

static TR mnrm2_ssq_unit(std::ptrdiff_t n, const TR *x, TR scale)
{
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) return mnrm2_ssq_unit_simd(n, x, scale);
#endif
    TR s{0.0, 0.0};
    for (std::ptrdiff_t i = 0; i < n; ++i) { TR u = x[i] / scale; s = s + u * u; }
    return s;
}

#ifdef _OPENMP
/* Threaded two-pass nrm2 (incx==1). Pass 1 reduces max|x.hi| (exact → global
 * scale BIT-EXACT vs serial); pass 2 reduces Σ(x/scale)² in tid order via the
 * shared pre-initialized partial[]-reduce wrapper (mf_omp::partial_reduce; the
 * cross-slice summation reorders, so the squared-sum matches serial within
 * fuzz tol). Returns false (caller falls through to serial) when below the
 * threshold or already inside a parallel region. */
__attribute__((noinline)) static bool mnrm2_omp(std::ptrdiff_t n, const TR *x, TR *out)
{
    if (n <= MNRM2_OMP_MIN || !blas_omp_should_thread())
        return false;
    const std::ptrdiff_t nthreads = mf_omp::l1_team();

    double scale_hi = 0.0;
    #pragma omp parallel num_threads(nthreads) reduction(max:scale_hi)
    {
        std::ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) {
            double local = mnrm2_maxabs_unit(hi - lo, x + lo);
            if (local > scale_hi) scale_hi = local;
        }
    }
    if (scale_hi == 0.0) { *out = TR{0.0, 0.0}; return true; }
    TR scale{scale_hi, 0.0};

    TR s = mf_omp::partial_reduce(n, TR{0.0, 0.0},
        [x, scale](std::ptrdiff_t lo, std::ptrdiff_t hi) { return mnrm2_ssq_unit(hi - lo, x + lo, scale); },
        [](const TR &a, const TR &b) { return a + b; });
    *out = scale * sqrtdd(s);
    return true;
}
#endif

static TR mnrm2_core(std::ptrdiff_t n, const TR *x, std::ptrdiff_t incx)
{
    TR zero{0.0, 0.0};
    if (n < 1 || incx < 1) return zero;
    if (n == 1) return fabsdd(x[0]);

    if (incx == 1) {
#ifdef _OPENMP
        TR out;
        if (mnrm2_omp(n, x, &out)) return out;
#endif
        double scale_hi = mnrm2_maxabs_unit(n, x);
        if (scale_hi == 0.0) return zero;
        TR scale{scale_hi, 0.0};
        return scale * sqrtdd(mnrm2_ssq_unit(n, x, scale));
    }

    /* Strided fallback (scalar two-pass). */
    double scale_hi = 0.0;
    std::ptrdiff_t ix = 0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        TR ax = fabsdd(x[ix]);
        if (ax.limbs[0] > scale_hi) scale_hi = ax.limbs[0];
        ix += incx;
    }
    if (scale_hi == 0.0) return zero;
    TR scale{scale_hi, 0.0};
    TR s = zero;
    ix = 0;
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        TR t = x[ix] / scale;
        s = s + t * t;
        ix += incx;
    }
    return scale * sqrtdd(s);
}

extern "C" { EPBLAS_FACADE_ASUM(mnrm2, TR, TR) }
