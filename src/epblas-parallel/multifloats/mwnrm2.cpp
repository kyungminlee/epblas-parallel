/* mwnrm2 — multifloats: ||X||₂ for complex DD X, returns real DD.
 * Two-pass scaled. Pass 1 SIMD vmaxpd. Pass 2 SIMD wide-acc over
 * (re/scale)² + (im/scale)². */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#define MWNRM2_OMP_MIN 8192
#define MWNRM2_MAX_CPUS 64
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"

namespace {
/* canonical EFTs — mf_simd_fast.h (2a-5) */
using simd_fast::twoprod;
using simd_fast::fast2sum;
using simd_fast::twosum;
using simd_exact::cload4;
using simd_fast::horizontal_dd;  /* Bailey 2-limb finalizer — mf_simd_fast.h (#4) */
using simd_fast::absorb;  /* Bailey 3-limb wide-acc — mf_simd_fast.h (#4) */
using simd_fast::renorm3;  /* Bailey 3-limb wide-acc — mf_simd_fast.h (#4) */

}
#endif

/* Pass-1 unit kernel: max(|re.hi|, |im.hi|) over a contiguous (incx==1) slice.
 * Max is exact → the combined global scale is BIT-EXACT regardless of split. */
static double mwnrm2_maxabs_unit(int n, const T *x)
{
    double scale_hi = 0.0;
#ifdef MBLAS_SIMD_DD
    __m256d mx = _mm256_setzero_pd();
    const __m256d absmask = _mm256_castsi256_pd(
        _mm256_set1_epi64x(static_cast<long long>(0x7FFFFFFFFFFFFFFFULL)));
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d rh, rl, ih, il;
        cload4(&x[i], rh, rl, ih, il); (void)rl; (void)il;
        mx = _mm256_max_pd(mx, _mm256_and_pd(rh, absmask));
        mx = _mm256_max_pd(mx, _mm256_and_pd(ih, absmask));
    }
    alignas(32) double mxa[4];
    _mm256_store_pd(mxa, mx);
    for (int k = 0; k < 4; ++k) if (mxa[k] > scale_hi) scale_hi = mxa[k];
    for (int i = n4; i < n; ++i) {
        R ar = fabsdd(x[i].re), ai = fabsdd(x[i].im);
        if (ar.limbs[0] > scale_hi) scale_hi = ar.limbs[0];
        if (ai.limbs[0] > scale_hi) scale_hi = ai.limbs[0];
    }
#else
    for (int i = 0; i < n; ++i) {
        R ar = fabsdd(x[i].re), ai = fabsdd(x[i].im);
        if (ar.limbs[0] > scale_hi) scale_hi = ar.limbs[0];
        if (ai.limbs[0] > scale_hi) scale_hi = ai.limbs[0];
    }
#endif
    return scale_hi;
}

/* Pass-2 unit kernel: Σ (re/scale)² + (im/scale)² over a contiguous slice. */
static R mwnrm2_ssq_unit(int n, const T *x, R scale)
{
    R s{0.0, 0.0};
#ifdef MBLAS_SIMD_DD
    R inv = R{1.0, 0.0} / scale;
    __m256d invh = _mm256_set1_pd(inv.limbs[0]);
    __m256d invl = _mm256_set1_pd(inv.limbs[1]);
    __m256d a0 = _mm256_setzero_pd();
    __m256d a1 = _mm256_setzero_pd();
    __m256d a2 = _mm256_setzero_pd();
    constexpr int K = 64;
    int counter = K;
    const int n4 = n & ~3;
    auto sq_into = [&](__m256d xh, __m256d xl) {
        /* t = x · inv */
        __m256d th, tl;
        twoprod(xh, invh, th, tl);
        tl = _mm256_add_pd(tl,
                _mm256_add_pd(_mm256_mul_pd(xh, invl), _mm256_mul_pd(xl, invh)));
        /* p = t · t */
        __m256d ph, pl;
        twoprod(th, th, ph, pl);
        __m256d cross = _mm256_mul_pd(th, tl);
        pl = _mm256_add_pd(pl, _mm256_add_pd(cross, cross));
        absorb(ph, pl, a0, a1, a2);
    };
    for (int i = 0; i < n4; i += 4) {
        __m256d rh, rl, ih, il;
        cload4(&x[i], rh, rl, ih, il);
        sq_into(rh, rl);
        sq_into(ih, il);
        if (--counter == 0) { renorm3(a0, a1, a2); counter = K; }
    }
    __m256d t = _mm256_add_pd(a1, a2);
    s = horizontal_dd(a0, t);
    for (int i = n4; i < n; ++i) {
        R r = x[i].re / scale, m = x[i].im / scale;
        s = s + r * r + m * m;
    }
#else
    for (int i = 0; i < n; ++i) {
        R r = x[i].re / scale, m = x[i].im / scale;
        s = s + r * r + m * m;
    }
#endif
    return s;
}

#ifdef _OPENMP
/* Threaded two-pass nrm2 (incx==1): parallel max (exact → global scale
 * BIT-EXACT) then partial-reduce Σ(re²+im²)/scale² in tid order (reorders, so
 * matches serial within fuzz tol). Returns false below threshold. */
__attribute__((noinline)) static bool mwnrm2_omp(int n, const T *x, R *out)
{
    if (n <= MWNRM2_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return false;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MWNRM2_MAX_CPUS) nthreads = MWNRM2_MAX_CPUS;

    double scale_hi = 0.0;
    #pragma omp parallel num_threads(nthreads) reduction(max:scale_hi)
    {
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        int lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) {
            double local = mwnrm2_maxabs_unit(hi - lo, x + lo);
            if (local > scale_hi) scale_hi = local;
        }
    }
    if (scale_hi == 0.0) { *out = R{0.0, 0.0}; return true; }
    R scale{scale_hi, 0.0};

    R partial[MWNRM2_MAX_CPUS];
    for (int t = 0; t < nthreads; ++t) partial[t] = R{0.0, 0.0};
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num(), nth = omp_get_num_threads();
        int lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) partial[tid] = mwnrm2_ssq_unit(hi - lo, x + lo, scale);
    }
    R s{0.0, 0.0};
    for (int t = 0; t < nthreads; ++t) s = s + partial[t];
    *out = scale * sqrtdd(s);
    return true;
}
#endif

extern "C" R mwnrm2_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    R zero{0.0, 0.0};
    if (n < 1 || incx < 1) return zero;

    if (incx == 1) {
#ifdef _OPENMP
        R out;
        if (mwnrm2_omp(n, x, &out)) return out;
#endif
        double scale_hi = mwnrm2_maxabs_unit(n, x);
        if (scale_hi == 0.0) return zero;
        R scale{scale_hi, 0.0};
        return scale * sqrtdd(mwnrm2_ssq_unit(n, x, scale));
    }

    /* Strided fallback (scalar two-pass). */
    double scale_hi = 0.0;
    int ix = 0;
    for (int i = 0; i < n; ++i) {
        R ar = fabsdd(x[ix].re), ai = fabsdd(x[ix].im);
        if (ar.limbs[0] > scale_hi) scale_hi = ar.limbs[0];
        if (ai.limbs[0] > scale_hi) scale_hi = ai.limbs[0];
        ix += incx;
    }
    if (scale_hi == 0.0) return zero;
    R scale{scale_hi, 0.0};
    R s = zero;
    ix = 0;
    for (int i = 0; i < n; ++i) {
        R r = x[ix].re / scale, m = x[ix].im / scale;
        s = s + r * r + m * m;
        ix += incx;
    }
    return scale * sqrtdd(s);
}
