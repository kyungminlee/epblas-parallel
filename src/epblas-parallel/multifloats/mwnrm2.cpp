/* mwnrm2 — multifloats: ||X||₂ for complex DD X, returns real DD.
 * Two-pass scaled. Pass 1 SIMD vmaxpd. Pass 2 SIMD wide-acc over
 * (re/scale)² + (im/scale)². */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define MWNRM2_OMP_MIN 8192
#define MWNRM2_MAX_CPUS 64
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_fast.h"

namespace {
/* canonical EFTs — mf_simd_fast.h (2a-5) */
using simd_fast::twoprod;
using simd_fast::fast2sum;
using simd_fast::twosum;
inline void load_4cell_csoa(const T *p, __m256d &rh, __m256d &rl, __m256d &ih, __m256d &il) {
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double*>(&p[0]));
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double*>(&p[1]));
    __m256d v2 = _mm256_loadu_pd(reinterpret_cast<const double*>(&p[2]));
    __m256d v3 = _mm256_loadu_pd(reinterpret_cast<const double*>(&p[3]));
    __m256d t0 = _mm256_unpacklo_pd(v0, v1);
    __m256d t1 = _mm256_unpackhi_pd(v0, v1);
    __m256d t2 = _mm256_unpacklo_pd(v2, v3);
    __m256d t3 = _mm256_unpackhi_pd(v2, v3);
    rh = _mm256_permute2f128_pd(t0, t2, 0x20);
    rl = _mm256_permute2f128_pd(t1, t3, 0x20);
    ih = _mm256_permute2f128_pd(t0, t2, 0x31);
    il = _mm256_permute2f128_pd(t1, t3, 0x31);
}
inline R horizontal_dd(__m256d h, __m256d l) {
    alignas(32) double ha[4], la[4];
    _mm256_store_pd(ha, h); _mm256_store_pd(la, l);
    R s{ha[0], la[0]};
    for (int k = 1; k < 4; ++k) s = s + R{ha[k], la[k]};
    return s;
}

/* Absorb (ph, pl) into wide-acc (a0, a1, a2). */
inline void absorb(__m256d ph, __m256d pl,
                   __m256d &a0, __m256d &a1, __m256d &a2)
{
    __m256d e0, e1, e2;
    twosum(a0, ph, a0, e0);
    twosum(a1, pl, a1, e1);
    twosum(a1, e0, a1, e2);
    a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
}
inline void renorm(__m256d &a0, __m256d &a1, __m256d &a2) {
    __m256d t, e;
    fast2sum(a1, a2, t, e);
    a1 = t; a2 = e;
    fast2sum(a0, a1, a0, a1);
    a1 = _mm256_add_pd(a1, a2);
    fast2sum(a0, a1, a0, a1);
    a2 = _mm256_setzero_pd();
}
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
        load_4cell_csoa(&x[i], rh, rl, ih, il); (void)rl; (void)il;
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
        load_4cell_csoa(&x[i], rh, rl, ih, il);
        sq_into(rh, rl);
        sq_into(ih, il);
        if (--counter == 0) { renorm(a0, a1, a2); counter = K; }
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
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
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
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
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
