/* mwnrm2 — multifloats: ||X||₂ for complex DD X, returns real DD.
 * Two-pass scaled. Pass 1 SIMD vmaxpd. Pass 2 SIMD wide-acc over
 * (re/scale)² + (im/scale)². */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>

namespace {
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
inline void simd_twoprod(__m256d a, __m256d b, __m256d &p, __m256d &e) {
    p = _mm256_mul_pd(a, b);
    e = _mm256_fmsub_pd(a, b, p);
}
inline void simd_twosum(__m256d a, __m256d b, __m256d &s, __m256d &e) {
    s = _mm256_add_pd(a, b);
    __m256d bb = _mm256_sub_pd(s, a);
    __m256d aa = _mm256_sub_pd(s, bb);
    e = _mm256_add_pd(_mm256_sub_pd(a, aa), _mm256_sub_pd(b, bb));
}
inline void simd_fast_twosum(__m256d a, __m256d b, __m256d &s, __m256d &e) {
    s = _mm256_add_pd(a, b);
    e = _mm256_sub_pd(b, _mm256_sub_pd(s, a));
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
    simd_twosum(a0, ph, a0, e0);
    simd_twosum(a1, pl, a1, e1);
    simd_twosum(a1, e0, a1, e2);
    a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
}
inline void renorm(__m256d &a0, __m256d &a1, __m256d &a2) {
    __m256d t, e;
    simd_fast_twosum(a1, a2, t, e);
    a1 = t; a2 = e;
    simd_fast_twosum(a0, a1, a0, a1);
    a1 = _mm256_add_pd(a1, a2);
    simd_fast_twosum(a0, a1, a0, a1);
    a2 = _mm256_setzero_pd();
}
}
#endif

extern "C" R mwnrm2_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    R zero{0.0, 0.0};
    if (n < 1 || incx < 1) return zero;

    /* Pass 1: scale = max(|re.hi|, |im.hi|). */
    double scale_hi = 0.0;
#ifdef MBLAS_SIMD_DD
    if (incx == 1) {
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
    } else
#endif
    {
        int ix = 0;
        for (int i = 0; i < n; ++i) {
            R ar = fabsdd(x[ix].re), ai = fabsdd(x[ix].im);
            if (ar.limbs[0] > scale_hi) scale_hi = ar.limbs[0];
            if (ai.limbs[0] > scale_hi) scale_hi = ai.limbs[0];
            ix += incx;
        }
    }
    if (scale_hi == 0.0) return zero;
    R scale{scale_hi, 0.0};

    /* Pass 2: sum (re/s)² + (im/s)² via wide-acc. */
    R s = zero;
#ifdef MBLAS_SIMD_DD
    if (incx == 1) {
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
            simd_twoprod(xh, invh, th, tl);
            tl = _mm256_add_pd(tl,
                    _mm256_add_pd(_mm256_mul_pd(xh, invl), _mm256_mul_pd(xl, invh)));
            /* p = t · t */
            __m256d ph, pl;
            simd_twoprod(th, th, ph, pl);
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
    } else
#endif
    {
        int ix = 0;
        for (int i = 0; i < n; ++i) {
            R r = x[ix].re / scale, m = x[ix].im / scale;
            s = s + r * r + m * m;
            ix += incx;
        }
    }
    return scale * sqrtdd(s);
}
