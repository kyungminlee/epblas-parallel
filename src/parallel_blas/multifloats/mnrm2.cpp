/* mnrm2 — multifloats real DD: ||X||₂ via two-pass scaled.
 *
 * Pass 1 (SIMD vmaxpd): find scale = max(|x[i].hi|).
 * Pass 2 (SIMD wide-acc): Σ (x[i]/scale)² then scale·√sum.
 */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_gt(T a, T b) {
    return a.limbs[0] > b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] > b.limbs[1]);
}
}

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>

namespace {
inline void load_4cell_soa(const T *p, __m256d &h, __m256d &l) {
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double*>(p));
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double*>(p + 2));
    __m256d lo = _mm256_unpacklo_pd(v0, v1);
    __m256d hi = _mm256_unpackhi_pd(v0, v1);
    h = _mm256_permute4x64_pd(lo, 0xD8);
    l = _mm256_permute4x64_pd(hi, 0xD8);
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
inline T horizontal_dd(__m256d h, __m256d l) {
    alignas(32) double ha[4], la[4];
    _mm256_store_pd(ha, h); _mm256_store_pd(la, l);
    T s{ha[0], la[0]};
    for (int k = 1; k < 4; ++k) s = s + T{ha[k], la[k]};
    return s;
}
}
#endif

extern "C" T mnrm2_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    T zero{0.0, 0.0};
    if (n < 1 || incx < 1) return zero;
    if (n == 1) return fabsdd(x[0]);

    /* Pass 1: scale = max|x.hi| (sufficient for overflow protection). */
    double scale_hi = 0.0;
    int ix = 0;
#ifdef MBLAS_SIMD_DD
    if (incx == 1) {
        __m256d mx = _mm256_setzero_pd();
        const __m256d absmask = _mm256_castsi256_pd(
            _mm256_set1_epi64x(static_cast<long long>(0x7FFFFFFFFFFFFFFFULL)));
        const int n4 = n & ~3;
        for (int i = 0; i < n4; i += 4) {
            __m256d xh, xl; load_4cell_soa(&x[i], xh, xl); (void)xl;
            __m256d ax = _mm256_and_pd(xh, absmask);
            mx = _mm256_max_pd(mx, ax);
        }
        alignas(32) double mxa[4];
        _mm256_store_pd(mxa, mx);
        for (int k = 0; k < 4; ++k) if (mxa[k] > scale_hi) scale_hi = mxa[k];
        for (int i = n4; i < n; ++i) {
            T ax = fabsdd(x[i]);
            if (ax.limbs[0] > scale_hi) scale_hi = ax.limbs[0];
        }
    } else
#endif
    {
        for (int i = 0; i < n; ++i) {
            T ax = fabsdd(x[ix]);
            if (ax.limbs[0] > scale_hi) scale_hi = ax.limbs[0];
            ix += incx;
        }
    }
    if (scale_hi == 0.0) return zero;
    T scale{scale_hi, 0.0};

    /* Pass 2: sum (x/scale)² via wide-acc. */
    T s = zero;
#ifdef MBLAS_SIMD_DD
    if (incx == 1) {
        /* Precompute inverse so the inner loop avoids dd_div. */
        T inv = T{1.0, 0.0} / scale;
        __m256d invh = _mm256_set1_pd(inv.limbs[0]);
        __m256d invl = _mm256_set1_pd(inv.limbs[1]);
        __m256d a0 = _mm256_setzero_pd();
        __m256d a1 = _mm256_setzero_pd();
        __m256d a2 = _mm256_setzero_pd();
        constexpr int K = 64;
        int counter = K;
        const int n4 = n & ~3;
        for (int i = 0; i < n4; i += 4) {
            __m256d xh, xl; load_4cell_soa(&x[i], xh, xl);
            /* t = x · inv  (DD mul, drop xl*invl) */
            __m256d th, tl;
            simd_twoprod(xh, invh, th, tl);
            tl = _mm256_add_pd(tl,
                    _mm256_add_pd(_mm256_mul_pd(xh, invl), _mm256_mul_pd(xl, invh)));
            /* p = t · t */
            __m256d ph, pl;
            simd_twoprod(th, th, ph, pl);
            pl = _mm256_add_pd(pl,
                    _mm256_add_pd(_mm256_mul_pd(th, tl), _mm256_mul_pd(tl, th)));
            __m256d e0, e1, e2;
            simd_twosum(a0, ph, a0, e0);
            simd_twosum(a1, pl, a1, e1);
            simd_twosum(a1, e0, a1, e2);
            a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
            if (--counter == 0) {
                __m256d t, e;
                simd_fast_twosum(a1, a2, t, e);
                a1 = t; a2 = e;
                simd_fast_twosum(a0, a1, a0, a1);
                a1 = _mm256_add_pd(a1, a2);
                simd_fast_twosum(a0, a1, a0, a1);
                a2 = _mm256_setzero_pd();
                counter = K;
            }
        }
        __m256d t = _mm256_add_pd(a1, a2);
        s = horizontal_dd(a0, t);
        for (int i = n4; i < n; ++i) { T u = x[i] / scale; s = s + u * u; }
    } else
#endif
    {
        ix = 0;
        for (int i = 0; i < n; ++i) {
            T t = x[ix] / scale;
            s = s + t * t;
            ix += incx;
        }
    }
    return scale * sqrtdd(s);
}
