/* masum — multifloats real DD: Σ |X|.
 *
 * SIMD Bailey-wide accumulator. Inner loop computes |x[i]| (per-cell DD abs)
 * and absorbs into the wide accumulator with periodic renorm.
 */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using T = mf::float64x2;

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

extern "C" T masum_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    T s{0.0, 0.0};
    if (n < 1 || incx < 1) return s;

#ifdef MBLAS_SIMD_DD
    if (incx == 1) {
        __m256d a0 = _mm256_setzero_pd();
        __m256d a1 = _mm256_setzero_pd();
        __m256d a2 = _mm256_setzero_pd();
        const __m256d signbit = _mm256_castsi256_pd(
            _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
        constexpr int K = 64;
        int counter = K;
        const int n4 = n & ~3;
        for (int i = 0; i < n4; i += 4) {
            __m256d xh, xl;
            load_4cell_soa(&x[i], xh, xl);
            /* DD abs: clear sign on hi, conditionally flip lo */
            __m256d sg = _mm256_and_pd(xh, signbit);
            __m256d ph = _mm256_andnot_pd(signbit, xh);
            __m256d pl = _mm256_xor_pd(xl, sg);
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
        for (int i = n4; i < n; ++i) s = s + fabsdd(x[i]);
        return s;
    }
#endif
    T s0{0.0, 0.0}, s1{0.0, 0.0};
    if (incx == 1) {
        int i = 0;
        for (; i + 1 < n; i += 2) {
            s0 = s0 + fabsdd(x[i]);
            s1 = s1 + fabsdd(x[i + 1]);
        }
        if (i < n) s0 = s0 + fabsdd(x[i]);
    } else {
        for (int i = 0, ix = 0; i < n; ++i, ix += incx)
            s0 = s0 + fabsdd(x[ix]);
    }
    return s0 + s1;
}
