/* mdot — multifloats real DD: Σ X·Y.
 *
 * SIMD Bailey-wide accumulator (V5). 4-wide SIMD with a 3-double-per-lane
 * wide accumulator (a0, a1, a2). Periodic renorm every K=64 iters keeps the
 * accumulator stable. Final horizontal reduce.
 *
 * Bench (vs scalar full-DD):
 *   N=64..1024 : 1.0–1.4× (small-N dispatch dominates)
 *   N=8K..1M   : ~5× faster, full DD precision (~30–32 digits)
 *
 * See V3 vs V5 study: Bailey-wide consistently beats SIMD full-DD on both
 * speed and precision; Kahan-defer was rejected because it collapses to
 * double precision.
 */
#include <cstddef>
#include <multifloats.h>

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

extern "C" T mdot_(const int *n_,
                   const T *x, const int *incx_,
                   const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    T s{0.0, 0.0};
    if (n <= 0) return s;

#ifdef MBLAS_SIMD_DD
    if (incx == 1 && incy == 1) {
        __m256d a0 = _mm256_setzero_pd();
        __m256d a1 = _mm256_setzero_pd();
        __m256d a2 = _mm256_setzero_pd();
        constexpr int K = 64;
        int counter = K;
        const int n4 = n & ~3;
        for (int i = 0; i < n4; i += 4) {
            __m256d xh, xl, yh, yl;
            load_4cell_soa(&x[i], xh, xl);
            load_4cell_soa(&y[i], yh, yl);
            /* DD product via simplified twoprod: drop xl*yl (~2^-106 of xh*yh) */
            __m256d ph, pl;
            simd_twoprod(xh, yh, ph, pl);
            pl = _mm256_add_pd(pl,
                    _mm256_add_pd(_mm256_mul_pd(xh, yl), _mm256_mul_pd(xl, yh)));
            /* Wide-acc absorb (a0, a1, a2) += (ph, pl) */
            __m256d e0, e1, e2;
            simd_twosum(a0, ph, a0, e0);
            simd_twosum(a1, pl, a1, e1);
            simd_twosum(a1, e0, a1, e2);
            a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
            if (--counter == 0) {
                /* Renormalize 3→2 doubles */
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
        for (int i = n4; i < n; ++i) s = s + x[i] * y[i];
        return s;
    }
#endif
    /* Strided / scalar fallback with 2-acc unroll */
    {
        T s0{0.0, 0.0}, s1{0.0, 0.0};
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        int i = 0;
        if (incx == 1 && incy == 1) {
            for (; i + 1 < n; i += 2) {
                s0 = s0 + x[i]     * y[i];
                s1 = s1 + x[i + 1] * y[i + 1];
            }
            s = s0 + s1;
            for (; i < n; ++i) s = s + x[i] * y[i];
        } else {
            for (; i < n; ++i) { s = s + x[ix] * y[iy]; ix += incx; iy += incy; }
        }
    }
    return s;
}
