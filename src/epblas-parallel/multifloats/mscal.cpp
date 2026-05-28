/*
 * mscal — multifloats real DD vector scale: X := α · X.
 *
 * 4-wide AVX2 SIMD path for INCX==1, scalar fallback for strided.
 */
#include <cstddef>
#include <multifloats.h>
#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline bool dd_isone(T x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }

#ifdef MBLAS_SIMD_DD
inline void load_4cell_soa(const T *p, __m256d &h, __m256d &l) {
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double*>(p));
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double*>(p + 2));
    __m256d lo = _mm256_unpacklo_pd(v0, v1);
    __m256d hi = _mm256_unpackhi_pd(v0, v1);
    h = _mm256_permute4x64_pd(lo, 0xD8);
    l = _mm256_permute4x64_pd(hi, 0xD8);
}
inline void store_4cell_soa(T *p, __m256d h, __m256d l) {
    __m256d lo = _mm256_unpacklo_pd(h, l);
    __m256d hi = _mm256_unpackhi_pd(h, l);
    __m256d v0 = _mm256_permute2f128_pd(lo, hi, 0x20);
    __m256d v1 = _mm256_permute2f128_pd(lo, hi, 0x31);
    _mm256_storeu_pd(reinterpret_cast<double*>(p),     v0);
    _mm256_storeu_pd(reinterpret_cast<double*>(p + 2), v1);
}
#endif
}  // namespace

extern "C" void mscal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const T alpha = *alpha_;
    if (n <= 0 || dd_isone(alpha)) return;

    if (incx == 1) {
#ifdef MBLAS_SIMD_DD
        const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
        const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
        const int n4 = n & ~3;
        for (int i = 0; i < n4; i += 4) {
            __m256d xh, xl;
            load_4cell_soa(&x[i], xh, xl);
            __m256d nh, nl;
            simd_dd::dd_mul(xh, xl, ah, al, nh, nl);
            store_4cell_soa(&x[i], nh, nl);
        }
        for (int i = n4; i < n; ++i) x[i] = x[i] * alpha;
#else
        for (int i = 0; i < n; ++i) x[i] = x[i] * alpha;
#endif
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) { x[ix] = x[ix] * alpha; ix += incx; }
    }
}
