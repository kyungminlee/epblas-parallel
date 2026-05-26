/*
 * wmscal — multifloats: X := α · X with α real DD, X complex DD.
 * Equivalent to BLAS CSSCAL / ZDSCAL pattern.
 */
#include <cstddef>
#include <multifloats.h>
#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline bool dd_isone(R x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }

#ifdef MBLAS_SIMD_DD
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
inline void store_4cell_csoa(T *p, __m256d rh, __m256d rl, __m256d ih, __m256d il) {
    __m256d t0 = _mm256_unpacklo_pd(rh, rl);
    __m256d t1 = _mm256_unpackhi_pd(rh, rl);
    __m256d t2 = _mm256_unpacklo_pd(ih, il);
    __m256d t3 = _mm256_unpackhi_pd(ih, il);
    _mm256_storeu_pd(reinterpret_cast<double*>(&p[0]), _mm256_permute2f128_pd(t0, t2, 0x20));
    _mm256_storeu_pd(reinterpret_cast<double*>(&p[1]), _mm256_permute2f128_pd(t1, t3, 0x20));
    _mm256_storeu_pd(reinterpret_cast<double*>(&p[2]), _mm256_permute2f128_pd(t0, t2, 0x31));
    _mm256_storeu_pd(reinterpret_cast<double*>(&p[3]), _mm256_permute2f128_pd(t1, t3, 0x31));
}
#endif
}  // namespace

extern "C" void wmscal_(const int *n_, const R *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const R alpha = *alpha_;
    if (n <= 0 || dd_isone(alpha)) return;

    if (incx == 1) {
#ifdef MBLAS_SIMD_DD
        const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
        const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
        const int n4 = n & ~3;
        for (int i = 0; i < n4; i += 4) {
            __m256d xrh, xrl, xih, xil;
            load_4cell_csoa(&x[i], xrh, xrl, xih, xil);
            /* α (real) × (xre + j·xim) → α·xre + j·α·xim — scale each limb-pair */
            __m256d nrh, nrl, nih, nil_;
            simd_dd::dd_mul(xrh, xrl, ah, al, nrh, nrl);
            simd_dd::dd_mul(xih, xil, ah, al, nih, nil_);
            store_4cell_csoa(&x[i], nrh, nrl, nih, nil_);
        }
        for (int i = n4; i < n; ++i) { x[i].re = x[i].re * alpha; x[i].im = x[i].im * alpha; }
#else
        for (int i = 0; i < n; ++i) { x[i].re = x[i].re * alpha; x[i].im = x[i].im * alpha; }
#endif
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) {
            x[ix].re = x[ix].re * alpha; x[ix].im = x[ix].im * alpha;
            ix += incx;
        }
    }
}
