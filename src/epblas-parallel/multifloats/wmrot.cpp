/*
 * wmrot — multifloats: complex DD Givens rotation with real DD c, s.
 *   X' = c·X + s·Y
 *   Y' = c·Y - s·X
 * c, s are real DD; X, Y are complex DD.
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
}

extern "C" void wmrot_(const int *n_,
                       T *x, const int *incx_,
                       T *y, const int *incy_,
                       const R *c_, const R *s_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const R c = *c_, s = *s_;
    if (n <= 0) return;

    if (incx == 1 && incy == 1) {
#ifdef MBLAS_SIMD_DD
        const __m256d ch = _mm256_set1_pd(c.limbs[0]);
        const __m256d cl = _mm256_set1_pd(c.limbs[1]);
        const __m256d sh = _mm256_set1_pd(s.limbs[0]);
        const __m256d sl = _mm256_set1_pd(s.limbs[1]);
        const int n4 = n & ~3;
        for (int i = 0; i < n4; i += 4) {
            __m256d xrh, xrl, xih, xil, yrh, yrl, yih, yil;
            load_4cell_csoa(&x[i], xrh, xrl, xih, xil);
            load_4cell_csoa(&y[i], yrh, yrl, yih, yil);
            /* Real-scale and combine per limb-pair (re and im halves
             * are independent in real-c × complex-X). */
            auto apply = [&](__m256d xh, __m256d xl, __m256d yh, __m256d yl,
                             __m256d &nxh, __m256d &nxl, __m256d &nyh, __m256d &nyl) {
                __m256d cxh, cxl; simd_dd::dd_mul(ch, cl, xh, xl, cxh, cxl);
                __m256d syh, syl; simd_dd::dd_mul(sh, sl, yh, yl, syh, syl);
                simd_dd::dd_add(cxh, cxl, syh, syl, nxh, nxl);
                __m256d cyh, cyl; simd_dd::dd_mul(ch, cl, yh, yl, cyh, cyl);
                __m256d sxh, sxl; simd_dd::dd_mul(sh, sl, xh, xl, sxh, sxl);
                simd_dd::dd_neg(sxh, sxl);
                simd_dd::dd_add(cyh, cyl, sxh, sxl, nyh, nyl);
            };
            __m256d nxrh, nxrl, nyrh, nyrl;
            apply(xrh, xrl, yrh, yrl, nxrh, nxrl, nyrh, nyrl);
            __m256d nxih, nxil, nyih, nyil;
            apply(xih, xil, yih, yil, nxih, nxil, nyih, nyil);
            store_4cell_csoa(&x[i], nxrh, nxrl, nxih, nxil);
            store_4cell_csoa(&y[i], nyrh, nyrl, nyih, nyil);
        }
        for (int i = n4; i < n; ++i) {
            R nxr = c * x[i].re + s * y[i].re;
            R nxi = c * x[i].im + s * y[i].im;
            R nyr = c * y[i].re - s * x[i].re;
            R nyi = c * y[i].im - s * x[i].im;
            x[i].re = nxr; x[i].im = nxi;
            y[i].re = nyr; y[i].im = nyi;
        }
#else
        for (int i = 0; i < n; ++i) {
            R nxr = c * x[i].re + s * y[i].re;
            R nxi = c * x[i].im + s * y[i].im;
            R nyr = c * y[i].re - s * x[i].re;
            R nyi = c * y[i].im - s * x[i].im;
            x[i].re = nxr; x[i].im = nxi;
            y[i].re = nyr; y[i].im = nyi;
        }
#endif
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) {
            R nxr = c * x[ix].re + s * y[iy].re;
            R nxi = c * x[ix].im + s * y[iy].im;
            R nyr = c * y[iy].re - s * x[ix].re;
            R nyi = c * y[iy].im - s * x[ix].im;
            x[ix].re = nxr; x[ix].im = nxi;
            y[iy].re = nyr; y[iy].im = nyi;
            ix += incx; iy += incy;
        }
    }
}
