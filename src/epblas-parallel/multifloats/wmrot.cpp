/*
 * wmrot — multifloats: complex DD Givens rotation with real DD c, s.
 *   X' = c·X + s·Y
 *   Y' = c·Y - s·X
 * c, s are real DD; X, Y are complex DD.
 */
#include <cstddef>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#endif
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
#ifdef MBLAS_SIMD_DD
using simd_exact::cload4;
using simd_exact::cstore4;
#endif
}

/* Complex Givens rotation (real c,s) over a unit-stride range — serial kernel,
 * unchanged. X and Y slices are disjoint per thread → safe to partition. */
static void wmrot_unit(std::ptrdiff_t n, const R c, const R s, T *x, T *y)
{
#ifdef MBLAS_SIMD_DD
        const __m256d ch = _mm256_set1_pd(c.limbs[0]);
        const __m256d cl = _mm256_set1_pd(c.limbs[1]);
        const __m256d sh = _mm256_set1_pd(s.limbs[0]);
        const __m256d sl = _mm256_set1_pd(s.limbs[1]);
        const std::ptrdiff_t n4 = n & ~3;
        for (std::ptrdiff_t i = 0; i < n4; i += 4) {
            __m256d xrh, xrl, xih, xil, yrh, yrl, yih, yil;
            cload4(&x[i], xrh, xrl, xih, xil);
            cload4(&y[i], yrh, yrl, yih, yil);
            /* Real-scale and combine per limb-pair (re and im halves
             * are independent in real-c × complex-X). */
            auto apply = [&](__m256d xh, __m256d xl, __m256d yh, __m256d yl,
                             __m256d &nxh, __m256d &nxl, __m256d &nyh, __m256d &nyl) {
                __m256d cxh, cxl; simd_fast::mul(ch, cl, xh, xl, cxh, cxl);
                __m256d syh, syl; simd_fast::mul(sh, sl, yh, yl, syh, syl);
                simd_fast::add(cxh, cxl, syh, syl, nxh, nxl);
                __m256d cyh, cyl; simd_fast::mul(ch, cl, yh, yl, cyh, cyl);
                __m256d sxh, sxl; simd_fast::mul(sh, sl, xh, xl, sxh, sxl);
                simd_fast::neg(sxh, sxl);
                simd_fast::add(cyh, cyl, sxh, sxl, nyh, nyl);
            };
            __m256d nxrh, nxrl, nyrh, nyrl;
            apply(xrh, xrl, yrh, yrl, nxrh, nxrl, nyrh, nyrl);
            __m256d nxih, nxil, nyih, nyil;
            apply(xih, xil, yih, yil, nxih, nxil, nyih, nyil);
            cstore4(&x[i], nxrh, nxrl, nxih, nxil);
            cstore4(&y[i], nyrh, nyrl, nyih, nyil);
        }
        for (std::ptrdiff_t i = n4; i < n; ++i) {
            R nxr = c * x[i].re + s * y[i].re;
            R nxi = c * x[i].im + s * y[i].im;
            R nyr = c * y[i].re - s * x[i].re;
            R nyi = c * y[i].im - s * x[i].im;
            x[i].re = nxr; x[i].im = nxi;
            y[i].re = nyr; y[i].im = nyi;
        }
#else
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            R nxr = c * x[i].re + s * y[i].re;
            R nxi = c * x[i].im + s * y[i].im;
            R nyr = c * y[i].re - s * x[i].re;
            R nyi = c * y[i].im - s * x[i].im;
            x[i].re = nxr; x[i].im = nxi;
            y[i].re = nyr; y[i].im = nyi;
        }
#endif
}

#ifdef _OPENMP
#define WMROT_OMP_MIN 2048
__attribute__((noinline)) static std::ptrdiff_t wmrot_omp(std::ptrdiff_t n, R c, R s, T *x, T *y)
{
    if (n <= WMROT_OMP_MIN || !blas_omp_should_thread())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) wmrot_unit(hi - lo, c, s, x + lo, y + lo);
    }
    return 1;
}
#endif

static void wmrot_core(std::ptrdiff_t n,
                       T *x, std::ptrdiff_t incx,
                       T *y, std::ptrdiff_t incy,
                       const R *c_, const R *s_)
{
    const R c = *c_, s = *s_;
    if (n <= 0) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (wmrot_omp(n, c, s, x, y)) return;
#endif
        wmrot_unit(n, c, s, x, y);
    } else {
        std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) {
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

extern "C" {
EPBLAS_FACADE_ROT(wmrot, R, T)
}
