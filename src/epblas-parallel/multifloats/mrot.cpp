/*
 * mrot — multifloats real DD Givens rotation:
 *   X' = c·X + s·Y
 *   Y' = c·Y - s·X
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
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;

namespace {
#ifdef MBLAS_SIMD_DD
using simd_exact::load_dd4;
using simd_exact::store_dd4;
#endif
}

#ifdef MBLAS_SIMD_DD
/* AVX2+FMA kernel body — compiled under target("avx2,fma") so it builds even
 * when the library's baseline -march is pre-Haswell; reached only behind the
 * mf_have_avx2_fma() runtime probe in mrot_unit below. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static void mrot_unit_simd(std::ptrdiff_t n, const TR c, const TR s, TR *x, TR *y)
{
    const __m256d ch = _mm256_set1_pd(c.limbs[0]);
    const __m256d cl = _mm256_set1_pd(c.limbs[1]);
    const __m256d sh = _mm256_set1_pd(s.limbs[0]);
    const __m256d sl = _mm256_set1_pd(s.limbs[1]);
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xh, xl, yh, yl;
        load_dd4(&x[i], xh, xl);
        load_dd4(&y[i], yh, yl);
        __m256d cxh, cxl; simd_fast::mul(ch, cl, xh, xl, cxh, cxl);
        __m256d syh, syl; simd_fast::mul(sh, sl, yh, yl, syh, syl);
        __m256d nxh, nxl; simd_fast::add(cxh, cxl, syh, syl, nxh, nxl);
        __m256d cyh, cyl; simd_fast::mul(ch, cl, yh, yl, cyh, cyl);
        __m256d sxh, sxl; simd_fast::mul(sh, sl, xh, xl, sxh, sxl);
        simd_fast::neg(sxh, sxl);
        __m256d nyh, nyl; simd_fast::add(cyh, cyl, sxh, sxl, nyh, nyl);
        store_dd4(&x[i], nxh, nxl);
        store_dd4(&y[i], nyh, nyl);
    }
    for (std::ptrdiff_t i = n4; i < n; ++i) {
        TR tx = c * x[i] + s * y[i];
        y[i] = c * y[i] - s * x[i];
        x[i] = tx;
    }
}
#pragma GCC pop_options
#endif

/* Givens rotation over a contiguous unit-stride range — serial kernel. Runtime
 * dispatch: SIMD on Haswell+, scalar (always compiled) on Sandybridge/Ivy.
 * X and Y slices are disjoint per thread → safe to partition. */
static void mrot_unit(std::ptrdiff_t n, const TR c, const TR s, TR *x, TR *y)
{
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) { mrot_unit_simd(n, c, s, x, y); return; }
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        TR tx = c * x[i] + s * y[i];
        y[i] = c * y[i] - s * x[i];
        x[i] = tx;
    }
}

#ifdef _OPENMP
#define MROT_OMP_MIN 2048
__attribute__((noinline)) static std::ptrdiff_t mrot_omp(std::ptrdiff_t n, TR c, TR s, TR *x, TR *y)
{
    if (n <= MROT_OMP_MIN || !blas_omp_should_thread())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) mrot_unit(hi - lo, c, s, x + lo, y + lo);
    }
    return 1;
}
#endif

static void mrot_core(std::ptrdiff_t n,
                      TR *x, std::ptrdiff_t incx,
                      TR *y, std::ptrdiff_t incy,
                      const TR *c_, const TR *s_)
{
    const TR c = *c_, s = *s_;
    if (n <= 0) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (mrot_omp(n, c, s, x, y)) return;
#endif
        mrot_unit(n, c, s, x, y);
    } else {
        std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            TR tx = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = tx;
            ix += incx; iy += incy;
        }
    }
}

extern "C" { EPBLAS_FACADE_ROT(mrot, TR, TR) }
