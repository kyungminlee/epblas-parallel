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
#endif
#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
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
}

/* Givens rotation over a contiguous unit-stride range — serial kernel,
 * unchanged. X and Y slices are disjoint per thread → safe to partition. */
static void mrot_unit(int n, const T c, const T s, T *x, T *y)
{
#ifdef MBLAS_SIMD_DD
    const __m256d ch = _mm256_set1_pd(c.limbs[0]);
    const __m256d cl = _mm256_set1_pd(c.limbs[1]);
    const __m256d sh = _mm256_set1_pd(s.limbs[0]);
    const __m256d sl = _mm256_set1_pd(s.limbs[1]);
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d xh, xl, yh, yl;
        load_4cell_soa(&x[i], xh, xl);
        load_4cell_soa(&y[i], yh, yl);
        __m256d cxh, cxl; simd_dd::dd_mul(ch, cl, xh, xl, cxh, cxl);
        __m256d syh, syl; simd_dd::dd_mul(sh, sl, yh, yl, syh, syl);
        __m256d nxh, nxl; simd_dd::dd_add(cxh, cxl, syh, syl, nxh, nxl);
        __m256d cyh, cyl; simd_dd::dd_mul(ch, cl, yh, yl, cyh, cyl);
        __m256d sxh, sxl; simd_dd::dd_mul(sh, sl, xh, xl, sxh, sxl);
        simd_dd::dd_neg(sxh, sxl);
        __m256d nyh, nyl; simd_dd::dd_add(cyh, cyl, sxh, sxl, nyh, nyl);
        store_4cell_soa(&x[i], nxh, nxl);
        store_4cell_soa(&y[i], nyh, nyl);
    }
    for (int i = n4; i < n; ++i) {
        T tx = c * x[i] + s * y[i];
        y[i] = c * y[i] - s * x[i];
        x[i] = tx;
    }
#else
    for (int i = 0; i < n; ++i) {
        T tx = c * x[i] + s * y[i];
        y[i] = c * y[i] - s * x[i];
        x[i] = tx;
    }
#endif
}

#ifdef _OPENMP
#define MROT_OMP_MIN 2048
__attribute__((noinline)) static int mrot_omp(int n, T c, T s, T *x, T *y)
{
    if (n <= MROT_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) mrot_unit(hi - lo, c, s, x + lo, y + lo);
    }
    return 1;
}
#endif

extern "C" void mrot_(const int *n_,
                      T *x, const int *incx_,
                      T *y, const int *incy_,
                      const T *c_, const T *s_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T c = *c_, s = *s_;
    if (n <= 0) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (mrot_omp(n, c, s, x, y)) return;
#endif
        mrot_unit(n, c, s, x, y);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) {
            T tx = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = tx;
            ix += incx; iy += incy;
        }
    }
}
