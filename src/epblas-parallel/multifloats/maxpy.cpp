/*
 * maxpy — multifloats real DD: Y := α · X + Y.
 *
 * 4-wide AVX2 SIMD path for INCX=INCY=1, scalar fallback otherwise.
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
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }

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

/* Y := α·X + Y over a contiguous unit-stride range — serial kernel, unchanged.
 * Carved out so the OpenMP path can run it per disjoint sub-range. */
static void maxpy_unit(int n, T alpha, const T *x, T *y)
{
#ifdef MBLAS_SIMD_DD
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d xh, xl, yh, yl;
        load_4cell_soa(&x[i], xh, xl);
        load_4cell_soa(&y[i], yh, yl);
        __m256d ph, pl;
        simd_dd::dd_mul(ah, al, xh, xl, ph, pl);
        __m256d nh, nl;
        simd_dd::dd_add(yh, yl, ph, pl, nh, nl);
        store_4cell_soa(&y[i], nh, nl);
    }
    for (int i = n4; i < n; ++i) y[i] = y[i] + alpha * x[i];
#else
    for (int i = 0; i < n; ++i) y[i] = y[i] + alpha * x[i];
#endif
}

#ifdef _OPENMP
/* Threaded elementwise AXPY. Outputs are disjoint per slice, so each thread
 * runs the SIMD kernel over its own [lo,hi) range — no reduction. DD math is
 * compute-bound, so this threads profitably above the crossover. */
#define MAXPY_OMP_MIN 2048
__attribute__((noinline)) static int maxpy_omp(int n, T alpha, const T *x, T *y)
{
    if (n <= MAXPY_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) maxpy_unit(hi - lo, alpha, x + lo, y + lo);
    }
    return 1;
}
#endif

extern "C" void maxpy_(const int *n_, const T *alpha_,
                       const T *x, const int *incx_,
                       T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    if (n <= 0 || dd_iszero(alpha)) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (maxpy_omp(n, alpha, x, y)) return;
#endif
        maxpy_unit(n, alpha, x, y);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] = y[iy] + alpha * x[ix]; ix += incx; iy += incy; }
    }
}
