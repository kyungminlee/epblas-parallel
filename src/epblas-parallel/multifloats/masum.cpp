/* masum — multifloats real DD: Σ |X|.
 *
 * SIMD Bailey-wide accumulator. Inner loop computes |x[i]| (per-cell DD abs)
 * and absorbs into the wide accumulator with periodic renorm.
 */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_fast.h"

namespace {
/* canonical EFTs — mf_simd_fast.h (2a-5) */
using simd_fast::fast2sum;
using simd_fast::twosum;
inline void load_4cell_soa(const T *p, __m256d &h, __m256d &l) {
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double*>(p));
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double*>(p + 2));
    __m256d lo = _mm256_unpacklo_pd(v0, v1);
    __m256d hi = _mm256_unpackhi_pd(v0, v1);
    h = _mm256_permute4x64_pd(lo, 0xD8);
    l = _mm256_permute4x64_pd(hi, 0xD8);
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

/* Σ|x| over a contiguous unit-stride range — the serial kernel, unchanged.
 * Carved out so the OpenMP partial-reduction can call it per sub-range. */
static T masum_unit(int n, const T *x)
{
#ifdef MBLAS_SIMD_DD
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
        twosum(a0, ph, a0, e0);
        twosum(a1, pl, a1, e1);
        twosum(a1, e0, a1, e2);
        a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
        if (--counter == 0) {
            __m256d t, e;
            fast2sum(a1, a2, t, e);
            a1 = t; a2 = e;
            fast2sum(a0, a1, a0, a1);
            a1 = _mm256_add_pd(a1, a2);
            fast2sum(a0, a1, a0, a1);
            a2 = _mm256_setzero_pd();
            counter = K;
        }
    }
    __m256d t = _mm256_add_pd(a1, a2);
    T s = horizontal_dd(a0, t);
    for (int i = n4; i < n; ++i) s = s + fabsdd(x[i]);
    return s;
#else
    T s0{0.0, 0.0}, s1{0.0, 0.0};
    int i = 0;
    for (; i + 1 < n; i += 2) {
        s0 = s0 + fabsdd(x[i]);
        s1 = s1 + fabsdd(x[i + 1]);
    }
    if (i < n) s0 = s0 + fabsdd(x[i]);
    return s0 + s1;
#endif
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride X. Each thread sums its
 * contiguous slice with the serial kernel; partials are combined in tid order.
 * Reduction order differs from serial → not bit-identical, but within fuzz
 * tolerance for a sum of magnitudes (same rationale as kind10 easum). */
#define MASUM_OMP_MIN 8192
#define MASUM_MAX_CPUS 64
__attribute__((noinline)) static int masum_omp(int n, const T *x, T *out)
{
    if (n <= MASUM_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MASUM_MAX_CPUS) nthreads = MASUM_MAX_CPUS;
    T partial[MASUM_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        partial[tid] = (lo < hi) ? masum_unit(hi - lo, x + lo) : T{0.0, 0.0};
    }
    T s{0.0, 0.0};
    for (int i = 0; i < nthreads; ++i) s = s + partial[i];
    *out = s;
    return 1;
}
#endif

extern "C" T masum_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    T s{0.0, 0.0};
    if (n < 1 || incx < 1) return s;

    if (incx == 1) {
#ifdef _OPENMP
        if (masum_omp(n, x, &s)) return s;
#endif
        return masum_unit(n, x);
    }

    T s0{0.0, 0.0}, s1{0.0, 0.0};
    for (int i = 0, ix = 0; i < n; ++i, ix += incx)
        s0 = s0 + fabsdd(x[ix]);
    return s0 + s1;
}
