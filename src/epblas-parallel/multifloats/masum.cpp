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
#include "mf_omp.h"
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"

namespace {
/* canonical EFTs — mf_simd_fast.h (2a-5) */
using simd_fast::fast2sum;
using simd_fast::twosum;
using simd_exact::load_dd4;
using simd_fast::horizontal_dd;  /* Bailey 2-limb finalizer — mf_simd_fast.h (#4) */
}
#endif

/* Σ|x| over a contiguous unit-stride range — the serial kernel, unchanged.
 * Carved out so the OpenMP partial-reduction can call it per sub-range. */
static TR masum_unit(std::ptrdiff_t n, const TR *x)
{
#ifdef MBLAS_SIMD_DD
    __m256d a0 = _mm256_setzero_pd();
    __m256d a1 = _mm256_setzero_pd();
    __m256d a2 = _mm256_setzero_pd();
    const __m256d signbit = _mm256_castsi256_pd(
        _mm256_set1_epi64x(static_cast<long long>(0x8000000000000000ULL)));
    constexpr std::ptrdiff_t k = 64;
    std::ptrdiff_t counter = k;
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xh, xl;
        load_dd4(&x[i], xh, xl);
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
            counter = k;
        }
    }
    __m256d t = _mm256_add_pd(a1, a2);
    TR s = horizontal_dd(a0, t);
    for (std::ptrdiff_t i = n4; i < n; ++i) s = s + fabsdd(x[i]);
    return s;
#else
    TR s0{0.0, 0.0}, s1{0.0, 0.0};
    std::ptrdiff_t i = 0;
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
__attribute__((noinline)) static std::ptrdiff_t masum_omp(std::ptrdiff_t n, const TR *x, TR *out)
{
    if (n <= MASUM_OMP_MIN || !blas_omp_should_thread())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > MASUM_MAX_CPUS) nthreads = MASUM_MAX_CPUS;
    TR partial[MASUM_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        partial[tid] = (lo < hi) ? masum_unit(hi - lo, x + lo) : TR{0.0, 0.0};
    }
    TR s{0.0, 0.0};
    for (std::ptrdiff_t i = 0; i < nthreads; ++i) s = s + partial[i];
    *out = s;
    return 1;
}
#endif

static TR masum_core(std::ptrdiff_t n, const TR *x, std::ptrdiff_t incx)
{
    TR s{0.0, 0.0};
    if (n < 1 || incx < 1) return s;

    if (incx == 1) {
#ifdef _OPENMP
        if (masum_omp(n, x, &s)) return s;
#endif
        return masum_unit(n, x);
    }

    TR s0{0.0, 0.0}, s1{0.0, 0.0};
    for (std::ptrdiff_t i = 0, ix = 0; i < n; ++i, ix += incx)
        s0 = s0 + fabsdd(x[ix]);
    return s0 + s1;
}

extern "C" { EPBLAS_FACADE_ASUM(masum, TR, TR) }
