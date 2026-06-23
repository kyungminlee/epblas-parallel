/* mwasum — multifloats: Σ (|re(X)| + |im(X)|) for complex DD X.
 * SIMD Bailey-wide accumulator over 4 complex cells per iter. */
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
using R = mf::float64x2;
using TC = mf::complex64x2;

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"

namespace {
/* canonical EFTs — mf_simd_fast.h (2a-5) */
using simd_fast::fast2sum;
using simd_fast::twosum;
using simd_exact::cload4;
using simd_fast::horizontal_dd;  /* Bailey 2-limb finalizer — mf_simd_fast.h (#4) */
}
#endif

/* Σ(|re|+|im|) over a contiguous unit-stride range — serial kernel, unchanged.
 * Carved out so the OpenMP partial-reduction can call it per sub-range. */
static R mwasum_unit(std::ptrdiff_t n, const TC *x)
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
        __m256d rh, rl, ih, il;
        cload4(&x[i], rh, rl, ih, il);
        /* |re|: clear sign on hi, xor lo */
        __m256d sgr = _mm256_and_pd(rh, signbit);
        __m256d arh = _mm256_andnot_pd(signbit, rh);
        __m256d arl = _mm256_xor_pd(rl, sgr);
        /* |im|: same */
        __m256d sgi = _mm256_and_pd(ih, signbit);
        __m256d aih = _mm256_andnot_pd(signbit, ih);
        __m256d ail = _mm256_xor_pd(il, sgi);
        /* sum_pair = |re| + |im| as DD via twosum */
        __m256d ph, pl, eh;
        twosum(arh, aih, ph, eh);
        pl = _mm256_add_pd(arl, ail);
        pl = _mm256_add_pd(pl, eh);
        /* Absorb into wide-acc */
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
    R s = horizontal_dd(a0, t);
    for (std::ptrdiff_t i = n4; i < n; ++i) s = s + fabsdd(x[i].re) + fabsdd(x[i].im);
    return s;
#else
    R s{0.0, 0.0};
    for (std::ptrdiff_t i = 0; i < n; ++i) s = s + fabsdd(x[i].re) + fabsdd(x[i].im);
    return s;
#endif
}

#ifdef _OPENMP
#define MWASUM_OMP_MIN 8192
#define MWASUM_MAX_CPUS 64
__attribute__((noinline)) static std::ptrdiff_t mwasum_omp(std::ptrdiff_t n, const TC *x, R *out)
{
    if (n <= MWASUM_OMP_MIN || !blas_omp_should_thread())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > MWASUM_MAX_CPUS) nthreads = MWASUM_MAX_CPUS;
    R partial[MWASUM_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        partial[tid] = (lo < hi) ? mwasum_unit(hi - lo, x + lo) : R{0.0, 0.0};
    }
    R s{0.0, 0.0};
    for (std::ptrdiff_t i = 0; i < nthreads; ++i) s = s + partial[i];
    *out = s;
    return 1;
}
#endif

static R mwasum_core(std::ptrdiff_t n, const TC *x, std::ptrdiff_t incx)
{
    R s{0.0, 0.0};
    if (n < 1 || incx < 1) return s;

    if (incx == 1) {
#ifdef _OPENMP
        if (mwasum_omp(n, x, &s)) return s;
#endif
        return mwasum_unit(n, x);
    }

    for (std::ptrdiff_t i = 0, ix = 0; i < n; ++i, ix += incx)
        s = s + fabsdd(x[ix].re) + fabsdd(x[ix].im);
    return s;
}

extern "C" { EPBLAS_FACADE_ASUM(mwasum, R, TC) }
