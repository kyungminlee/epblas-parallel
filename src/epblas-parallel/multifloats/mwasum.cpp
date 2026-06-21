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

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

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
static R mwasum_unit(int n, const T *x)
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
            counter = K;
        }
    }
    __m256d t = _mm256_add_pd(a1, a2);
    R s = horizontal_dd(a0, t);
    for (int i = n4; i < n; ++i) s = s + fabsdd(x[i].re) + fabsdd(x[i].im);
    return s;
#else
    R s{0.0, 0.0};
    for (int i = 0; i < n; ++i) s = s + fabsdd(x[i].re) + fabsdd(x[i].im);
    return s;
#endif
}

#ifdef _OPENMP
#define MWASUM_OMP_MIN 8192
#define MWASUM_MAX_CPUS 64
__attribute__((noinline)) static int mwasum_omp(int n, const T *x, R *out)
{
    if (n <= MWASUM_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MWASUM_MAX_CPUS) nthreads = MWASUM_MAX_CPUS;
    R partial[MWASUM_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        partial[tid] = (lo < hi) ? mwasum_unit(hi - lo, x + lo) : R{0.0, 0.0};
    }
    R s{0.0, 0.0};
    for (int i = 0; i < nthreads; ++i) s = s + partial[i];
    *out = s;
    return 1;
}
#endif

extern "C" R mwasum_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    R s{0.0, 0.0};
    if (n < 1 || incx < 1) return s;

    if (incx == 1) {
#ifdef _OPENMP
        if (mwasum_omp(n, x, &s)) return s;
#endif
        return mwasum_unit(n, x);
    }

    for (int i = 0, ix = 0; i < n; ++i, ix += incx)
        s = s + fabsdd(x[ix].re) + fabsdd(x[ix].im);
    return s;
}
