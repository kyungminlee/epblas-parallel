/* mwasum — multifloats: Σ (|re(X)| + |im(X)|) for complex DD X.
 * SIMD Bailey-wide accumulator over 4 complex cells per iter. */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>

namespace {
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
inline void simd_twosum(__m256d a, __m256d b, __m256d &s, __m256d &e) {
    s = _mm256_add_pd(a, b);
    __m256d bb = _mm256_sub_pd(s, a);
    __m256d aa = _mm256_sub_pd(s, bb);
    e = _mm256_add_pd(_mm256_sub_pd(a, aa), _mm256_sub_pd(b, bb));
}
inline void simd_fast_twosum(__m256d a, __m256d b, __m256d &s, __m256d &e) {
    s = _mm256_add_pd(a, b);
    e = _mm256_sub_pd(b, _mm256_sub_pd(s, a));
}
inline R horizontal_dd(__m256d h, __m256d l) {
    alignas(32) double ha[4], la[4];
    _mm256_store_pd(ha, h); _mm256_store_pd(la, l);
    R s{ha[0], la[0]};
    for (int k = 1; k < 4; ++k) s = s + R{ha[k], la[k]};
    return s;
}
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
        load_4cell_csoa(&x[i], rh, rl, ih, il);
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
        simd_twosum(arh, aih, ph, eh);
        pl = _mm256_add_pd(arl, ail);
        pl = _mm256_add_pd(pl, eh);
        /* Absorb into wide-acc */
        __m256d e0, e1, e2;
        simd_twosum(a0, ph, a0, e0);
        simd_twosum(a1, pl, a1, e1);
        simd_twosum(a1, e0, a1, e2);
        a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
        if (--counter == 0) {
            __m256d t, e;
            simd_fast_twosum(a1, a2, t, e);
            a1 = t; a2 = e;
            simd_fast_twosum(a0, a1, a0, a1);
            a1 = _mm256_add_pd(a1, a2);
            simd_fast_twosum(a0, a1, a0, a1);
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
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
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
