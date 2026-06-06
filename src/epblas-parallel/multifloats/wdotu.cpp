/* wdotu — multifloats complex DD: Σ X·Y (unconjugated).
 * Two SIMD Bailey-wide accumulators (re, im), each 3-double-per-lane. */
#include <cstddef>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
}

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
inline void simd_twoprod(__m256d a, __m256d b, __m256d &p, __m256d &e) {
    p = _mm256_mul_pd(a, b);
    e = _mm256_fmsub_pd(a, b, p);
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
inline void absorb(__m256d ph, __m256d pl,
                   __m256d &a0, __m256d &a1, __m256d &a2)
{
    __m256d e0, e1, e2;
    simd_twosum(a0, ph, a0, e0);
    simd_twosum(a1, pl, a1, e1);
    simd_twosum(a1, e0, a1, e2);
    a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
}
inline void renorm3(__m256d &a0, __m256d &a1, __m256d &a2) {
    __m256d t, e;
    simd_fast_twosum(a1, a2, t, e);
    a1 = t; a2 = e;
    simd_fast_twosum(a0, a1, a0, a1);
    a1 = _mm256_add_pd(a1, a2);
    simd_fast_twosum(a0, a1, a0, a1);
    a2 = _mm256_setzero_pd();
}
/* DD product (drop xl*yl): outputs (ph, pl). */
inline void dd_prod(__m256d xh, __m256d xl, __m256d yh, __m256d yl,
                    __m256d &ph, __m256d &pl) {
    simd_twoprod(xh, yh, ph, pl);
    pl = _mm256_add_pd(pl,
            _mm256_add_pd(_mm256_mul_pd(xh, yl), _mm256_mul_pd(xl, yh)));
}
}
#endif

/* Σ X·Y over contiguous unit-stride ranges — serial kernel, unchanged.
 * Carved out so the OpenMP partial-reduction can call it per sub-range. */
static T wdotu_unit(int n, const T *x, const T *y)
{
    T s{R{0.0, 0.0}, R{0.0, 0.0}};
#ifdef MBLAS_SIMD_DD
    __m256d rA0 = _mm256_setzero_pd(), rA1 = _mm256_setzero_pd(), rA2 = _mm256_setzero_pd();
    __m256d iA0 = _mm256_setzero_pd(), iA1 = _mm256_setzero_pd(), iA2 = _mm256_setzero_pd();
    constexpr int K = 64;
    int counter = K;
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d xrh, xrl, xih, xil, yrh, yrl, yih, yil;
        load_4cell_csoa(&x[i], xrh, xrl, xih, xil);
        load_4cell_csoa(&y[i], yrh, yrl, yih, yil);
        /* re(prod) = xr·yr - xi·yi ; im(prod) = xr·yi + xi·yr */
        __m256d rh, rl, ph, pl;
        dd_prod(xrh, xrl, yrh, yrl, rh, rl);
        dd_prod(xih, xil, yih, yil, ph, pl);
        /* Absorb +(rh, rl) into re-acc, then -(ph, pl) */
        absorb(rh, rl, rA0, rA1, rA2);
        __m256d nph = _mm256_sub_pd(_mm256_setzero_pd(), ph);
        __m256d npl = _mm256_sub_pd(_mm256_setzero_pd(), pl);
        absorb(nph, npl, rA0, rA1, rA2);
        /* im: +xr·yi +xi·yr */
        dd_prod(xrh, xrl, yih, yil, rh, rl);
        dd_prod(xih, xil, yrh, yrl, ph, pl);
        absorb(rh, rl, iA0, iA1, iA2);
        absorb(ph, pl, iA0, iA1, iA2);
        if (--counter == 0) {
            renorm3(rA0, rA1, rA2);
            renorm3(iA0, iA1, iA2);
            counter = K;
        }
    }
    __m256d rt = _mm256_add_pd(rA1, rA2);
    __m256d it = _mm256_add_pd(iA1, iA2);
    s.re = horizontal_dd(rA0, rt);
    s.im = horizontal_dd(iA0, it);
    for (int i = n4; i < n; ++i) s = cadd(s, cmul(x[i], y[i]));
    return s;
#else
    for (int i = 0; i < n; ++i) s = cadd(s, cmul(x[i], y[i]));
    return s;
#endif
}

#ifdef _OPENMP
#define WDOTU_OMP_MIN 8192
#define WDOTU_MAX_CPUS 64
__attribute__((noinline)) static int wdotu_omp(int n, const T *x, const T *y, T *out)
{
    if (n <= WDOTU_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > WDOTU_MAX_CPUS) nthreads = WDOTU_MAX_CPUS;
    T partial[WDOTU_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        partial[tid] = (lo < hi) ? wdotu_unit(hi - lo, x + lo, y + lo)
                                 : T{R{0.0, 0.0}, R{0.0, 0.0}};
    }
    T s{R{0.0, 0.0}, R{0.0, 0.0}};
    for (int i = 0; i < nthreads; ++i) s = cadd(s, partial[i]);
    *out = s;
    return 1;
}
#endif

extern "C" T wdotu_(const int *n_,
                    const T *x, const int *incx_,
                    const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    T s{R{0.0, 0.0}, R{0.0, 0.0}};
    if (n <= 0) return s;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (wdotu_omp(n, x, y, &s)) return s;
#endif
        return wdotu_unit(n, x, y);
    }

    int ix = (incx < 0) ? (-n + 1) * incx : 0;
    int iy = (incy < 0) ? (-n + 1) * incy : 0;
    for (int i = 0; i < n; ++i) { s = cadd(s, cmul(x[ix], y[iy])); ix += incx; iy += incy; }
    return s;
}
