/*
 * wscal — multifloats complex DD: X := α · X (α complex).
 */
#include <cstddef>
#include <multifloats.h>
#include "mf_pred.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq1;
namespace {
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}

#ifdef MBLAS_SIMD_DD
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
inline void store_4cell_csoa(T *p, __m256d rh, __m256d rl, __m256d ih, __m256d il) {
    __m256d t0 = _mm256_unpacklo_pd(rh, rl);
    __m256d t1 = _mm256_unpackhi_pd(rh, rl);
    __m256d t2 = _mm256_unpacklo_pd(ih, il);
    __m256d t3 = _mm256_unpackhi_pd(ih, il);
    _mm256_storeu_pd(reinterpret_cast<double*>(&p[0]), _mm256_permute2f128_pd(t0, t2, 0x20));
    _mm256_storeu_pd(reinterpret_cast<double*>(&p[1]), _mm256_permute2f128_pd(t1, t3, 0x20));
    _mm256_storeu_pd(reinterpret_cast<double*>(&p[2]), _mm256_permute2f128_pd(t0, t2, 0x31));
    _mm256_storeu_pd(reinterpret_cast<double*>(&p[3]), _mm256_permute2f128_pd(t1, t3, 0x31));
}
#endif
}  // namespace

/* X := α·X over a contiguous unit-stride range — serial kernel, unchanged. */
static void wscal_unit(int n, T alpha, T *x)
{
#ifdef MBLAS_SIMD_DD
    const __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    const __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    const __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    const __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d xrh, xrl, xih, xil;
        load_4cell_csoa(&x[i], xrh, xrl, xih, xil);
        __m256d nrh, nrl, nih, nil_;
        simd_fast::cmul(xrh, xrl, xih, xil, arh, arl, aih, ail,
                         nrh, nrl, nih, nil_);
        store_4cell_csoa(&x[i], nrh, nrl, nih, nil_);
    }
    for (int i = n4; i < n; ++i) x[i] = cmul(x[i], alpha);
#else
    for (int i = 0; i < n; ++i) x[i] = cmul(x[i], alpha);
#endif
}

#ifdef _OPENMP
#define WSCAL_OMP_MIN 2048
__attribute__((noinline)) static int wscal_omp(int n, T alpha, T *x)
{
    if (n <= WSCAL_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) wscal_unit(hi - lo, alpha, x + lo);
    }
    return 1;
}
#endif

extern "C" void wscal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const T alpha = *alpha_;
    if (n <= 0 || ceq1(alpha)) return;

    if (incx == 1) {
#ifdef _OPENMP
        if (wscal_omp(n, alpha, x)) return;
#endif
        wscal_unit(n, alpha, x);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) { x[ix] = cmul(x[ix], alpha); ix += incx; }
    }
}
