/* wgerc — multifloats complex DD conjugated rank-1.
 * Same SIMD structure as wgeru with t = alpha * conj(y[j]). */

#include <cstddef>
#include <cstdlib>
#include <multifloats.h>
#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
#define WGERC_OMP_MIN 64
inline bool cdd_iszero(const T &x) {
    return x.re.limbs[0] == 0.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }

#ifdef MBLAS_SIMD_DD
static inline __attribute__((always_inline)) void
soa_load4_cdd(const double *p,
              __m256d &rh, __m256d &rl, __m256d &ih, __m256d &il)
{
    __m256d v0 = _mm256_loadu_pd(p +  0);
    __m256d v1 = _mm256_loadu_pd(p +  4);
    __m256d v2 = _mm256_loadu_pd(p +  8);
    __m256d v3 = _mm256_loadu_pd(p + 12);
    __m256d t0 = _mm256_unpacklo_pd(v0, v1);
    __m256d t1 = _mm256_unpackhi_pd(v0, v1);
    __m256d t2 = _mm256_unpacklo_pd(v2, v3);
    __m256d t3 = _mm256_unpackhi_pd(v2, v3);
    rh = _mm256_permute2f128_pd(t0, t2, 0x20);
    ih = _mm256_permute2f128_pd(t0, t2, 0x31);
    rl = _mm256_permute2f128_pd(t1, t3, 0x20);
    il = _mm256_permute2f128_pd(t1, t3, 0x31);
}
static inline __attribute__((always_inline)) void
soa_store4_cdd(double *p,
               __m256d rh, __m256d rl, __m256d ih, __m256d il)
{
    __m256d t0 = _mm256_permute2f128_pd(rh, ih, 0x20);
    __m256d t2 = _mm256_permute2f128_pd(rh, ih, 0x31);
    __m256d t1 = _mm256_permute2f128_pd(rl, il, 0x20);
    __m256d t3 = _mm256_permute2f128_pd(rl, il, 0x31);
    __m256d v0 = _mm256_unpacklo_pd(t0, t1);
    __m256d v1 = _mm256_unpackhi_pd(t0, t1);
    __m256d v2 = _mm256_unpacklo_pd(t2, t3);
    __m256d v3 = _mm256_unpackhi_pd(t2, t3);
    _mm256_storeu_pd(p +  0, v0);
    _mm256_storeu_pd(p +  4, v1);
    _mm256_storeu_pd(p +  8, v2);
    _mm256_storeu_pd(p + 12, v3);
}
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void wgerc_(
    const int *m_, const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    const T *y, const int *incy_,
    T *a, const int *lda_)
{
    const int M = *m_, N = *n_;
    const int incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;

    if (M == 0 || N == 0 || cdd_iszero(alpha)) return;

    if (incx == 1 && incy == 1) {
#ifdef MBLAS_SIMD_DD
        const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
        double *x_rh = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_rl = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_ih = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_il = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        for (int i = 0; i < M; ++i) {
            x_rh[i] = x[i].re.limbs[0]; x_rl[i] = x[i].re.limbs[1];
            x_ih[i] = x[i].im.limbs[0]; x_il[i] = x[i].im.limbs[1];
        }
        for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) {
            x_rh[i] = 0.0; x_rl[i] = 0.0; x_ih[i] = 0.0; x_il[i] = 0.0;
        }
#ifdef _OPENMP
        const int use_omp = (N >= WGERC_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const T yj = cconj(y[j]);
            if (cdd_iszero(yj)) continue;
            const T t = cmul(alpha, yj);
            const __m256d trh = _mm256_set1_pd(t.re.limbs[0]);
            const __m256d trl = _mm256_set1_pd(t.re.limbs[1]);
            const __m256d tih = _mm256_set1_pd(t.im.limbs[0]);
            const __m256d til = _mm256_set1_pd(t.im.limbs[1]);
            double *aj = reinterpret_cast<double *>(&A_(0, j));
            int i = 0;
            for (; i + 3 < M; i += 4) {
                __m256d a_rh, a_rl, a_ih, a_il;
                soa_load4_cdd(aj + 4 * i, a_rh, a_rl, a_ih, a_il);
                __m256d xrh = _mm256_loadu_pd(x_rh + i);
                __m256d xrl = _mm256_loadu_pd(x_rl + i);
                __m256d xih = _mm256_loadu_pd(x_ih + i);
                __m256d xil = _mm256_loadu_pd(x_il + i);
                __m256d p_rh, p_rl, p_ih, p_il;
                simd_dd::cdd_mul(trh, trl, tih, til, xrh, xrl, xih, xil,
                                 p_rh, p_rl, p_ih, p_il);
                __m256d nrh, nrl, nih, nil;
                simd_dd::cdd_add(a_rh, a_rl, a_ih, a_il, p_rh, p_rl, p_ih, p_il,
                                 nrh, nrl, nih, nil);
                soa_store4_cdd(aj + 4 * i, nrh, nrl, nih, nil);
            }
            T *ajs = &A_(0, j);
            for (; i < M; ++i) ajs[i] = cadd(ajs[i], cmul(t, x[i]));
        }
        std::free(x_rh); std::free(x_rl); std::free(x_ih); std::free(x_il);
#else
#ifdef _OPENMP
        const int use_omp = (N >= WGERC_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const T yj = cconj(y[j]);
            if (!cdd_iszero(yj)) {
                const T t = cmul(alpha, yj);
                T *aj = &A_(0, j);
                for (int i = 0; i < M; ++i) aj[i] = cadd(aj[i], cmul(t, x[i]));
            }
        }
#endif
    } else {
        int jy = (incy < 0) ? -(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            const T yj = cconj(y[jy]);
            if (!cdd_iszero(yj)) {
                const T t = cmul(alpha, yj);
                int ix = (incx < 0) ? -(M - 1) * incx : 0;
                T *aj = &A_(0, j);
                for (int i = 0; i < M; ++i) {
                    aj[i] = cadd(aj[i], cmul(t, x[ix]));
                    ix += incx;
                }
            }
            jy += incy;
        }
    }
}

#undef A_
