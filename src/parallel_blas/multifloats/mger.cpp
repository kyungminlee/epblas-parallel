/* mger — multifloats real DD rank-1 update.
 * SIMD path: per j, broadcast t=alpha*y[j]; inner i loop SoA-SIMD
 * with simd_dd::dd_mul + dd_add into A column j. Pre-pack x once. */

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
using T = mf::float64x2;

namespace {
#define MGER_OMP_MIN 64
const T zero_dd{0.0, 0.0};
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }

#ifdef MBLAS_SIMD_DD
static inline __attribute__((always_inline)) void
soa_load4(const double *p, __m256d &hi, __m256d &lo)
{
    __m256d a01 = _mm256_loadu_pd(p);
    __m256d a23 = _mm256_loadu_pd(p + 4);
    __m256d t0 = _mm256_unpacklo_pd(a01, a23);
    __m256d t1 = _mm256_unpackhi_pd(a01, a23);
    hi = _mm256_permute4x64_pd(t0, 0xD8);
    lo = _mm256_permute4x64_pd(t1, 0xD8);
}
static inline __attribute__((always_inline)) void
soa_store4(double *p, __m256d hi, __m256d lo)
{
    __m256d hp = _mm256_permute4x64_pd(hi, 0xD8);  /* [h0,h2,h1,h3] */
    __m256d lp = _mm256_permute4x64_pd(lo, 0xD8);  /* [l0,l2,l1,l3] */
    __m256d a01 = _mm256_unpacklo_pd(hp, lp);      /* [h0,l0,h1,l1] */
    __m256d a23 = _mm256_unpackhi_pd(hp, lp);      /* [h2,l2,h3,l3] */
    _mm256_storeu_pd(p,     a01);
    _mm256_storeu_pd(p + 4, a23);
}
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

extern "C" void mger_(
    const int *m_, const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    const T *y, const int *incy_,
    T *a, const int *lda_)
{
    const int M = *m_, N = *n_;
    const int incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;

    if (M == 0 || N == 0 || dd_iszero(alpha)) return;

    if (incx == 1 && incy == 1) {
#ifdef MBLAS_SIMD_DD
        const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
        double *x_hi = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_lo = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        for (int i = 0; i < M; ++i) { x_hi[i] = x[i].limbs[0]; x_lo[i] = x[i].limbs[1]; }
        for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) { x_hi[i] = 0.0; x_lo[i] = 0.0; }
#ifdef _OPENMP
        const int use_omp = (N >= MGER_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const T yj = y[j];
            if (dd_iszero(yj)) continue;
            const T t = alpha * yj;
            const __m256d thi = _mm256_set1_pd(t.limbs[0]);
            const __m256d tlo = _mm256_set1_pd(t.limbs[1]);
            double *aj = reinterpret_cast<double *>(&A_(0, j));
            int i = 0;
            for (; i + 3 < M; i += 4) {
                __m256d a_h, a_l;
                soa_load4(aj + 2 * i, a_h, a_l);
                __m256d xh = _mm256_loadu_pd(x_hi + i);
                __m256d xl = _mm256_loadu_pd(x_lo + i);
                __m256d p_h, p_l;
                simd_dd::dd_mul(thi, tlo, xh, xl, p_h, p_l);
                __m256d nh, nl;
                simd_dd::dd_add(a_h, a_l, p_h, p_l, nh, nl);
                soa_store4(aj + 2 * i, nh, nl);
            }
            T *ajs = &A_(0, j);
            for (; i < M; ++i) ajs[i] = ajs[i] + t * x[i];
        }
        std::free(x_hi); std::free(x_lo);
#else
#ifdef _OPENMP
        const int use_omp = (N >= MGER_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const T yj = y[j];
            if (!dd_iszero(yj)) {
                const T t = alpha * yj;
                T *aj = &A_(0, j);
                for (int i = 0; i < M; ++i) aj[i] = aj[i] + t * x[i];
            }
        }
#endif
    } else {
        int jy = (incy < 0) ? -(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            const T yj = y[jy];
            if (!dd_iszero(yj)) {
                const T t = alpha * yj;
                int ix = (incx < 0) ? -(M - 1) * incx : 0;
                T *aj = &A_(0, j);
                for (int i = 0; i < M; ++i) {
                    aj[i] = aj[i] + t * x[ix];
                    ix += incx;
                }
            }
            jy += incy;
        }
    }
}

#undef A_
