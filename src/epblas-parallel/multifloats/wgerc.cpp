/* wgerc — multifloats complex DD conjugated rank-1.
 * Same SIMD structure as wgeru with t = alpha * conj(y[j]). */

#include <cstddef>
#include <cstdlib>
#include <vector>
#include <multifloats.h>
#include "mf_kernels.h"
#include "mf_pred.h"
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
namespace {
#define WGERC_OMP_MIN 64
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;

#ifdef MBLAS_SIMD_DD
using simd_exact::cload4;
using simd_exact::cstore4;
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Contiguous (unit-stride) core: A += alpha * x * conj(y)^T, x length M, y length
 * N. SIMD column-AXPY (cmul/cadd) when MBLAS_SIMD_DD; columns of A disjoint
 * so OMP-over-j is race-free and bit-exact. Strided callers gather x/y around it. */
static void wgerc_contig(std::ptrdiff_t M, std::ptrdiff_t N, T alpha, T *a, std::size_t lda,
                         const T *x, const T *y)
{
#ifdef MBLAS_SIMD_DD
        const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
        double *x_rh = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_rl = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_ih = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_il = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        for (std::ptrdiff_t i = 0; i < M; ++i) {
            x_rh[i] = x[i].re.limbs[0]; x_rl[i] = x[i].re.limbs[1];
            x_ih[i] = x[i].im.limbs[0]; x_il[i] = x[i].im.limbs[1];
        }
        for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) {
            x_rh[i] = 0.0; x_rl[i] = 0.0; x_ih[i] = 0.0; x_il[i] = 0.0;
        }
#ifdef _OPENMP
        const bool use_omp = (N >= WGERC_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            const T yj = cconj(y[j]);
            if (ceq0(yj)) continue;
            const T t = cmul(alpha, yj);
            const __m256d trh = _mm256_set1_pd(t.re.limbs[0]);
            const __m256d trl = _mm256_set1_pd(t.re.limbs[1]);
            const __m256d tih = _mm256_set1_pd(t.im.limbs[0]);
            const __m256d til = _mm256_set1_pd(t.im.limbs[1]);
            double *aj = reinterpret_cast<double *>(&A_(0, j));
            std::ptrdiff_t i = 0;
            for (; i + 3 < M; i += 4) {
                __m256d a_rh, a_rl, a_ih, a_il;
                cload4(aj + 4 * i, a_rh, a_rl, a_ih, a_il);
                __m256d xrh = _mm256_loadu_pd(x_rh + i);
                __m256d xrl = _mm256_loadu_pd(x_rl + i);
                __m256d xih = _mm256_loadu_pd(x_ih + i);
                __m256d xil = _mm256_loadu_pd(x_il + i);
                __m256d p_rh, p_rl, p_ih, p_il;
                simd_fast::cmul(trh, trl, tih, til, xrh, xrl, xih, xil,
                                 p_rh, p_rl, p_ih, p_il);
                __m256d nrh, nrl, nih, nil;
                simd_fast::cadd(a_rh, a_rl, a_ih, a_il, p_rh, p_rl, p_ih, p_il,
                                 nrh, nrl, nih, nil);
                cstore4(aj + 4 * i, nrh, nrl, nih, nil);
            }
            T *ajs = &A_(0, j);
            for (; i < M; ++i) ajs[i] = cadd(ajs[i], cmul(t, x[i]));
        }
        std::free(x_rh); std::free(x_rl); std::free(x_ih); std::free(x_il);
#else
#ifdef _OPENMP
        const bool use_omp = (N >= WGERC_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < N; ++j) {
            const T yj = cconj(y[j]);
            if (!ceq0(yj)) {
                const T t = cmul(alpha, yj);
                T *aj = &A_(0, j);
                for (std::ptrdiff_t i = 0; i < M; ++i) aj[i] = cadd(aj[i], cmul(t, x[i]));
            }
        }
#endif
}

static void wgerc_core(
    std::ptrdiff_t M, std::ptrdiff_t N,
    const T *alpha_,
    const T *x, std::ptrdiff_t incx,
    const T *y, std::ptrdiff_t incy,
    T *a, std::ptrdiff_t lda)
{
    const T alpha = *alpha_;

    if (M == 0 || N == 0 || ceq0(alpha)) return;

    if (incx == 1 && incy == 1) {
        wgerc_contig(M, N, alpha, a, lda, x, y);
        return;
    }
    /* Strided: gather x (len M) and y (len N) to unit-stride scratch, run the
     * SIMD core (A is column-major/lda regardless of vector strides). */
    const T *xbase = (incx < 0) ? x - static_cast<std::ptrdiff_t>(M - 1) * incx : x;
    const T *ybase = (incy < 0) ? y - static_cast<std::ptrdiff_t>(N - 1) * incy : y;
    std::vector<T> xs(static_cast<std::size_t>(M)), ys(static_cast<std::size_t>(N));
    for (std::ptrdiff_t i = 0; i < M; ++i) xs[i] = xbase[static_cast<std::ptrdiff_t>(i) * incx];
    for (std::ptrdiff_t j = 0; j < N; ++j) ys[j] = ybase[static_cast<std::ptrdiff_t>(j) * incy];
    wgerc_contig(M, N, alpha, a, lda, xs.data(), ys.data());
}

extern "C" {
EPBLAS_FACADE_GER(wgerc, T)
}

#undef A_
