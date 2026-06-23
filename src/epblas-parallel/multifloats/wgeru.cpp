/* wgeru — multifloats complex DD unconjugated rank-1.
 * SIMD: SoA-pack x once; per j, broadcast t = alpha*y[j]; SoA-SIMD
 * cmul + cadd into A column. */

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
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
namespace {
#define WGERU_OMP_MIN 64
using mf_kernels::cmul;
using mf_kernels::cadd;

#ifdef MBLAS_SIMD_DD
using simd_exact::cload4;
using simd_exact::cstore4;
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Contiguous (unit-stride) core: A += alpha * x * y^T, x length M, y length N.
 * SIMD column-AXPY (cmul/cadd) when MBLAS_SIMD_DD; columns of A disjoint
 * so OMP-over-j is race-free and bit-exact. Strided callers gather x/y around it. */
static void wgeru_contig(std::ptrdiff_t m, std::ptrdiff_t n, TC alpha, TC *a, std::size_t lda,
                         const TC *x, const TC *y)
{
#ifdef MBLAS_SIMD_DD
        const std::size_t M_pad = (static_cast<std::size_t>(m) + 3) & ~static_cast<std::size_t>(3);
        double *x_rh = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_rl = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_ih = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_il = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            x_rh[i] = x[i].re.limbs[0]; x_rl[i] = x[i].re.limbs[1];
            x_ih[i] = x[i].im.limbs[0]; x_il[i] = x[i].im.limbs[1];
        }
        for (std::size_t i = static_cast<std::size_t>(m); i < M_pad; ++i) {
            x_rh[i] = 0.0; x_rl[i] = 0.0; x_ih[i] = 0.0; x_il[i] = 0.0;
        }
#ifdef _OPENMP
        const bool use_omp = (n >= WGERU_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            const TC yj = y[j];
            if (ceq0(yj)) continue;
            const TC t = cmul(alpha, yj);
            const __m256d trh = _mm256_set1_pd(t.re.limbs[0]);
            const __m256d trl = _mm256_set1_pd(t.re.limbs[1]);
            const __m256d tih = _mm256_set1_pd(t.im.limbs[0]);
            const __m256d til = _mm256_set1_pd(t.im.limbs[1]);
            double *aj = reinterpret_cast<double *>(&A_(0, j));
            std::ptrdiff_t i = 0;
            for (; i + 3 < m; i += 4) {
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
            TC *ajs = &A_(0, j);
            for (; i < m; ++i) ajs[i] = cadd(ajs[i], cmul(t, x[i]));
        }
        std::free(x_rh); std::free(x_rl); std::free(x_ih); std::free(x_il);
#else
#ifdef _OPENMP
        const bool use_omp = (n >= WGERU_OMP_MIN && blas_omp_available());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            const TC yj = y[j];
            if (!ceq0(yj)) {
                const TC t = cmul(alpha, yj);
                TC *aj = &A_(0, j);
                for (std::ptrdiff_t i = 0; i < m; ++i) aj[i] = cadd(aj[i], cmul(t, x[i]));
            }
        }
#endif
}

static void wgeru_core(
    std::ptrdiff_t m, std::ptrdiff_t n,
    const TC *alpha_,
    const TC *x, std::ptrdiff_t incx,
    const TC *y, std::ptrdiff_t incy,
    TC *a, std::ptrdiff_t lda)
{
    const TC alpha = *alpha_;

    if (m == 0 || n == 0 || ceq0(alpha)) return;

    if (incx == 1 && incy == 1) {
        wgeru_contig(m, n, alpha, a, lda, x, y);
        return;
    }
    /* Strided: gather x (len M) and y (len N) to unit-stride scratch, run the
     * SIMD core (A is column-major/lda regardless of vector strides). */
    const TC *xbase = (incx < 0) ? x - static_cast<std::ptrdiff_t>(m - 1) * incx : x;
    const TC *ybase = (incy < 0) ? y - static_cast<std::ptrdiff_t>(n - 1) * incy : y;
    std::vector<TC> xs(static_cast<std::size_t>(m)), ys(static_cast<std::size_t>(n));
    for (std::ptrdiff_t i = 0; i < m; ++i) xs[i] = xbase[static_cast<std::ptrdiff_t>(i) * incx];
    for (std::ptrdiff_t j = 0; j < n; ++j) ys[j] = ybase[static_cast<std::ptrdiff_t>(j) * incy];
    wgeru_contig(m, n, alpha, a, lda, xs.data(), ys.data());
}

extern "C" {
EPBLAS_FACADE_GER(wgeru, TC)
}

#undef A_
