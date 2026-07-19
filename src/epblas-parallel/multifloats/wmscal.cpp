/*
 * wmscal — multifloats: X := α · X with α real DD, X complex DD.
 * Equivalent to BLAS CSSCAL / ZDSCAL pattern.
 */
#include <cstddef>
#include <multifloats.h>
#include "mf_pred.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#endif
#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::eq1;
namespace {

#ifdef MBLAS_SIMD_DD
using simd_exact::cload4;
using simd_exact::cstore4;
#endif
}  // namespace

#ifdef MBLAS_SIMD_DD
/* AVX2+FMA kernel body — compiled under target("avx2,fma") so it builds even
 * when the library's baseline -march is pre-Haswell; reached only behind the
 * mf_have_avx2_fma() runtime probe below. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static void wmscal_unit_simd(std::ptrdiff_t n, R alpha, TC *x)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xrh, xrl, xih, xil;
        cload4(&x[i], xrh, xrl, xih, xil);
        /* α (real) × (xre + j·xim) → α·xre + j·α·xim — scale each limb-pair */
        __m256d nrh, nrl, nih, nil_;
        simd_fast::mul(xrh, xrl, ah, al, nrh, nrl);
        simd_fast::mul(xih, xil, ah, al, nih, nil_);
        cstore4(&x[i], nrh, nrl, nih, nil_);
    }
    for (std::ptrdiff_t i = n4; i < n; ++i) { x[i].re = x[i].re * alpha; x[i].im = x[i].im * alpha; }
}
#pragma GCC pop_options
#endif

/* X := α·X (α real, X complex) over a unit-stride range — serial kernel.
 * Runtime dispatch: SIMD on Haswell+, scalar (always compiled) elsewhere. */
static void wmscal_unit(std::ptrdiff_t n, R alpha, TC *x)
{
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) { wmscal_unit_simd(n, alpha, x); return; }
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) { x[i].re = x[i].re * alpha; x[i].im = x[i].im * alpha; }
}

#ifdef _OPENMP
#define WMSCAL_OMP_MIN 2048
__attribute__((noinline)) static std::ptrdiff_t wmscal_omp(std::ptrdiff_t n, R alpha, TC *x)
{
    if (n <= WMSCAL_OMP_MIN || !blas_omp_should_thread())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) wmscal_unit(hi - lo, alpha, x + lo);
    }
    return 1;
}
#endif

static void wmscal_core(std::ptrdiff_t n, const R *alpha_, TC *x, std::ptrdiff_t incx)
{
    const R alpha = *alpha_;
    if (n <= 0 || eq1(alpha)) return;

    if (incx == 1) {
#ifdef _OPENMP
        if (wmscal_omp(n, alpha, x)) return;
#endif
        wmscal_unit(n, alpha, x);
    } else {
        std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) {
            x[ix].re = x[ix].re * alpha; x[ix].im = x[ix].im * alpha;
            ix += incx;
        }
    }
}

extern "C" {
EPBLAS_FACADE_SCAL(wmscal, R, TC)
}
