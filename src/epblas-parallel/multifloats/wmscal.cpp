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

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq1;
namespace {

#ifdef MBLAS_SIMD_DD
using simd_exact::cload4;
using simd_exact::cstore4;
#endif
}  // namespace

/* X := α·X (α real, X complex) over a unit-stride range — serial, unchanged. */
static void wmscal_unit(int n, R alpha, T *x)
{
#ifdef MBLAS_SIMD_DD
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d xrh, xrl, xih, xil;
        cload4(&x[i], xrh, xrl, xih, xil);
        /* α (real) × (xre + j·xim) → α·xre + j·α·xim — scale each limb-pair */
        __m256d nrh, nrl, nih, nil_;
        simd_fast::mul(xrh, xrl, ah, al, nrh, nrl);
        simd_fast::mul(xih, xil, ah, al, nih, nil_);
        cstore4(&x[i], nrh, nrl, nih, nil_);
    }
    for (int i = n4; i < n; ++i) { x[i].re = x[i].re * alpha; x[i].im = x[i].im * alpha; }
#else
    for (int i = 0; i < n; ++i) { x[i].re = x[i].re * alpha; x[i].im = x[i].im * alpha; }
#endif
}

#ifdef _OPENMP
#define WMSCAL_OMP_MIN 2048
__attribute__((noinline)) static int wmscal_omp(int n, R alpha, T *x)
{
    if (n <= WMSCAL_OMP_MIN || !blas_omp_available() || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) wmscal_unit(hi - lo, alpha, x + lo);
    }
    return 1;
}
#endif

extern "C" void wmscal_(const int *n_, const R *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const R alpha = *alpha_;
    if (n <= 0 || eq1(alpha)) return;

    if (incx == 1) {
#ifdef _OPENMP
        if (wmscal_omp(n, alpha, x)) return;
#endif
        wmscal_unit(n, alpha, x);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) {
            x[ix].re = x[ix].re * alpha; x[ix].im = x[ix].im * alpha;
            ix += incx;
        }
    }
}
