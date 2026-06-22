/*
 * wscal — multifloats complex DD: X := α · X (α complex).
 */
#include <cstddef>
#include <multifloats.h>
#include "mf_kernels.h"
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
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq1;
namespace {
using mf_kernels::cmul;

#ifdef MBLAS_SIMD_DD
using simd_exact::cload4;
using simd_exact::cstore4;
#endif
}  // namespace

/* X := α·X over a contiguous unit-stride range — serial kernel, unchanged. */
static void wscal_unit(std::ptrdiff_t n, T alpha, T *x)
{
#ifdef MBLAS_SIMD_DD
    const __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    const __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    const __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    const __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xrh, xrl, xih, xil;
        cload4(&x[i], xrh, xrl, xih, xil);
        __m256d nrh, nrl, nih, nil_;
        simd_fast::cmul(xrh, xrl, xih, xil, arh, arl, aih, ail,
                         nrh, nrl, nih, nil_);
        cstore4(&x[i], nrh, nrl, nih, nil_);
    }
    for (std::ptrdiff_t i = n4; i < n; ++i) x[i] = cmul(x[i], alpha);
#else
    for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = cmul(x[i], alpha);
#endif
}

#ifdef _OPENMP
#define WSCAL_OMP_MIN 2048
__attribute__((noinline)) static std::ptrdiff_t wscal_omp(std::ptrdiff_t n, T alpha, T *x)
{
    if (n <= WSCAL_OMP_MIN || !blas_omp_should_thread())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) wscal_unit(hi - lo, alpha, x + lo);
    }
    return 1;
}
#endif

static void wscal_core(std::ptrdiff_t n, const T *alpha_, T *x, std::ptrdiff_t incx)
{
    const T alpha = *alpha_;
    if (n <= 0 || ceq1(alpha)) return;

    if (incx == 1) {
#ifdef _OPENMP
        if (wscal_omp(n, alpha, x)) return;
#endif
        wscal_unit(n, alpha, x);
    } else {
        std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) { x[ix] = cmul(x[ix], alpha); ix += incx; }
    }
}

extern "C" {
EPBLAS_FACADE_SCAL(wscal, T, T)
}
