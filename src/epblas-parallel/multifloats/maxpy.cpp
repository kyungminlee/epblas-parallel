/*
 * maxpy — multifloats real DD: Y := α · X + Y.
 *
 * 4-wide AVX2 SIMD path for INCX=INCY=1, scalar fallback otherwise.
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
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
namespace {

#ifdef MBLAS_SIMD_DD
using simd_exact::load_dd4;
using simd_exact::store_dd4;
#endif
}  // namespace

/* Y := α·X + Y over a contiguous unit-stride range — serial kernel, unchanged.
 * Carved out so the OpenMP path can run it per disjoint sub-range. */
static void maxpy_unit(std::ptrdiff_t n, T alpha, const T *x, T *y)
{
#ifdef MBLAS_SIMD_DD
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xh, xl, yh, yl;
        load_dd4(&x[i], xh, xl);
        load_dd4(&y[i], yh, yl);
        __m256d ph, pl;
        simd_fast::mul(ah, al, xh, xl, ph, pl);
        __m256d nh, nl;
        simd_fast::add(yh, yl, ph, pl, nh, nl);
        store_dd4(&y[i], nh, nl);
    }
    for (std::ptrdiff_t i = n4; i < n; ++i) y[i] = y[i] + alpha * x[i];
#else
    for (std::ptrdiff_t i = 0; i < n; ++i) y[i] = y[i] + alpha * x[i];
#endif
}

#ifdef _OPENMP
/* Threaded elementwise AXPY. Outputs are disjoint per slice, so each thread
 * runs the SIMD kernel over its own [lo,hi) range — no reduction. DD math is
 * compute-bound, so this threads profitably above the crossover. */
#define MAXPY_OMP_MIN 2048
__attribute__((noinline)) static std::ptrdiff_t maxpy_omp(std::ptrdiff_t n, T alpha, const T *x, T *y)
{
    if (n <= MAXPY_OMP_MIN || !blas_omp_should_thread())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        if (lo < hi) maxpy_unit(hi - lo, alpha, x + lo, y + lo);
    }
    return 1;
}
#endif

static void maxpy_core(std::ptrdiff_t n, const T *alpha_,
                       const T *x, std::ptrdiff_t incx,
                       T *y, std::ptrdiff_t incy)
{
    const T alpha = *alpha_;
    if (n <= 0 || eq0(alpha)) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (maxpy_omp(n, alpha, x, y)) return;
#endif
        maxpy_unit(n, alpha, x, y);
    } else {
        std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) { y[iy] = y[iy] + alpha * x[ix]; ix += incx; iy += incy; }
    }
}

extern "C" { EPBLAS_FACADE_AXPY(maxpy, T, T) }
