/* wdotu — multifloats complex DD: Σ X·Y (unconjugated).
 * Two SIMD Bailey-wide accumulators (re, im), each 3-double-per-lane. */
#include <cstddef>
#include <multifloats.h>
#include "mf_kernels.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
using mf_kernels::cmul;
using mf_kernels::cadd;
}

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"

namespace {
/* canonical EFTs — mf_simd_fast.h (2a-5) */
using simd_fast::twoprod;
using simd_fast::fast2sum;
using simd_fast::twosum;
using simd_exact::cload4;
using simd_fast::horizontal_dd;  /* Bailey 2-limb finalizer — mf_simd_fast.h (#4) */
using simd_fast::absorb;  /* Bailey 3-limb wide-acc — mf_simd_fast.h (#4) */
using simd_fast::renorm3;  /* Bailey 3-limb wide-acc — mf_simd_fast.h (#4) */
using simd_fast::dd_prod;  /* Bailey 3-limb wide-acc — mf_simd_fast.h (#4) */
}
#endif

/* Σ X·Y over contiguous unit-stride ranges — serial kernel, unchanged.
 * Carved out so the OpenMP partial-reduction (and packed/banded triangular
 * matvecs, via mf_kernels.h) can call it per sub-range. */
multifloats::complex64x2
mf_kernels::wdotu_unit(std::ptrdiff_t n, const multifloats::complex64x2 *x,
                       const multifloats::complex64x2 *y)
{
    /* spelled out: mf_kernels::T is the real (float64x2) alias and would shadow
     * this file's complex T inside a member-of-mf_kernels definition. */
    multifloats::complex64x2 s{R{0.0, 0.0}, R{0.0, 0.0}};
#ifdef MBLAS_SIMD_DD
    __m256d rA0 = _mm256_setzero_pd(), rA1 = _mm256_setzero_pd(), rA2 = _mm256_setzero_pd();
    __m256d iA0 = _mm256_setzero_pd(), iA1 = _mm256_setzero_pd(), iA2 = _mm256_setzero_pd();
    constexpr std::ptrdiff_t K = 64;
    std::ptrdiff_t counter = K;
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xrh, xrl, xih, xil, yrh, yrl, yih, yil;
        cload4(&x[i], xrh, xrl, xih, xil);
        cload4(&y[i], yrh, yrl, yih, yil);
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
    for (std::ptrdiff_t i = n4; i < n; ++i) s = cadd(s, cmul(x[i], y[i]));
    return s;
#else
    for (std::ptrdiff_t i = 0; i < n; ++i) s = cadd(s, cmul(x[i], y[i]));
    return s;
#endif
}

#ifdef _OPENMP
#define WDOTU_OMP_MIN 8192
#define WDOTU_MAX_CPUS 64
__attribute__((noinline)) static std::ptrdiff_t wdotu_omp(std::ptrdiff_t n, const T *x, const T *y, T *out)
{
    if (n <= WDOTU_OMP_MIN || !blas_omp_should_thread())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > WDOTU_MAX_CPUS) nthreads = WDOTU_MAX_CPUS;
    T partial[WDOTU_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        partial[tid] = (lo < hi) ? mf_kernels::wdotu_unit(hi - lo, x + lo, y + lo)
                                 : T{R{0.0, 0.0}, R{0.0, 0.0}};
    }
    T s{R{0.0, 0.0}, R{0.0, 0.0}};
    for (std::ptrdiff_t i = 0; i < nthreads; ++i) s = cadd(s, partial[i]);
    *out = s;
    return 1;
}
#endif

static T wdotu_core(std::ptrdiff_t n,
                    const T *x, std::ptrdiff_t incx,
                    const T *y, std::ptrdiff_t incy)
{
    T s{R{0.0, 0.0}, R{0.0, 0.0}};
    if (n <= 0) return s;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (wdotu_omp(n, x, y, &s)) return s;
#endif
        return mf_kernels::wdotu_unit(n, x, y);
    }

    std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
    std::ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
    for (std::ptrdiff_t i = 0; i < n; ++i) { s = cadd(s, cmul(x[ix], y[iy])); ix += incx; iy += incy; }
    return s;
}

extern "C" {
EPBLAS_FACADE_DOT(wdotu, T, T)
}
