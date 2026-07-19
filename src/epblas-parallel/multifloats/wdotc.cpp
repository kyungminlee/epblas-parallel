/* wdotc — multifloats complex DD: Σ conj(X)·Y.
 * conj(X)·Y = (xr·yr + xi·yi) + j·(xr·yi - xi·yr). */
#include <cstddef>
#include <multifloats.h>
#include "mf_kernels.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#endif
#include "../common/epblas_facade.h"
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;

namespace {
using mf_kernels::cconj;
using mf_kernels::cmul;
using mf_kernels::cadd;
}

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"

namespace {
/* canonical EFTs — mf_simd_fast.h */
using simd_fast::twoprod;
using simd_fast::fast2sum;
using simd_fast::twosum;
using simd_exact::cload4;
using simd_fast::horizontal_dd;  /* Bailey 2-limb finalizer — mf_simd_fast.h */
using simd_fast::absorb;  /* Bailey 3-limb wide-acc — mf_simd_fast.h */
using simd_fast::renorm3;  /* Bailey 3-limb wide-acc — mf_simd_fast.h */
using simd_fast::dd_prod;  /* Bailey 3-limb wide-acc — mf_simd_fast.h */
}
#endif

#ifdef MBLAS_SIMD_DD
/* AVX2+FMA complex-dot reduction kernel — compiled under target("avx2,fma") so
 * it builds even at a pre-Haswell baseline -march; reached only behind the
 * mf_have_avx2_fma() runtime probe in wdotc_unit below. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static TC wdotc_unit_simd(std::ptrdiff_t n, const TC *x, const TC *y)
{
    multifloats::complex64x2 s{R{0.0, 0.0}, R{0.0, 0.0}};
    __m256d rA0 = _mm256_setzero_pd(), rA1 = _mm256_setzero_pd(), rA2 = _mm256_setzero_pd();
    __m256d iA0 = _mm256_setzero_pd(), iA1 = _mm256_setzero_pd(), iA2 = _mm256_setzero_pd();
    constexpr std::ptrdiff_t k = 64;
    std::ptrdiff_t counter = k;
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xrh, xrl, xih, xil, yrh, yrl, yih, yil;
        cload4(&x[i], xrh, xrl, xih, xil);
        cload4(&y[i], yrh, yrl, yih, yil);
        /* conj(x)·y = (xr·yr + xi·yi) + j·(xr·yi - xi·yr) */
        __m256d rh, rl, ph, pl;
        dd_prod(xrh, xrl, yrh, yrl, rh, rl);
        dd_prod(xih, xil, yih, yil, ph, pl);
        absorb(rh, rl, rA0, rA1, rA2);
        absorb(ph, pl, rA0, rA1, rA2);
        /* im: +xr·yi - xi·yr */
        dd_prod(xrh, xrl, yih, yil, rh, rl);
        dd_prod(xih, xil, yrh, yrl, ph, pl);
        absorb(rh, rl, iA0, iA1, iA2);
        __m256d nph = _mm256_sub_pd(_mm256_setzero_pd(), ph);
        __m256d npl = _mm256_sub_pd(_mm256_setzero_pd(), pl);
        absorb(nph, npl, iA0, iA1, iA2);
        if (--counter == 0) {
            renorm3(rA0, rA1, rA2);
            renorm3(iA0, iA1, iA2);
            counter = k;
        }
    }
    __m256d rt = _mm256_add_pd(rA1, rA2);
    __m256d it = _mm256_add_pd(iA1, iA2);
    s.re = horizontal_dd(rA0, rt);
    s.im = horizontal_dd(iA0, it);
    for (std::ptrdiff_t i = n4; i < n; ++i) s = cadd(s, cmul(cconj(x[i]), y[i]));
    return s;
}
#pragma GCC pop_options
#endif

/* Σ conj(X)·Y over contiguous unit-stride ranges — serial kernel. Runtime
 * dispatch: SIMD Bailey-wide on Haswell+, scalar (always compiled) otherwise.
 * Carved out so the OpenMP partial-reduction (and packed/banded triangular
 * matvecs, via mf_kernels.h) can call it per sub-range. */
multifloats::complex64x2
mf_kernels::wdotc_unit(std::ptrdiff_t n, const multifloats::complex64x2 *x,
                       const multifloats::complex64x2 *y)
{
    /* spelled out: mf_kernels::T is the real (float64x2) alias and would shadow
     * this file's complex T inside a member-of-mf_kernels definition. */
    multifloats::complex64x2 s{R{0.0, 0.0}, R{0.0, 0.0}};
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) return wdotc_unit_simd(n, x, y);
#endif
    for (std::ptrdiff_t i = 0; i < n; ++i) s = cadd(s, cmul(cconj(x[i]), y[i]));
    return s;
}

#ifdef _OPENMP
/* Threaded partial-reduction — shared wrapper (mf_omp::partial_reduce,
 * pre-initialized slots); per-slice serial kernel, tid-order cadd merge. */
#define WDOTC_OMP_MIN 8192
__attribute__((noinline)) static std::ptrdiff_t wdotc_omp(std::ptrdiff_t n, const TC *x, const TC *y, TC *out)
{
    if (n <= WDOTC_OMP_MIN || !blas_omp_should_thread())
        return 0;
    *out = mf_omp::partial_reduce(n, TC{R{0.0, 0.0}, R{0.0, 0.0}},
        [x, y](std::ptrdiff_t lo, std::ptrdiff_t hi) { return mf_kernels::wdotc_unit(hi - lo, x + lo, y + lo); },
        [](const TC &a, const TC &b) { return cadd(a, b); });
    return 1;
}
#endif

static TC wdotc_core(std::ptrdiff_t n,
                    const TC *x, std::ptrdiff_t incx,
                    const TC *y, std::ptrdiff_t incy)
{
    TC s{R{0.0, 0.0}, R{0.0, 0.0}};
    if (n <= 0) return s;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (wdotc_omp(n, x, y, &s)) return s;
#endif
        return mf_kernels::wdotc_unit(n, x, y);
    }

    std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
    std::ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
    for (std::ptrdiff_t i = 0; i < n; ++i) { s = cadd(s, cmul(cconj(x[ix]), y[iy])); ix += incx; iy += incy; }
    return s;
}

extern "C" {
EPBLAS_FACADE_DOT(wdotc, TC, TC)
}
