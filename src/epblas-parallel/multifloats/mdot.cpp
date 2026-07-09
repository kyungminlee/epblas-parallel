/* mdot — multifloats real DD: Σ X·Y.
 *
 * SIMD Bailey-wide accumulator (V5). 4-wide SIMD with a 3-double-per-lane
 * wide accumulator (a0, a1, a2). Periodic renorm every K=64 iters keeps the
 * accumulator stable. Final horizontal reduce.
 *
 * Bench (vs scalar full-DD):
 *   N=64..1024 : 1.0–1.4× (small-N dispatch dominates)
 *   N=8K..1M   : ~5× faster, full DD precision (~30–32 digits)
 *
 * See V3 vs V5 study: Bailey-wide consistently beats SIMD full-DD on both
 * speed and precision; Kahan-defer was rejected because it collapses to
 * double precision.
 */
#include <cstddef>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#endif
#include "../common/epblas_facade.h"
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */

namespace mf = multifloats;
using TR = mf::float64x2;

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"

namespace {
/* canonical EFTs — mf_simd_fast.h (2a-5) */
using simd_fast::twoprod;
using simd_fast::fast2sum;
using simd_fast::twosum;
using simd_exact::load_dd4;
using simd_fast::horizontal_dd;  /* Bailey 2-limb finalizer — mf_simd_fast.h (#4) */
}
#endif

#ifdef MBLAS_SIMD_DD
/* AVX2+FMA reduction kernel — compiled under target("avx2,fma") so it builds
 * even at a pre-Haswell baseline -march; reached only behind mf_have_avx2_fma(). */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
static TR mdot_unit_simd(std::ptrdiff_t n, const TR *x, const TR *y)
{
    __m256d a0 = _mm256_setzero_pd();
    __m256d a1 = _mm256_setzero_pd();
    __m256d a2 = _mm256_setzero_pd();
    constexpr std::ptrdiff_t k = 64;
    std::ptrdiff_t counter = k;
    const std::ptrdiff_t n4 = n & ~3;
    for (std::ptrdiff_t i = 0; i < n4; i += 4) {
        __m256d xh, xl, yh, yl;
        load_dd4(&x[i], xh, xl);
        load_dd4(&y[i], yh, yl);
        /* DD product via simplified twoprod: drop xl*yl (~2^-106 of xh*yh) */
        __m256d ph, pl;
        twoprod(xh, yh, ph, pl);
        pl = _mm256_add_pd(pl,
                _mm256_add_pd(_mm256_mul_pd(xh, yl), _mm256_mul_pd(xl, yh)));
        /* Wide-acc absorb (a0, a1, a2) += (ph, pl) */
        __m256d e0, e1, e2;
        twosum(a0, ph, a0, e0);
        twosum(a1, pl, a1, e1);
        twosum(a1, e0, a1, e2);
        a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
        if (--counter == 0) {
            /* Renormalize 3→2 doubles */
            __m256d t, e;
            fast2sum(a1, a2, t, e);
            a1 = t; a2 = e;
            fast2sum(a0, a1, a0, a1);
            a1 = _mm256_add_pd(a1, a2);
            fast2sum(a0, a1, a0, a1);
            a2 = _mm256_setzero_pd();
            counter = k;
        }
    }
    __m256d t = _mm256_add_pd(a1, a2);
    TR s = horizontal_dd(a0, t);
    for (std::ptrdiff_t i = n4; i < n; ++i) s = s + x[i] * y[i];
    return s;
}
#pragma GCC pop_options
#endif

/* Σ X·Y over contiguous unit-stride ranges — the serial kernel. Runtime
 * dispatch: SIMD Bailey-wide on Haswell+, scalar (always compiled) otherwise.
 * Carved out so the OpenMP partial-reduction can call it per sub-range. */
static TR mdot_unit(std::ptrdiff_t n, const TR *x, const TR *y)
{
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) return mdot_unit_simd(n, x, y);
#endif
    TR s0{0.0, 0.0}, s1{0.0, 0.0}, s{0.0, 0.0};
    std::ptrdiff_t i = 0;
    for (; i + 1 < n; i += 2) {
        s0 = s0 + x[i]     * y[i];
        s1 = s1 + x[i + 1] * y[i + 1];
    }
    s = s0 + s1;
    for (; i < n; ++i) s = s + x[i] * y[i];
    return s;
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride X·Y. Each thread dots its
 * contiguous slice with the serial kernel; partials combine in tid order.
 * Reduction order differs from serial → within fuzz tolerance (kind10 edot). */
#define MDOT_OMP_MIN 8192
#define MDOT_MAX_CPUS 64
__attribute__((noinline)) static std::ptrdiff_t mdot_omp(std::ptrdiff_t n, const TR *x, const TR *y, TR *out)
{
    if (n <= MDOT_OMP_MIN || !blas_omp_should_thread())
        return 0;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > MDOT_MAX_CPUS) nthreads = MDOT_MAX_CPUS;
    TR partial[MDOT_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t nth = omp_get_num_threads();
        std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        partial[tid] = (lo < hi) ? mdot_unit(hi - lo, x + lo, y + lo) : TR{0.0, 0.0};
    }
    TR s{0.0, 0.0};
    for (std::ptrdiff_t i = 0; i < nthreads; ++i) s = s + partial[i];
    *out = s;
    return 1;
}
#endif

static TR mdot_core(std::ptrdiff_t n,
                   const TR *x, std::ptrdiff_t incx,
                   const TR *y, std::ptrdiff_t incy)
{
    TR s{0.0, 0.0};
    if (n <= 0) return s;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (mdot_omp(n, x, y, &s)) return s;
#endif
        return mdot_unit(n, x, y);
    }

    /* Strided fallback */
    std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
    std::ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
    for (std::ptrdiff_t i = 0; i < n; ++i) { s = s + x[ix] * y[iy]; ix += incx; iy += incy; }
    return s;
}

extern "C" { EPBLAS_FACADE_DOT(mdot, TR, TR) }
