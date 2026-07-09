/* mger — multifloats real DD rank-1 update.
 * SIMD path: per j, broadcast t=alpha*y[j]; inner i loop SoA-SIMD
 * with simd_fast::mul + add into A column j. Pre-pack x once. */

#include <cstddef>
#include <cstdlib>
#include <multifloats.h>
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
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
namespace {
#define MGER_OMP_MIN 64

#ifdef MBLAS_SIMD_DD
using simd_exact::load_dd4;
using simd_exact::store_dd4;
#endif
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

#ifdef MBLAS_SIMD_DD
/* AVX2+FMA under a possibly pre-Haswell baseline -march: compiled with the
 * feature enabled and reached only behind mf_have_avx2_fma() at the call site
 * below. Plain static (NOT always_inline) so it is legally CALLED across the
 * target mismatch from the baseline core. See mf_dispatch.h. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")
namespace {
/* Rank-1 update of one A column: A(:,j) += t * x, x supplied SoA. The matrix is
 * column-major so A(:,j) is contiguous regardless of incx/incy — only x (gathered
 * to SoA once) and the y read differ between the contiguous and strided drivers,
 * so this one SIMD core serves both. */
static void
mger_col_simd(std::ptrdiff_t m, const double *x_hi, const double *x_lo, TR t, TR *ajT)
{
    const __m256d thi = _mm256_set1_pd(t.limbs[0]);
    const __m256d tlo = _mm256_set1_pd(t.limbs[1]);
    double *aj = reinterpret_cast<double *>(ajT);
    std::ptrdiff_t i = 0;
    /* 8-wide head: two INDEPENDENT 4-lane DD chains hide the long mul->add
     * EFT latency (no native SIMD for one scalar DD) that a single chain leaves
     * exposed while the column streams from cache. ~7% at N=1024, ~11% cache-
     * resident. Bit-identical — each 4-lane group matches the 4-wide loop. */
    for (; i + 8 <= m; i += 8) {
        __m256d a0h, a0l, a1h, a1l;
        load_dd4(aj + 2 * i,       a0h, a0l);
        load_dd4(aj + 2 * (i + 4), a1h, a1l);
        __m256d p0h, p0l, p1h, p1l;
        simd_fast::mul(thi, tlo, _mm256_loadu_pd(x_hi + i),     _mm256_loadu_pd(x_lo + i),     p0h, p0l);
        simd_fast::mul(thi, tlo, _mm256_loadu_pd(x_hi + i + 4), _mm256_loadu_pd(x_lo + i + 4), p1h, p1l);
        __m256d n0h, n0l, n1h, n1l;
        simd_fast::add(a0h, a0l, p0h, p0l, n0h, n0l);
        simd_fast::add(a1h, a1l, p1h, p1l, n1h, n1l);
        store_dd4(aj + 2 * i,       n0h, n0l);
        store_dd4(aj + 2 * (i + 4), n1h, n1l);
    }
    for (; i + 3 < m; i += 4) {
        __m256d a_h, a_l;
        load_dd4(aj + 2 * i, a_h, a_l);
        __m256d xh = _mm256_loadu_pd(x_hi + i);
        __m256d xl = _mm256_loadu_pd(x_lo + i);
        __m256d p_h, p_l;
        simd_fast::mul(thi, tlo, xh, xl, p_h, p_l);
        __m256d nh, nl;
        simd_fast::add(a_h, a_l, p_h, p_l, nh, nl);
        store_dd4(aj + 2 * i, nh, nl);
    }
    for (; i < m; ++i) ajT[i] = ajT[i] + t * TR{x_hi[i], x_lo[i]};
}
}
#pragma GCC pop_options
#endif

static void mger_core(
    std::ptrdiff_t m, std::ptrdiff_t n,
    const TR *alpha_,
    const TR *x, std::ptrdiff_t incx,
    const TR *y, std::ptrdiff_t incy,
    TR *a, std::ptrdiff_t lda)
{
    const TR alpha = *alpha_;

    if (m == 0 || n == 0 || eq0(alpha)) return;

    const std::ptrdiff_t jy0 = (incy < 0) ? -(n - 1) * incy : 0;
    const std::ptrdiff_t ix0 = (incx < 0) ? -(m - 1) * incx : 0;
#ifdef _OPENMP
    const bool use_omp = (n >= MGER_OMP_MIN && blas_omp_available());
#endif

#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) {
        /* Gather x to a unit-stride SoA pair once (O(M)); the matrix columns are
         * already contiguous, so every column update runs the SIMD core. Columns
         * of A are disjoint -> OMP-over-j is race-free. */
        const std::size_t M_pad = (static_cast<std::size_t>(m) + 3) & ~static_cast<std::size_t>(3);
        double *x_hi = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        double *x_lo = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
        { std::ptrdiff_t ix = ix0; for (std::ptrdiff_t i = 0; i < m; ++i) { x_hi[i] = x[ix].limbs[0]; x_lo[i] = x[ix].limbs[1]; ix += incx; } }
        for (std::size_t i = static_cast<std::size_t>(m); i < M_pad; ++i) { x_hi[i] = 0.0; x_lo[i] = 0.0; }
#ifdef _OPENMP
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            const TR yj = y[jy0 + j * incy];
            if (eq0(yj)) continue;
            mger_col_simd(m, x_hi, x_lo, alpha * yj, &A_(0, j));
        }
        std::free(x_hi); std::free(x_lo);
        return;
    }
#endif
    /* Scalar fallback (always compiled): gather strided x to unit stride once,
     * then a plain column AXPY per j (O(M) pack vs O(M*N) repeated gathers). */
    {
    const TR *xp = x;
    TR *x_buf = NULL;
    if (incx != 1) {
        x_buf = static_cast<TR *>(std::malloc(static_cast<size_t>(m) * sizeof(TR)));
        if (x_buf) { std::ptrdiff_t ix = ix0; for (std::ptrdiff_t i = 0; i < m; ++i) { x_buf[i] = x[ix]; ix += incx; } xp = x_buf; }
    }
    const std::ptrdiff_t x_unit = (incx == 1) || (x_buf != NULL);
#ifdef _OPENMP
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        const TR yj = y[jy0 + j * incy];
        if (!eq0(yj)) {
            const TR t = alpha * yj;
            TR *aj = &A_(0, j);
            if (x_unit) { for (std::ptrdiff_t i = 0; i < m; ++i) aj[i] = aj[i] + t * xp[i]; }
            else { std::ptrdiff_t ix = ix0; for (std::ptrdiff_t i = 0; i < m; ++i) { aj[i] = aj[i] + t * x[ix]; ix += incx; } }
        }
    }
    std::free(x_buf);
    }
}

extern "C" {
EPBLAS_FACADE_GER(mger, TR)
}

#undef A_
