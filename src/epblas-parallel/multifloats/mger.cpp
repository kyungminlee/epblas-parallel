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

namespace mf = multifloats;
using T = mf::float64x2;


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
namespace {
/* Rank-1 update of one A column: A(:,j) += t * x, x supplied SoA. The matrix is
 * column-major so A(:,j) is contiguous regardless of incx/incy — only x (gathered
 * to SoA once) and the y read differ between the contiguous and strided drivers,
 * so this one SIMD core serves both. */
static inline __attribute__((always_inline)) void
mger_col(int M, const double *x_hi, const double *x_lo, T t, T *ajT)
{
    const __m256d thi = _mm256_set1_pd(t.limbs[0]);
    const __m256d tlo = _mm256_set1_pd(t.limbs[1]);
    double *aj = reinterpret_cast<double *>(ajT);
    int i = 0;
    /* 8-wide head: two INDEPENDENT 4-lane DD chains hide the long mul->add
     * EFT latency (no native SIMD for one scalar DD) that a single chain leaves
     * exposed while the column streams from cache. ~7% at N=1024, ~11% cache-
     * resident. Bit-identical — each 4-lane group matches the 4-wide loop. */
    for (; i + 8 <= M; i += 8) {
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
    for (; i + 3 < M; i += 4) {
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
    for (; i < M; ++i) ajT[i] = ajT[i] + t * T{x_hi[i], x_lo[i]};
}
}
#endif

extern "C" void mger_(
    const int *m_, const int *n_,
    const T *alpha_,
    const T *x, const int *incx_,
    const T *y, const int *incy_,
    T *a, const int *lda_)
{
    const int M = *m_, N = *n_;
    const int incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;

    if (M == 0 || N == 0 || eq0(alpha)) return;

    const int jy0 = (incy < 0) ? -(N - 1) * incy : 0;
    const int ix0 = (incx < 0) ? -(M - 1) * incx : 0;
#ifdef _OPENMP
    const int use_omp = (N >= MGER_OMP_MIN && blas_omp_max_threads() > 1);
#endif

#ifdef MBLAS_SIMD_DD
    /* Gather x to a unit-stride SoA pair once (O(M)); the matrix columns are
     * already contiguous, so every column update runs the SIMD core. Columns of A
     * are disjoint -> OMP-over-j is race-free. */
    const std::size_t M_pad = (static_cast<std::size_t>(M) + 3) & ~static_cast<std::size_t>(3);
    double *x_hi = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    double *x_lo = static_cast<double *>(std::aligned_alloc(32, M_pad * sizeof(double)));
    { int ix = ix0; for (int i = 0; i < M; ++i) { x_hi[i] = x[ix].limbs[0]; x_lo[i] = x[ix].limbs[1]; ix += incx; } }
    for (std::size_t i = static_cast<std::size_t>(M); i < M_pad; ++i) { x_hi[i] = 0.0; x_lo[i] = 0.0; }
#ifdef _OPENMP
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
        const T yj = y[jy0 + j * incy];
        if (eq0(yj)) continue;
        mger_col(M, x_hi, x_lo, alpha * yj, &A_(0, j));
    }
    std::free(x_hi); std::free(x_lo);
#else
    /* Scalar fallback: gather strided x to unit stride once, then a plain column
     * AXPY per j (O(M) pack vs O(M*N) repeated gathers). */
    const T *xp = x;
    T *x_buf = NULL;
    if (incx != 1) {
        x_buf = static_cast<T *>(std::malloc(static_cast<size_t>(M) * sizeof(T)));
        if (x_buf) { int ix = ix0; for (int i = 0; i < M; ++i) { x_buf[i] = x[ix]; ix += incx; } xp = x_buf; }
    }
    const int x_unit = (incx == 1) || (x_buf != NULL);
#ifdef _OPENMP
    #pragma omp parallel for if(use_omp) schedule(static)
#endif
    for (int j = 0; j < N; ++j) {
        const T yj = y[jy0 + j * incy];
        if (!eq0(yj)) {
            const T t = alpha * yj;
            T *aj = &A_(0, j);
            if (x_unit) { for (int i = 0; i < M; ++i) aj[i] = aj[i] + t * xp[i]; }
            else { int ix = ix0; for (int i = 0; i < M; ++i) { aj[i] = aj[i] + t * x[ix]; ix += incx; } }
        }
    }
    std::free(x_buf);
#endif
}

#undef A_
