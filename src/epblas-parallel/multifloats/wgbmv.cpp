/* wgbmv — multifloats complex DD general band matrix-vector multiply.
 *   y := alpha*op(A)*x + beta*y, A an M-by-N band with KL sub-/KU super-diagonals.
 */

#include <cstddef>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define WGBMV_OMP_MIN 64
#define WGBMV_MAX_CPUS 256
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
using mf_kernels::cmul;
using mf_kernels::cadd;
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* A band column stores its in-band rows [i_lo,i_hi) contiguously at &A_(KU-j+i_lo,j),
 * so per column NoTrans is a SIMD AXPY into y and Trans is a SIMD (conj-)dot of the
 * band run against x — the same stride-1 kernels used by the triangular twins. */
static inline void wgbmv_band(std::ptrdiff_t j, std::ptrdiff_t m, std::ptrdiff_t KL, std::ptrdiff_t KU, std::ptrdiff_t &i_lo, std::ptrdiff_t &i_hi) {
    i_lo = (j - KU > 0) ? (j - KU) : 0;
    i_hi = (j + KL + 1 < m) ? (j + KL + 1) : m;
}

#ifdef _OPENMP
/* Threaded NoTrans (contiguous x len N, y len M pre-beta'd). Each thread owns a
 * disjoint output-row range [r0,r1) and folds only the columns whose band touches
 * it, clipping each contiguous band run to [r0,r1). Disjoint y rows -> no race,
 * no reduction; each row still accumulates ascending-j -> BIT-EXACT vs serial. */
static bool wgbmv_n_omp(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t KL, std::ptrdiff_t KU, TC alpha,
                        const TC *a, std::size_t lda, const TC *x, TC *y)
{
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (!blas_omp_should_thread()) return false;
    if (nthreads > WGBMV_MAX_CPUS) nthreads = WGBMV_MAX_CPUS;

    #pragma omp parallel num_threads(nthreads)
    {
        const std::ptrdiff_t tid = omp_get_thread_num();
        const std::ptrdiff_t r0 = blas_part_bound(m, tid, omp_get_num_threads());
        const std::ptrdiff_t r1 = blas_part_bound(m, tid + 1, omp_get_num_threads());
        const std::ptrdiff_t j0 = (r0 - KL > 0) ? (r0 - KL) : 0;
        const std::ptrdiff_t j1 = (r1 + KU < n) ? (r1 + KU) : n;
        for (std::ptrdiff_t j = j0; j < j1; ++j) {
            const TC tmp = cmul(alpha, x[j]);
            if (ceq0(tmp)) continue;
            std::ptrdiff_t i_lo, i_hi; wgbmv_band(j, m, KL, KU, i_lo, i_hi);
            if (i_lo < r0) i_lo = r0;
            if (i_hi > r1) i_hi = r1;
            if (i_lo < i_hi)
                mf_kernels::caxpy_add(i_hi - i_lo, &y[i_lo], &A_(KU - j + i_lo, j), tmp);
        }
    }
    return true;
}
#endif

/* Contiguous NoTrans core (x len N, y len M, y pre-beta'd). Threaded row-disjoint
 * sweep when enabled, else a serial column AXPY (ascending-j -> bit-exact). */
static void wgbmv_n_contig(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t KL, std::ptrdiff_t KU, TC alpha,
                           const TC *a, std::size_t lda, const TC *x, TC *y)
{
#ifdef _OPENMP
    if (m >= WGBMV_OMP_MIN && blas_omp_available()
        && wgbmv_n_omp(m, n, KL, KU, alpha, a, lda, x, y))
        return;
#endif
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        const TC tmp = cmul(alpha, x[j]);
        if (ceq0(tmp)) continue;
        std::ptrdiff_t i_lo, i_hi; wgbmv_band(j, m, KL, KU, i_lo, i_hi);
        mf_kernels::caxpy_add(i_hi - i_lo, &y[i_lo], &A_(KU - j + i_lo, j), tmp);
    }
}

/* Contiguous Trans/ConjTrans core (x len M, y len N, y pre-beta'd). Each y[j] is
 * an independent band conj-dot (disjoint writes -> simple parallel for, the cdot
 * reduction reorders -> within DD fuzz tol). conj selects ConjTrans. */
static void wgbmv_t_contig(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t KL, std::ptrdiff_t KU, TC alpha,
                           const TC *a, std::size_t lda, const TC *x, TC *y, bool conj)
{
#ifdef _OPENMP
    const bool use_omp = (n >= WGBMV_OMP_MIN && blas_omp_should_thread());
    #pragma omp parallel for if(use_omp) schedule(static) num_threads(blas_omp_max_threads())
#endif
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        std::ptrdiff_t i_lo, i_hi; wgbmv_band(j, m, KL, KU, i_lo, i_hi);
        const TC s = mf_kernels::cdot(i_hi - i_lo, &A_(KU - j + i_lo, j), &x[i_lo], conj);
        y[j] = cadd(y[j], cmul(alpha, s));
    }
}

static void wgbmv_core(
    char trans,
    std::ptrdiff_t m, std::ptrdiff_t n,
    std::ptrdiff_t KL, std::ptrdiff_t KU,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    const TC *x, std::ptrdiff_t incx,
    const TC *beta_,
    TC *y, std::ptrdiff_t incy)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char TRANS = up(&trans);
    const bool notrans = (TRANS == 'N');
    const bool conj = (TRANS == 'C');

    if (m == 0 || n == 0 || (ceq0(alpha) && ceq1(beta))) return;

    const std::ptrdiff_t leny = notrans ? m : n;
    const std::ptrdiff_t lenx = notrans ? n : m;

    mf_kernels::cscale_y(leny, beta, y, incy);
    if (ceq0(alpha)) return;

    if (incx == 1 && incy == 1) {
        if (notrans) wgbmv_n_contig(m, n, KL, KU, alpha, a, lda, x, y);
        else         wgbmv_t_contig(m, n, KL, KU, alpha, a, lda, x, y, conj);
        return;
    }

    /* Strided x,y: gather to unit stride (y already beta-applied), run the SIMD
     * core, scatter y back. Handles negative increments; O(lenx+leny) gather. */
    const TC *xbase = (incx < 0) ? x - (std::ptrdiff_t)(lenx - 1) * incx : x;
    TC *ybase = (incy < 0) ? y - (std::ptrdiff_t)(leny - 1) * incy : y;
    std::vector<TC> xs(static_cast<std::size_t>(lenx)), ys(static_cast<std::size_t>(leny));
    for (std::ptrdiff_t i = 0; i < lenx; ++i) xs[i] = xbase[(std::ptrdiff_t)i * incx];
    for (std::ptrdiff_t i = 0; i < leny; ++i) ys[i] = ybase[(std::ptrdiff_t)i * incy];
    if (notrans) wgbmv_n_contig(m, n, KL, KU, alpha, a, lda, xs.data(), ys.data());
    else         wgbmv_t_contig(m, n, KL, KU, alpha, a, lda, xs.data(), ys.data(), conj);
    for (std::ptrdiff_t i = 0; i < leny; ++i) ybase[(std::ptrdiff_t)i * incy] = ys[i];
}

extern "C" {
EPBLAS_FACADE_GBMV(wgbmv, TC)
}

#undef A_
