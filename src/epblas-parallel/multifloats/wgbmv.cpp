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

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
const R rzero{0.0, 0.0};
const T czero{ rzero, rzero };
using mf_kernels::cmul;
using mf_kernels::cadd;
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* A band column stores its in-band rows [i_lo,i_hi) contiguously at &A_(KU-j+i_lo,j),
 * so per column NoTrans is a SIMD AXPY into y and Trans is a SIMD (conj-)dot of the
 * band run against x — the same stride-1 kernels used by the triangular twins. */
static inline void wgbmv_band(int j, int M, int KL, int KU, int &i_lo, int &i_hi) {
    i_lo = (j - KU > 0) ? (j - KU) : 0;
    i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
}

#ifdef _OPENMP
/* Threaded NoTrans (contiguous x len N, y len M pre-beta'd). Each thread owns a
 * disjoint output-row range [r0,r1) and folds only the columns whose band touches
 * it, clipping each contiguous band run to [r0,r1). Disjoint y rows -> no race,
 * no reduction; each row still accumulates ascending-j -> BIT-EXACT vs serial. */
static bool wgbmv_n_omp(int M, int N, int KL, int KU, T alpha,
                        const T *a, std::size_t lda, const T *x, T *y)
{
    int nthreads = blas_omp_max_threads();
    if (nthreads <= 1 || omp_in_parallel()) return false;
    if (nthreads > WGBMV_MAX_CPUS) nthreads = WGBMV_MAX_CPUS;

    #pragma omp parallel num_threads(nthreads)
    {
        const int tid = omp_get_thread_num();
        const int r0 = (int)((std::ptrdiff_t)M * tid / nthreads);
        const int r1 = (int)((std::ptrdiff_t)M * (tid + 1) / nthreads);
        const int j0 = (r0 - KL > 0) ? (r0 - KL) : 0;
        const int j1 = (r1 + KU < N) ? (r1 + KU) : N;
        for (int j = j0; j < j1; ++j) {
            const T tmp = cmul(alpha, x[j]);
            if (ceq0(tmp)) continue;
            int i_lo, i_hi; wgbmv_band(j, M, KL, KU, i_lo, i_hi);
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
static void wgbmv_n_contig(int M, int N, int KL, int KU, T alpha,
                           const T *a, std::size_t lda, const T *x, T *y)
{
#ifdef _OPENMP
    if (M >= WGBMV_OMP_MIN && blas_omp_max_threads() > 1
        && wgbmv_n_omp(M, N, KL, KU, alpha, a, lda, x, y))
        return;
#endif
    for (int j = 0; j < N; ++j) {
        const T tmp = cmul(alpha, x[j]);
        if (ceq0(tmp)) continue;
        int i_lo, i_hi; wgbmv_band(j, M, KL, KU, i_lo, i_hi);
        mf_kernels::caxpy_add(i_hi - i_lo, &y[i_lo], &A_(KU - j + i_lo, j), tmp);
    }
}

/* Contiguous Trans/ConjTrans core (x len M, y len N, y pre-beta'd). Each y[j] is
 * an independent band conj-dot (disjoint writes -> simple parallel for, the cdot
 * reduction reorders -> within DD fuzz tol). conj selects ConjTrans. */
static void wgbmv_t_contig(int M, int N, int KL, int KU, T alpha,
                           const T *a, std::size_t lda, const T *x, T *y, bool conj)
{
#ifdef _OPENMP
    const int use_omp = (N >= WGBMV_OMP_MIN && blas_omp_max_threads() > 1
                         && !omp_in_parallel());
    #pragma omp parallel for if(use_omp) schedule(static) num_threads(blas_omp_max_threads())
#endif
    for (int j = 0; j < N; ++j) {
        int i_lo, i_hi; wgbmv_band(j, M, KL, KU, i_lo, i_hi);
        const T s = mf_kernels::cdot(i_hi - i_lo, &A_(KU - j + i_lo, j), &x[i_lo], conj);
        y[j] = cadd(y[j], cmul(alpha, s));
    }
}

extern "C" void wgbmv_(
    const char *trans,
    const int *m_, const int *n_,
    const int *kl_, const int *ku_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *x, const int *incx_,
    const T *beta_,
    T *y, const int *incy_,
    std::size_t trans_len)
{
    (void)trans_len;
    const int M = *m_, N = *n_;
    const int KL = *kl_, KU = *ku_;
    const std::size_t lda = static_cast<std::size_t>(*lda_);
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char TR = up(trans);
    const bool notrans = (TR == 'N');
    const bool conj = (TR == 'C');

    if (M == 0 || N == 0 || (ceq0(alpha) && ceq1(beta))) return;

    const int leny = notrans ? M : N;
    const int lenx = notrans ? N : M;

    if (!ceq1(beta)) {
        int iy = (incy < 0) ? -(leny - 1) * incy : 0;
        if (ceq0(beta)) for (int i = 0; i < leny; ++i) { y[iy] = czero; iy += incy; }
        else                  for (int i = 0; i < leny; ++i) { y[iy] = cmul(beta, y[iy]); iy += incy; }
    }
    if (ceq0(alpha)) return;

    if (incx == 1 && incy == 1) {
        if (notrans) wgbmv_n_contig(M, N, KL, KU, alpha, a, lda, x, y);
        else         wgbmv_t_contig(M, N, KL, KU, alpha, a, lda, x, y, conj);
        return;
    }

    /* Strided x,y: gather to unit stride (y already beta-applied), run the SIMD
     * core, scatter y back. Handles negative increments; O(lenx+leny) gather. */
    const T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(lenx - 1) * incx : x;
    T *ybase = (incy < 0) ? y - (std::ptrdiff_t)(leny - 1) * incy : y;
    std::vector<T> xs(static_cast<std::size_t>(lenx)), ys(static_cast<std::size_t>(leny));
    for (int i = 0; i < lenx; ++i) xs[i] = xbase[(std::ptrdiff_t)i * incx];
    for (int i = 0; i < leny; ++i) ys[i] = ybase[(std::ptrdiff_t)i * incy];
    if (notrans) wgbmv_n_contig(M, N, KL, KU, alpha, a, lda, xs.data(), ys.data());
    else         wgbmv_t_contig(M, N, KL, KU, alpha, a, lda, xs.data(), ys.data(), conj);
    for (int i = 0; i < leny; ++i) ybase[(std::ptrdiff_t)i * incy] = ys[i];
}

#undef A_
