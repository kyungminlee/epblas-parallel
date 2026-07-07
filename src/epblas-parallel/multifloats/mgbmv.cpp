/* mgbmv — multifloats real DD general band matrix-vector multiply.
 *   y := alpha*A*x + beta*y  or  y := alpha*A^T*x + beta*y
 * Band storage: A(i,j) at AB[(ku + i - j) + j*lda].
 *
 * Reference algorithm + OMP over j on T-path only (N-path writes overlap).
 */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
#define MGBMV_OMP_MIN 64
#define MGBMV_MAX_CPUS 256
const TR zero_dd{0.0, 0.0};
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* A band column stores its in-band rows [i_lo,i_hi) contiguously at &A_(KU-j+i_lo,j),
 * so per column NoTrans is a SIMD AXPY into y and Trans is a SIMD dot of the band run
 * against x — the same stride-1 kernels used by the triangular twins. (The OMP paths
 * thread over ptrdiff_t rows and clamp the band to the owned range inline, so they
 * keep their own bounds.) */
static inline void mgbmv_band(std::ptrdiff_t j, std::ptrdiff_t m, std::ptrdiff_t KL, std::ptrdiff_t KU, std::ptrdiff_t &i_lo, std::ptrdiff_t &i_hi) {
    i_lo = (j - KU > 0) ? (j - KU) : 0;
    i_hi = (j + KL + 1 < m) ? (j + KL + 1) : m;
}

namespace {
/* Contiguous (incx==incy==1) cores. Each column's band segment A_(KU-j+i, j) is
 * stride-1 in the row index. NoTrans scatters that segment into y via SoA AXPY
 * (y[i] += tmp*col[i], distinct rows, ascending j -> bit-exact). Trans reduces it
 * against x via a vector dot (within DD fuzz tol). The strided entries gather
 * x/y to scratch and reuse these. */
void mgbmv_n_contig(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t KL, std::ptrdiff_t KU, TR alpha,
                    const TR *a, std::size_t lda, const TR *x, TR *y)
{
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        const TR tmp = alpha * x[j];
        std::ptrdiff_t i_lo, i_hi; mgbmv_band(j, m, KL, KU, i_lo, i_hi);
        const std::ptrdiff_t k = KU - j;
        if (i_hi > i_lo) mf_kernels::axpy_add(i_hi - i_lo, &y[i_lo], &A_(k + i_lo, j), tmp);
    }
}
void mgbmv_t_contig(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t KL, std::ptrdiff_t KU, TR alpha,
                    const TR *a, std::size_t lda, const TR *x, TR *y)
{
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        std::ptrdiff_t i_lo, i_hi; mgbmv_band(j, m, KL, KU, i_lo, i_hi);
        const std::ptrdiff_t k = KU - j;
        const TR s = (i_hi > i_lo) ? mf_kernels::dot(i_hi - i_lo, &A_(k + i_lo, j), &x[i_lo]) : zero_dd;
        y[j] = y[j] + alpha * s;
    }
}
}

#ifdef _OPENMP
/* Threaded NoTrans DD band matvec via restricted column-scatter. Each thread
 * owns a disjoint output-row range [lo,hi) and walks the columns touching it,
 * reading each column's band segment CONTIGUOUSLY (A_(KU-j+i, j) is stride-1 in
 * the row index — the same layout the serial scatter reads — vs the row-gather's
 * anti-diagonal lda-1 stride) and scattering only into its owned rows. Disjoint
 * writes -> no race, no fold. y already holds post-beta values, so each owned
 * y[i] accumulates alpha*x[j]*A(i,j) in ascending j -> identical association as
 * the serial/netlib scatter (bit-exact). alpha*x[j] is recomputed per column
 * (read-only x), which removes the shared ax buffer and its barrier. NoTrans
 * reads N of x, writes M of y. Returns true if handled. */
static bool mgbmv_n_omp(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t KL, std::ptrdiff_t KU, TR alpha,
                        const TR *a, std::ptrdiff_t lda,
                        const TR *x, std::ptrdiff_t incx, TR *y, std::ptrdiff_t incy)
{
    if (m < MGBMV_OMP_MIN || !blas_omp_should_thread())
        return false;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > MGBMV_MAX_CPUS) nthreads = MGBMV_MAX_CPUS;

    const std::ptrdiff_t ix0 = (incx < 0) ? -(n - 1) * incx : 0;
    const std::ptrdiff_t iy0 = (incy < 0) ? -static_cast<std::ptrdiff_t>(m - 1) * incy : 0;

    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t lo = blas_part_bound(m, tid, omp_get_num_threads());
        std::ptrdiff_t hi = blas_part_bound(m, tid + 1, omp_get_num_threads());
        /* columns whose band [j-KU, j+KL] intersects owned rows [lo,hi) */
        std::ptrdiff_t jlo = (lo - KL > 0) ? (lo - KL) : 0;
        std::ptrdiff_t jhi = (hi - 1 + KU + 1 < n) ? (hi + KU) : n;
        for (std::ptrdiff_t j = jlo; j < jhi; ++j) {
            const TR tmp = alpha * x[ix0 + j * incx];
            std::ptrdiff_t i_lo = (j - KU > lo) ? (j - KU) : lo;
            std::ptrdiff_t i_hi = (j + KL + 1 < hi) ? (j + KL + 1) : hi;
            const TR *col = &A_(KU - j + i_lo, j);   /* contiguous in i */
            if (incy == 1) {                        /* contiguous owned rows -> SoA AXPY */
                mf_kernels::axpy_add(i_hi - i_lo, &y[iy0 + i_lo], col, tmp);
            } else {
                for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                    TR *yi = &y[iy0 + i * incy];
                    *yi = *yi + tmp * (*col++);
                }
            }
        }
    }
    return true;
}

/* Threaded Trans DD band matvec (strided x and/or y; C mapped to T by caller).
 * Output columns partition across threads (each y[j]=alpha*Σ_i A(i,j)*x[i] disjoint).
 * Strided x gathered to contiguous so the inner dot reads x[i] directly. Trans reads
 * M of x, writes N of y. Bit-identical to the serial strided gather (ascending-i). */
static bool mgbmv_t_omp(std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t KL, std::ptrdiff_t KU, TR alpha,
                        const TR *a, std::ptrdiff_t lda,
                        const TR *x, std::ptrdiff_t incx, TR *y, std::ptrdiff_t incy)
{
    if (n < MGBMV_OMP_MIN || !blas_omp_should_thread())
        return false;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > MGBMV_MAX_CPUS) nthreads = MGBMV_MAX_CPUS;

    if (incy < 0) y -= static_cast<std::ptrdiff_t>(n - 1) * incy;

    const TR *xptr = x;
    TR *xbuf = nullptr;
    if (incx != 1) {
        xbuf = static_cast<TR *>(std::malloc(static_cast<std::size_t>(m) * sizeof(TR)));
        if (!xbuf) return false;
        mf_kernels::gather_strided(m, x, incx, xbuf);
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        std::ptrdiff_t tid = omp_get_thread_num();
        std::ptrdiff_t lo = blas_part_bound(n, tid, omp_get_num_threads());
        std::ptrdiff_t hi = blas_part_bound(n, tid + 1, omp_get_num_threads());
        for (std::ptrdiff_t j = lo; j < hi; ++j) {
            std::ptrdiff_t i_lo = (j - KU > 0) ? (j - KU) : 0;
            std::ptrdiff_t i_hi = (j + KL + 1 < m) ? (j + KL + 1) : m;
            std::ptrdiff_t k = KU - j;
            const TR *col = &A_(k + i_lo, j);
            TR s = mf_kernels::dot(i_hi - i_lo, col, &xptr[i_lo]);
            y[j * incy] = y[j * incy] + alpha * s;
        }
    }
    std::free(xbuf);
    return true;
}
#endif /* _OPENMP */

static void mgbmv_core(
    char trans,
    std::ptrdiff_t m, std::ptrdiff_t n,
    std::ptrdiff_t KL, std::ptrdiff_t KU,
    const TR *alpha_,
    const TR *a, std::ptrdiff_t lda,
    const TR *x, std::ptrdiff_t incx,
    const TR *beta_,
    TR *y, std::ptrdiff_t incy)
{
    const TR alpha = *alpha_, beta = *beta_;
    char TRANS = up(&trans);
    if (TRANS == 'C') TRANS = 'T';

    if (m == 0 || n == 0 || (eq0(alpha) && eq1(beta))) return;

    const std::ptrdiff_t leny = (TRANS == 'N') ? m : n;
    const std::ptrdiff_t lenx = (TRANS == 'N') ? n : m;

    mf_kernels::scale_y(leny, beta, y, incy);
    if (eq0(alpha)) return;

    if (TRANS == 'N') {
#ifdef _OPENMP
        /* NoTrans threads for contiguous AND strided x/y (the helper gathers strided
         * x and writes strided y); bit-identical to the serial scatter (ascending-j). */
        if (m >= MGBMV_OMP_MIN && blas_omp_available()
            && mgbmv_n_omp(m, n, KL, KU, alpha, a, lda, x, incx, y, incy))
            return;
#endif
        if (incx == 1 && incy == 1) {
            mgbmv_n_contig(m, n, KL, KU, alpha, a, (std::size_t)lda, x, y);
            return;
        }
        /* Strided: gather x (len N) and beta-scaled y (len M) to contiguous
         * scratch, run the SIMD scatter core, scatter y back. O(M+N) gather vs
         * O(M*band) work; the in-place strided walk is the alloc-fail fallback. */
        {
            TR *xs = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
            TR *ys = static_cast<TR *>(std::malloc((std::size_t)m * sizeof(TR)));
            if (xs && ys) {
                mf_kernels::gather_strided(n, x, incx, xs);
                mf_kernels::gather_strided(m, y, incy, ys);
                mgbmv_n_contig(m, n, KL, KU, alpha, a, (std::size_t)lda, xs, ys);
                mf_kernels::scatter_strided(m, y, incy, ys);
                std::free(xs); std::free(ys);
                return;
            }
            std::free(xs); std::free(ys);
        }
        {
            std::ptrdiff_t kx = (incx < 0) ? -(lenx - 1) * incx : 0;
            std::ptrdiff_t ky = (incy < 0) ? -(leny - 1) * incy : 0;
            std::ptrdiff_t jx = kx;
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const TR tmp = alpha * x[jx];
                std::ptrdiff_t iy = ky;
                std::ptrdiff_t i_lo, i_hi; mgbmv_band(j, m, KL, KU, i_lo, i_hi);
                const std::ptrdiff_t k = KU - j;
                for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                    y[iy] = y[iy] + tmp * A_(k + i, j);
                    iy += incy;
                }
                jx += incx;
                if (j >= KU) ky += incy;
            }
        }
    } else {
#ifdef _OPENMP
        /* Trans threads for contiguous AND strided x/y (the helper gathers strided
         * x and writes strided y). */
        if (n >= MGBMV_OMP_MIN && blas_omp_available()
            && mgbmv_t_omp(m, n, KL, KU, alpha, a, lda, x, incx, y, incy))
            return;
#endif
        if (incx == 1 && incy == 1) {
            mgbmv_t_contig(m, n, KL, KU, alpha, a, (std::size_t)lda, x, y);
            return;
        }
        /* Strided: gather x (len M) and beta-scaled y (len N), run the SIMD dot
         * core, scatter y back; the in-place strided walk is the alloc-fail fallback. */
        {
            TR *xs = static_cast<TR *>(std::malloc((std::size_t)m * sizeof(TR)));
            TR *ys = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
            if (xs && ys) {
                mf_kernels::gather_strided(m, x, incx, xs);
                mf_kernels::gather_strided(n, y, incy, ys);
                mgbmv_t_contig(m, n, KL, KU, alpha, a, (std::size_t)lda, xs, ys);
                mf_kernels::scatter_strided(n, y, incy, ys);
                std::free(xs); std::free(ys);
                return;
            }
            std::free(xs); std::free(ys);
        }
        {
            std::ptrdiff_t kx = (incx < 0) ? -(lenx - 1) * incx : 0;
            std::ptrdiff_t ky = (incy < 0) ? -(leny - 1) * incy : 0;
            std::ptrdiff_t jy = ky;
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                TR s = zero_dd;
                std::ptrdiff_t ix = kx;
                std::ptrdiff_t i_lo, i_hi; mgbmv_band(j, m, KL, KU, i_lo, i_hi);
                const std::ptrdiff_t k = KU - j;
                for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                    s = s + A_(k + i, j) * x[ix];
                    ix += incx;
                }
                y[jy] = y[jy] + alpha * s;
                jy += incy;
                if (j >= KU) kx += incx;
            }
        }
    }
}

extern "C" {
EPBLAS_FACADE_GBMV(mgbmv, TR)
}

#undef A_
