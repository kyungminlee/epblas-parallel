/* mtrmv — multifloats real DD triangular matrix-vector. */

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
#include "mf_omp.h"
#define MTRMV_OMP_MIN 128
#define MTRMV_MAX_CPUS 256
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::eq0;

using mf_util::up;  /* char flag uppercase — mf_util.h */

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Contiguous (unit-stride) dense triangular matvec, x[0..n-1] in logical order.
 * Per column j this is the full-triangle twin of the band cores: NoTrans is an
 * axpy of column j (scaled by x[j]) into the off-diagonal rows -> mf_kernels::axpy_add
 * (order-free, bit-exact); Trans is a dot of column j with the off-diagonal x
 * rows -> mf_kernels::dot (vector accumulate + hreduce, within tolerance). The
 * diagonal scaling matches the scalar reference: applied after the axpy for
 * NoTrans, folded into the dot seed for Trans. A strided x is gathered to a
 * contiguous scratch by the caller and scattered back. */
static void mtrmv_contig(bool upper, bool trans, bool nounit,
                         std::ptrdiff_t n, const TR *a, std::size_t lda, TR *x)
{
    if (!trans) {
        if (!upper) {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                const TR *aj = &A_(0, j);
                const TR temp = x[j];
                if (!eq0(temp))
                    mf_kernels::axpy_add(n - 1 - j, &x[j + 1], &aj[j + 1], temp);
                if (nounit) x[j] = x[j] * aj[j];
            }
        } else {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const TR *aj = &A_(0, j);
                const TR temp = x[j];
                if (!eq0(temp))
                    mf_kernels::axpy_add(j, &x[0], &aj[0], temp);
                if (nounit) x[j] = x[j] * aj[j];
            }
        }
    } else {
        if (!upper) {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const TR *aj = &A_(0, j);
                TR temp = nounit ? (x[j] * aj[j]) : x[j];
                temp = temp + mf_kernels::dot(n - 1 - j, &aj[j + 1], &x[j + 1]);
                x[j] = temp;
            }
        } else {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                const TR *aj = &A_(0, j);
                TR temp = nounit ? (x[j] * aj[j]) : x[j];
                temp = temp + mf_kernels::dot(j, &aj[0], &x[0]);
                x[j] = temp;
            }
        }
    }
}

#ifdef _OPENMP
/* NoTrans tiled column-AXPY kernel (real-DD twin of wtrmv_kernel_N). Reads only
 * the contiguous matrix column block [m_from,m_to); writes its own rows directly
 * plus the off-block "spill" rows into the thread's private y slot (merged by a
 * bounded reduction). DTB tiling keeps the active y-tile hot. Each contiguous
 * column run is a SIMD AXPY (mf_kernels::axpy_add) so the threaded path stays
 * vectorized like the serial one. */
static void mtrmv_kernel_N(bool upper, bool nounit, std::ptrdiff_t n,
                           std::ptrdiff_t m_from, std::ptrdiff_t m_to,
                           const TR *a, std::size_t lda, const TR *x, TR *y)
{
    const std::ptrdiff_t TB = 32;
    for (std::ptrdiff_t is = m_from; is < m_to; is += TB) {
        std::ptrdiff_t min_i = (m_to - is < TB) ? m_to - is : TB;
        if (upper && is > 0) {
            for (std::ptrdiff_t j = is; j < is + min_i; ++j) {
                const TR xj = x[j];
                if (!eq0(xj)) mf_kernels::axpy_add(is, &y[0], &A_(0, j), xj);
            }
        }
        for (std::ptrdiff_t i = is; i < is + min_i; ++i) {
            if (upper && i > is) {
                const TR xi = x[i];
                if (!eq0(xi)) mf_kernels::axpy_add(i - is, &y[is], &A_(is, i), xi);
            }
            y[i] = y[i] + (nounit ? A_(i, i) * x[i] : x[i]);
            if (!upper && i + 1 < is + min_i) {
                const TR xi = x[i];
                if (!eq0(xi))
                    mf_kernels::axpy_add(is + min_i - (i + 1), &y[i + 1], &A_(i + 1, i), xi);
            }
        }
        if (!upper && is + min_i < n) {
            for (std::ptrdiff_t j = is; j < is + min_i; ++j) {
                const TR xj = x[j];
                if (!eq0(xj))
                    mf_kernels::axpy_add(n - (is + min_i), &y[is + min_i], &A_(is + min_i, j), xj);
            }
        }
    }
}

/* Threaded contiguous-x core, mirroring wtrmv_omp_contig with real-DD math.
 * Matrix access stays COLUMN-contiguous:
 *   - Trans: each x[j] is an independent dot over a contiguous column slice
 *     (disjoint writes → no reduction), cyclic schedule(static,1).
 *   - NoTrans: OpenBLAS contiguous row-block scheme via mf_omp::tri_area_bounds —
 *     each thread reads only its column block and merges its BOUNDED spill rows,
 *     replacing the old per-thread accumulator + O(nthreads·n) reduction (which floored
 *     par4 scaling at large n; see [[project_l2_rowgather_scaling]]).
 * DD addition reorders vs the serial path → within fuzz tol; the serial path
 * stays bit-exact. Operates on a contiguous x; the strided dispatch gathers /
 * scatters around it. Returns true on success, false if a scratch alloc failed
 * (caller falls back to serial). */
static bool mtrmv_omp_contig(bool upper, bool trans, bool nounit,
                             std::ptrdiff_t n, const TR *a, std::size_t lda,
                             TR *x, std::ptrdiff_t nthreads)
{
    if (trans) {
        TR *y_buf = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
        if (!y_buf) return false;
        #pragma omp parallel num_threads(nthreads)
        {
            if (!upper) {
                #pragma omp for schedule(static, 1)
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    TR temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const TR *aj = &A_(0, j);
                    y_buf[j] = temp + mf_kernels::dot(n - 1 - j, &aj[j + 1], &x[j + 1]);
                }
            } else {
                #pragma omp for schedule(static, 1)
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    TR temp = nounit ? (x[j] * A_(j, j)) : x[j];
                    const TR *aj = &A_(0, j);
                    y_buf[j] = temp + mf_kernels::dot(j, &aj[0], &x[0]);
                }
            }
            #pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = y_buf[i];
        }
        std::free(y_buf);
        return true;
    } else {
        /* NoTrans: contiguous row-block scheme. Each thread reads only its matrix
         * column block (good cache locality vs cyclic) and merges its bounded
         * spill rows — beats the full per-thread accumulator + O(nthreads·n) reduction
         * at large n. Same equal-area split as wtrmv: UPPER column work grows with
         * the index ⇒ heavy_high=upper; the forward slices are read REVERSED for
         * upper so the thin top slice carries the heavy top rows. */
        std::ptrdiff_t *range = static_cast<std::ptrdiff_t *>(
            std::malloc((std::size_t)(nthreads + 1) * sizeof(std::ptrdiff_t)));
        if (!range) return false;
        std::ptrdiff_t ncpu = mf_omp::tri_area_bounds(n, nthreads, 7, 16, upper,
                                           MTRMV_MAX_CPUS, range);
        TR *buf_all = static_cast<TR *>(
            std::calloc((std::size_t)ncpu * (std::size_t)n, sizeof(TR)));
        if (!buf_all) { std::free(range); return false; }
        #pragma omp parallel for schedule(static, 1) num_threads(ncpu)
        for (std::ptrdiff_t tid = 0; tid < ncpu; ++tid)
        {
            TR *y = &buf_all[(std::size_t)tid * n];  /* calloc-zeroed */
            std::ptrdiff_t m_from, m_to;
            if (upper) { m_from = range[ncpu - tid - 1]; m_to = range[ncpu - tid]; }
            else       { m_from = range[tid];            m_to = range[tid + 1]; }
            if (m_from < m_to)
                mtrmv_kernel_N(upper, nounit, n, m_from, m_to, a, lda, x, y);
        }
        /* Bounded reduction: merge each thread's spill rows into slot 0. */
        if (upper) {
            for (std::ptrdiff_t t = 1; t < ncpu; ++t) {
                std::ptrdiff_t m_to_t = range[ncpu - t];
                const TR *slot = &buf_all[(std::size_t)t * n];
                for (std::ptrdiff_t i = 0; i < m_to_t; ++i)
                    buf_all[i] = buf_all[i] + slot[i];
            }
        } else {
            for (std::ptrdiff_t t = 1; t < ncpu; ++t) {
                std::ptrdiff_t m_from_t = range[t];
                const TR *slot = &buf_all[(std::size_t)t * n];
                for (std::ptrdiff_t i = m_from_t; i < n; ++i)
                    buf_all[i] = buf_all[i] + slot[i];
            }
        }
        for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = buf_all[i];
        std::free(buf_all); std::free(range);
        return true;
    }
}

/* Threaded in-place dense triangular matvec. incx==1 drives the contiguous core
 * directly; strided gathers into a contiguous buffer, drives the core, scatters
 * back. Returns true if handled. */
__attribute__((noinline)) static bool mtrmv_omp(
    bool upper, bool trans, bool nounit, std::ptrdiff_t n,
    const TR *a, std::size_t lda, TR *x, std::ptrdiff_t incx)
{
    if (n < MTRMV_OMP_MIN || !blas_omp_should_thread())
        return false;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > MTRMV_MAX_CPUS) nthreads = MTRMV_MAX_CPUS;

    if (incx == 1)
        return mtrmv_omp_contig(upper, trans, nounit, n, a, lda, x, nthreads);

    TR *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    TR *xbuf = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
    if (!xbuf) return false;
    for (std::ptrdiff_t i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];
    bool ok = mtrmv_omp_contig(upper, trans, nounit, n, a, lda, xbuf, nthreads);
    if (ok)
        for (std::ptrdiff_t i = 0; i < n; ++i) xbase[(std::ptrdiff_t)i * incx] = xbuf[i];
    std::free(xbuf);
    return ok;
}
#endif

static void mtrmv_core(
    char uplo, char trans, char diag,
    std::ptrdiff_t n,
    const TR *a, std::ptrdiff_t lda,
    TR *x, std::ptrdiff_t incx)
{
    const char UPLO = up(&uplo);
    char TRANS = up(&trans);
    if (TRANS == 'C') TRANS = 'T';
    const char DIAG = up(&diag);
    const bool nounit = (DIAG != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    if (n >= MTRMV_OMP_MIN && blas_omp_available()
        && mtrmv_omp(UPLO == 'U', TRANS != 'N', nounit, n, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        mtrmv_contig(UPLO == 'U', TRANS != 'N', nounit, n, a, (std::size_t)lda, x);
        return;
    }

    /* Strided x: linearize to a contiguous scratch in logical order, run the
     * SIMD contiguous core, scatter back (2N copies vs O(N^2) band work). The
     * in-place strided walk is kept only as the scratch-alloc-failure fallback. */
    const std::ptrdiff_t base = (incx < 0) ? -(std::ptrdiff_t)(n - 1) * incx : 0;
    TR *xs = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
    if (xs) {
        for (std::ptrdiff_t i = 0; i < n; ++i) xs[i] = x[base + (std::ptrdiff_t)i * incx];
        mtrmv_contig(UPLO == 'U', TRANS != 'N', nounit, n, a, (std::size_t)lda, xs);
        for (std::ptrdiff_t i = 0; i < n; ++i) x[base + (std::ptrdiff_t)i * incx] = xs[i];
        std::free(xs);
        return;
    }

    std::ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
    if (TRANS == 'N') {
        if (UPLO == 'L') {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                const TR temp = x[kx + j * incx];
                if (!eq0(temp))
                    for (std::ptrdiff_t i = j + 1; i < n; ++i) x[kx + i * incx] = x[kx + i * incx] + temp * A_(i, j);
                if (nounit) x[kx + j * incx] = x[kx + j * incx] * A_(j, j);
            }
        } else {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const TR temp = x[kx + j * incx];
                if (!eq0(temp))
                    for (std::ptrdiff_t i = 0; i < j; ++i) x[kx + i * incx] = x[kx + i * incx] + temp * A_(i, j);
                if (nounit) x[kx + j * incx] = x[kx + j * incx] * A_(j, j);
            }
        }
    } else {
        if (UPLO == 'L') {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                TR temp = x[kx + j * incx];
                if (nounit) temp = temp * A_(j, j);
                for (std::ptrdiff_t i = j + 1; i < n; ++i) temp = temp + A_(i, j) * x[kx + i * incx];
                x[kx + j * incx] = temp;
            }
        } else {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                TR temp = x[kx + j * incx];
                if (nounit) temp = temp * A_(j, j);
                for (std::ptrdiff_t i = 0; i < j; ++i) temp = temp + A_(i, j) * x[kx + i * incx];
                x[kx + j * incx] = temp;
            }
        }
    }
}

extern "C" {
EPBLAS_FACADE_TRMV(mtrmv, TR)
}

#undef A_
