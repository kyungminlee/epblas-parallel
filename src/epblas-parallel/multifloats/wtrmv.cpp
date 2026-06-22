/* wtrmv — multifloats complex DD triangular matrix-vector. */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <vector>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#ifdef _OPENMP
#include <cmath>
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#define WTRMV_OMP_MIN 128
#define WTRMV_MAX_CPUS 256
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]

/* Bit-exact row-tiled lower-triangular NoTrans serial matvec (incx==1).
 * The plain column-AXPY loop streams a full n-row column per j and thrashes the
 * x cache; the OpenBLAS clone tiles the output rows even at one thread and wins
 * ~8%. This recovers that by sweeping a 32-row output tile that stays cache-hot,
 * while preserving the plain loop's accumulation order for each output element
 * (diagonal first, then strictly descending column index) so the result is
 * byte-identical to the untiled path. */
static void wtrmv_serial_N_lower(bool nounit, std::ptrdiff_t n,
                                 const T *a, std::size_t lda, T *x) {
    const std::ptrdiff_t RB = 32;
    std::vector<T> ybuf(static_cast<std::size_t>(n));
    T *y = ybuf.data();
    for (std::ptrdiff_t ib = 0; ib < n; ib += RB) {
        const std::ptrdiff_t ie = ib + RB < n ? ib + RB : n;
        for (std::ptrdiff_t i = ib; i < ie; ++i)
            y[i] = nounit ? cmul(x[i], A_(i, i)) : x[i];
        for (std::ptrdiff_t j = ie - 1; j >= ib; --j) {            /* within-tile, descending j */
            const T xj = x[j];
            if (ceq0(xj)) continue;
            const T *col = &A_(0, j);
            mf_kernels::caxpy_add(ie - (j + 1), &y[j + 1], &col[j + 1], xj);
        }
        for (std::ptrdiff_t j = ib - 1; j >= 0; --j) {             /* below-tile columns, descending j */
            const T xj = x[j];
            if (ceq0(xj)) continue;
            const T *col = &A_(0, j);
            mf_kernels::caxpy_add(ie - ib, &y[ib], &col[ib], xj);
        }
    }
    for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = y[i];
}

#ifdef _OPENMP
/* NoTrans tiled column-AXPY kernel (OpenBLAS trmv_thread.c kernel_N). Reads only
 * the contiguous matrix column block [m_from,m_to); writes its own rows directly
 * plus the off-block "spill" rows into the thread's private y slot (merged by a
 * bounded reduction). DTB tiling keeps the active y-tile hot. Each contiguous
 * column run is a SIMD AXPY (mf_kernels::caxpy_add) so the threaded path stays
 * vectorized like the serial one. */
static void wtrmv_kernel_N(bool upper, bool nounit, std::ptrdiff_t n,
                           std::ptrdiff_t m_from, std::ptrdiff_t m_to,
                           const T *a, std::size_t lda, const T *x, T *y)
{
    const std::ptrdiff_t TB = 32;
    for (std::ptrdiff_t is = m_from; is < m_to; is += TB) {
        std::ptrdiff_t min_i = (m_to - is < TB) ? m_to - is : TB;
        if (upper && is > 0) {
            for (std::ptrdiff_t j = is; j < is + min_i; ++j) {
                const T xj = x[j];
                if (!ceq0(xj)) mf_kernels::caxpy_add(is, &y[0], &A_(0, j), xj);
            }
        }
        for (std::ptrdiff_t i = is; i < is + min_i; ++i) {
            if (upper && i > is) {
                const T xi = x[i];
                if (!ceq0(xi)) mf_kernels::caxpy_add(i - is, &y[is], &A_(is, i), xi);
            }
            y[i] = cadd(y[i], nounit ? cmul(A_(i, i), x[i]) : x[i]);
            if (!upper && i + 1 < is + min_i) {
                const T xi = x[i];
                if (!ceq0(xi))
                    mf_kernels::caxpy_add(is + min_i - (i + 1), &y[i + 1], &A_(i + 1, i), xi);
            }
        }
        if (!upper && is + min_i < n) {
            for (std::ptrdiff_t j = is; j < is + min_i; ++j) {
                const T xj = x[j];
                if (!ceq0(xj))
                    mf_kernels::caxpy_add(n - (is + min_i), &y[is + min_i], &A_(is + min_i, j), xj);
            }
        }
    }
}

/* Threaded contiguous-x core, mirroring mtrmv_omp_contig with complex DD math.
 * The earlier row-gather threaded poorly: NoTrans read the matrix by ROW
 * (A_(r,c) strided by lda) — cache-hostile for column-major storage — and the
 * contiguous-block row partition load-imbalanced the triangular work. This keeps
 * matrix access COLUMN-contiguous and uses schedule(static,1) cyclic balancing:
 *   - Trans/ConjTrans: each x[j] is an independent contiguous-column dot
 *     (conjugated when conj; disjoint writes → no reduction).
 *   - NoTrans: per-thread accumulator + column AXPY, reduced at the end.
 * DD addition reorders vs serial → within fuzz tol; serial stays bit-exact.
 * Returns true on success, false if a scratch alloc failed. */
static bool wtrmv_omp_contig(bool upper, bool trans, bool conj, bool nounit,
                             std::ptrdiff_t n, const T *a, std::size_t lda,
                             T *x, std::ptrdiff_t nthreads)
{
    if (trans) {
        T *y_buf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
        if (!y_buf) return false;
        #pragma omp parallel num_threads(nthreads)
        {
            #pragma omp for schedule(static, 1)
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                T diagc = A_(j, j); if (conj) diagc = cconj(diagc);
                T s = nounit ? cmul(diagc, x[j]) : x[j];
                const T *aj = &A_(0, j);
                if (upper) s = cadd(s, mf_kernels::cdot(j, &aj[0], &x[0], conj));
                else       s = cadd(s, mf_kernels::cdot(n - j - 1, &aj[j + 1], &x[j + 1], conj));
                y_buf[j] = s;
            }
            #pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = y_buf[i];
        }
        std::free(y_buf);
        return true;
    } else {
        /* NoTrans: OpenBLAS contiguous row-block scheme. Each thread reads only
         * its matrix column block (good cache locality vs cyclic) and merges its
         * bounded spill rows — beats the full per-thread accumulator + O(nthreads·n)
         * reduction at large n. */
        std::ptrdiff_t *range = static_cast<std::ptrdiff_t *>(
            std::malloc((std::size_t)(nthreads + 1) * sizeof(std::ptrdiff_t)));
        if (!range) return false;
        /* Same equal-area split as mspmv (proof in the simd-audit log): UPPER
         * column work grows with the index ⇒ heavy_high=upper. The forward
         * ascending slices are read REVERSED for upper so the thin top slice
         * carries the heavy top rows. mask 7/min 16 are this routine's tuning. */
        std::ptrdiff_t ncpu = mf_omp::tri_area_bounds(n, nthreads, 7, 16, upper,
                                           WTRMV_MAX_CPUS, range);
        T *buf_all = static_cast<T *>(
            std::calloc((std::size_t)ncpu * (std::size_t)n, sizeof(T)));
        if (!buf_all) { std::free(range); return false; }
        #pragma omp parallel num_threads(ncpu)
        {
            const std::ptrdiff_t tid = omp_get_thread_num();
            T *y = &buf_all[(std::size_t)tid * n];  /* calloc-zeroed */
            std::ptrdiff_t m_from, m_to;
            if (upper) { m_from = range[ncpu - tid - 1]; m_to = range[ncpu - tid]; }
            else       { m_from = range[tid];            m_to = range[tid + 1]; }
            if (m_from < m_to)
                wtrmv_kernel_N(upper, nounit, n, m_from, m_to, a, lda, x, y);
        }
        /* Bounded reduction: merge each thread's spill rows into slot 0. */
        if (upper) {
            for (std::ptrdiff_t t = 1; t < ncpu; ++t) {
                std::ptrdiff_t m_to_t = range[ncpu - t];
                const T *slot = &buf_all[(std::size_t)t * n];
                for (std::ptrdiff_t i = 0; i < m_to_t; ++i)
                    buf_all[i] = cadd(buf_all[i], slot[i]);
            }
        } else {
            for (std::ptrdiff_t t = 1; t < ncpu; ++t) {
                std::ptrdiff_t m_from_t = range[t];
                const T *slot = &buf_all[(std::size_t)t * n];
                for (std::ptrdiff_t i = m_from_t; i < n; ++i)
                    buf_all[i] = cadd(buf_all[i], slot[i]);
            }
        }
        for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = buf_all[i];
        std::free(buf_all); std::free(range);
        return true;
    }
}

/* Threaded in-place complex dense triangular matvec. incx==1 drives the
 * contiguous core directly; strided gathers/scatters around it. */
__attribute__((noinline)) static bool wtrmv_omp(
    bool upper, bool trans, bool conj, bool nounit, std::ptrdiff_t n,
    const T *a, std::size_t lda, T *x, std::ptrdiff_t incx)
{
    if (n < WTRMV_OMP_MIN || !blas_omp_should_thread())
        return false;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > WTRMV_MAX_CPUS) nthreads = WTRMV_MAX_CPUS;

    if (incx == 1)
        return wtrmv_omp_contig(upper, trans, conj, nounit, n, a, lda, x, nthreads);

    T *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    T *xbuf = static_cast<T *>(std::malloc((std::size_t)n * sizeof(T)));
    if (!xbuf) return false;
    for (std::ptrdiff_t i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];
    bool ok = wtrmv_omp_contig(upper, trans, conj, nounit, n, a, lda, xbuf, nthreads);
    if (ok)
        for (std::ptrdiff_t i = 0; i < n; ++i) xbase[(std::ptrdiff_t)i * incx] = xbuf[i];
    std::free(xbuf);
    return ok;
}
#endif

/* Contiguous (incx==1) in-place serial triangular matvec. NoTrans is a column
 * AXPY (mf_kernels::caxpy_add, bit-exact); Trans/ConjTrans is a column dot (mf_kernels::cdot,
 * within DD fuzz tol). The strided entry gathers x to scratch and reuses this. */
static void wtrmv_serial_contig(bool upper, bool trans, bool conj, bool nounit,
                                std::ptrdiff_t n, const T *a, std::size_t lda, T *x)
{
    if (!trans) {
        if (!upper) {
            if (n >= 128) { wtrmv_serial_N_lower(nounit, n, a, lda, x); return; }
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                const T temp = x[j];
                if (!ceq0(temp))
                    mf_kernels::caxpy_add(n - 1 - j, &x[j + 1], &A_(j + 1, j), temp);
                if (nounit) x[j] = cmul(x[j], A_(j, j));
            }
        } else {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const T temp = x[j];
                if (!ceq0(temp))
                    mf_kernels::caxpy_add(j, &x[0], &A_(0, j), temp);
                if (nounit) x[j] = cmul(x[j], A_(j, j));
            }
        }
    } else {
        if (!upper) {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                T temp = x[j];
                if (nounit) temp = cmul(temp, conj ? cconj(A_(j, j)) : A_(j, j));
                temp = cadd(temp, mf_kernels::cdot(n - 1 - j, &A_(j + 1, j), &x[j + 1], conj));
                x[j] = temp;
            }
        } else {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                T temp = x[j];
                if (nounit) temp = cmul(temp, conj ? cconj(A_(j, j)) : A_(j, j));
                temp = cadd(temp, mf_kernels::cdot(j, &A_(0, j), &x[0], conj));
                x[j] = temp;
            }
        }
    }
}

static void wtrmv_core(
    char uplo, char trans, char diag,
    std::ptrdiff_t N,
    const T *a, std::ptrdiff_t lda,
    T *x, std::ptrdiff_t incx)
{
    const char UPLO = up(&uplo);
    const char TR   = up(&trans);
    const char DIAG = up(&diag);
    const bool nounit = (DIAG != 'U');

    if (N == 0) return;

#ifdef _OPENMP
    if (N >= WTRMV_OMP_MIN && blas_omp_available()
        && wtrmv_omp(UPLO == 'U', TR != 'N', TR == 'C', nounit, N, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        wtrmv_serial_contig(UPLO == 'U', TR != 'N', TR == 'C', nounit, N, a, lda, x);
        return;
    }

    /* Strided: gather x to contiguous scratch, run the SIMD contiguous core,
     * scatter back. The in-place strided walk is the alloc-fail fallback. */
    {
        const std::ptrdiff_t base = (incx < 0) ? -(std::ptrdiff_t)(N - 1) * incx : 0;
        T *xs = static_cast<T *>(std::malloc((std::size_t)N * sizeof(T)));
        if (xs) {
            for (std::ptrdiff_t i = 0; i < N; ++i) xs[i] = x[base + (std::ptrdiff_t)i * incx];
            wtrmv_serial_contig(UPLO == 'U', TR != 'N', TR == 'C', nounit, N, a, lda, xs);
            for (std::ptrdiff_t i = 0; i < N; ++i) x[base + (std::ptrdiff_t)i * incx] = xs[i];
            std::free(xs);
            return;
        }
    }

    {
        std::ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (std::ptrdiff_t j = N - 1; j >= 0; --j) {
                    const T temp = x[kx + j * incx];
                    if (!ceq0(temp))
                        for (std::ptrdiff_t i = j + 1; i < N; ++i) x[kx + i * incx] = cadd(x[kx + i * incx], cmul(temp, A_(i, j)));
                    if (nounit) x[kx + j * incx] = cmul(x[kx + j * incx], A_(j, j));
                }
            } else {
                for (std::ptrdiff_t j = 0; j < N; ++j) {
                    const T temp = x[kx + j * incx];
                    if (!ceq0(temp))
                        for (std::ptrdiff_t i = 0; i < j; ++i) x[kx + i * incx] = cadd(x[kx + i * incx], cmul(temp, A_(i, j)));
                    if (nounit) x[kx + j * incx] = cmul(x[kx + j * incx], A_(j, j));
                }
            }
        } else {
            const bool conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (std::ptrdiff_t j = 0; j < N; ++j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    for (std::ptrdiff_t i = j + 1; i < N; ++i) {
                        const T aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                        temp = cadd(temp, cmul(aij, x[kx + i * incx]));
                    }
                    x[kx + j * incx] = temp;
                }
            } else {
                for (std::ptrdiff_t j = N - 1; j >= 0; --j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp = cmul(temp, conj_a ? cconj(A_(j, j)) : A_(j, j));
                    for (std::ptrdiff_t i = 0; i < j; ++i) {
                        const T aij = conj_a ? cconj(A_(i, j)) : A_(i, j);
                        temp = cadd(temp, cmul(aij, x[kx + i * incx]));
                    }
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

extern "C" {
EPBLAS_FACADE_TRMV(wtrmv, T)
}

#undef A_
