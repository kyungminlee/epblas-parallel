/* mspmv — multifloats real DD symmetric packed matrix-vector multiply.
 *   y := alpha*A*x + beta*y
 *
 * Serial — same overlapping-y-writes problem as msbmv.
 */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#ifdef _OPENMP
#include <cstdlib>
#include <cmath>
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#define MSPMV_OMP_MIN 256
#define MSPMV_MAX_CPUS 256
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {
const TR zero_dd{0.0, 0.0};
}

#define AP_(idx) ap[static_cast<std::size_t>(idx)]

namespace {
/* Contiguous (incx==incy==1) symmetric packed matvec, y += alpha*A*x (beta
 * already applied by the caller). Per column j the packed slice &aj[..] is a
 * contiguous run shared by the reflected AXPY (y[i] += t1*col[i], order-free ->
 * bit-exact) and the symmetric dot (t2 += col[i]*x[i], vector accumulate +
 * hreduce -> within tolerance); y and x are distinct (spmv forbids aliasing).
 * The strided entry gathers x/y to scratch and reuses this. */
void mspmv_contig(bool upper, std::ptrdiff_t n, const TR *ap, const TR *x, TR alpha, TR *y)
{
    std::size_t kk = 0;
    if (upper) {
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            const TR *aj = &AP_(kk);            /* column j: rows 0..j-1, diag at aj[j] */
            const TR t1 = alpha * x[j];
            TR t2 = zero_dd;
            if (j > 0) {
                mf_kernels::axpy_add(j, &y[0], aj, t1);
                t2 = mf_kernels::dot(j, aj, &x[0]);
            }
            y[j] = y[j] + t1 * aj[j] + alpha * t2;
            kk += static_cast<std::size_t>(j) + 1;
        }
    } else {
        for (std::ptrdiff_t j = 0; j < n; ++j) {
            const TR *aj = &AP_(kk);            /* column j: diag at aj[0], rows j+1.. at aj[1.. ] */
            const TR t1 = alpha * x[j];
            y[j] = y[j] + t1 * aj[0];
            const std::ptrdiff_t len = n - 1 - j;
            if (len > 0) {
                mf_kernels::axpy_add(len, &y[j + 1], &aj[1], t1);
                y[j] = y[j] + alpha * mf_kernels::dot(len, &aj[1], &x[j + 1]);
            }
            kk += static_cast<std::size_t>(n - j);
        }
    }
}
}

#ifdef _OPENMP
/* Unit-stride threaded path (port of ob mspmv axpydot). Threads own disjoint
 * COLUMN ranges and accumulate into a private slot[n]; a single contiguous pass
 * over column j does both the scatter (slot[i]+=x[j]*aj[i]) and the symmetric
 * dot (temp2+=aj[i]*x[i]), reading column j of the packed triangle contiguously.
 * Per-thread slots are then AXPY-reduced with alpha factored into the final fold.
 * Contiguous column read (vs the row-gather's anti-diagonal col-jump that spans
 * the whole packed array) is what lets this scale ~3.3x. Reorders the per-row
 * sum vs serial -> within DD fuzz tol (strided path keeps the row-gather). */
static bool mspmv_axpydot(bool upper, std::ptrdiff_t n, const TR *ap,
                          const TR *x, TR alpha, TR *y, std::ptrdiff_t nthreads)
{
    std::ptrdiff_t range[MSPMV_MAX_CPUS + 1];
    /* Area-balanced column partition: per-column triangular work is ~j (upper) /
     * ~(n-j) (lower), so heavy_high=upper. mask 3/min 4 keep slot writes off
     * shared cache lines. */
    std::ptrdiff_t num_cpu = mf_omp::tri_area_bounds(n, nthreads, 3, 4, upper,
                                          MSPMV_MAX_CPUS, range);
    if (num_cpu <= 1) return false;

    TR *buf = static_cast<TR *>(std::calloc((std::size_t)num_cpu * n, sizeof(TR)));
    if (!buf) return false;

    #pragma omp parallel num_threads(num_cpu)
    {
        std::ptrdiff_t t = omp_get_thread_num();
        std::ptrdiff_t m_from = range[t];
        std::ptrdiff_t m_to   = range[t + 1];
        TR *slot = buf + (std::size_t)t * n;
        /* SIMD per-column slot scatter + symmetric dot, reusing the same SoA DD
         * kernels as the serial mspmv_contig (two contiguous passes over column
         * j) — writing into the per-thread slot instead of y. The scalar fused
         * loop this replaces left the threaded path un-vectorized while the
         * serial path was SIMD, capping par4/par1 at ~0.68. */
        if (upper) {
            for (std::ptrdiff_t j = m_from; j < m_to; ++j) {
                const TR temp1 = x[j];
                const TR *aj = &AP_((std::size_t)j * (j + 1) / 2);
                TR temp2 = zero_dd;
                if (j > 0) {
                    mf_kernels::axpy_add(j, &slot[0], aj, temp1);   /* slot[i] += temp1*aj[i] */
                    temp2 = mf_kernels::dot(j, aj, &x[0]);          /* sum aj[i]*x[i] */
                }
                slot[j] = slot[j] + (temp1 * aj[j] + temp2);
            }
        } else {
            for (std::ptrdiff_t j = m_from; j < m_to; ++j) {
                const TR temp1 = x[j];
                const TR *aj = &AP_((std::size_t)j * (2 * (std::size_t)n - j - 1) / 2);
                slot[j] = slot[j] + temp1 * aj[j];
                const std::ptrdiff_t len = n - 1 - j;
                if (len > 0) {
                    mf_kernels::axpy_add(len, &slot[j + 1], &aj[1], temp1);
                    slot[j] = slot[j] + mf_kernels::dot(len, &aj[1], &x[j + 1]);
                }
            }
        }
    }

    /* Bounded reduction: fold each thread's populated row window (alpha deferred
     * to here) straight onto y. */
    for (std::ptrdiff_t t = 0; t < num_cpu; ++t) {
        const TR *slot = buf + (std::size_t)t * n;
        std::ptrdiff_t from, to;
        mf_omp::tri_row_window(t, upper, range, n, from, to);
        for (std::ptrdiff_t k = from; k < to; ++k) y[k] = y[k] + alpha * slot[k];
    }
    std::free(buf);
    return true;
}

/* Threaded symmetric packed matvec. Disjoint output-row ranges; x gathered to
 * contiguous when strided. Returns true if handled. Beta already applied. */
__attribute__((noinline)) static bool mspmv_omp(
    bool upper, std::ptrdiff_t n, const TR *ap,
    const TR *x, std::ptrdiff_t incx, TR alpha, TR *y, std::ptrdiff_t incy)
{
    if (n < MSPMV_OMP_MIN || !blas_omp_should_thread())
        return false;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > MSPMV_MAX_CPUS) nthreads = MSPMV_MAX_CPUS;

    /* Unit-stride: ob-style column-partition axpydot (contiguous column read,
     * ~3.3x). */
    if (incx == 1 && incy == 1)
        return mspmv_axpydot(upper, n, ap, x, alpha, y, nthreads);

    /* Strided: gather x AND y to contiguous scratch, run the SIMD'd threaded
     * axpydot core (already the unit-stride winner), then scatter y back —
     * O(N) gather vs the O(N^2) anti-diagonal column-jumping rowgather, and the
     * contiguous core is SIMD where rowgather is scalar. Mirrors the serial
     * strided path but threaded. */
    TR *xs = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
    TR *ys = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
    if (!xs || !ys) { std::free(xs); std::free(ys); return false; }
    mf_kernels::gather_strided(n, x, incx, xs);
    mf_kernels::gather_strided(n, y, incy, ys);
    bool ok = mspmv_axpydot(upper, n, ap, xs, alpha, ys, nthreads);
    if (ok)
        mf_kernels::scatter_strided(n, y, incy, ys);
    std::free(xs);
    std::free(ys);
    return ok;
}
#endif

static void mspmv_core(
    char uplo,
    std::ptrdiff_t n,
    const TR *alpha_,
    const TR *ap,
    const TR *x, std::ptrdiff_t incx,
    const TR *beta_,
    TR *y, std::ptrdiff_t incy)
{
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = up(&uplo);

    if (n == 0 || (eq0(alpha) && eq1(beta))) return;

    mf_kernels::scale_y(n, beta, y, incy);
    if (eq0(alpha)) return;

#ifdef _OPENMP
    if (n >= MSPMV_OMP_MIN && blas_omp_available()
        && mspmv_omp(UPLO == 'U', n, ap, x, incx, alpha, y, incy))
        return;
#endif

    if (incx == 1 && incy == 1) {
        mspmv_contig(UPLO == 'U', n, ap, x, alpha, y);
        return;
    }

    /* Strided: gather x and the beta-scaled y into contiguous scratch, run the
     * SIMD contiguous core (y += alpha*A*x), scatter y back. O(N) gather/scatter
     * vs O(N*N) packed work; the in-place strided walk is the alloc-fail fallback. */
    {
        TR *xs = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
        TR *ys = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
        if (xs && ys) {
            mf_kernels::gather_strided(n, x, incx, xs);
            mf_kernels::gather_strided(n, y, incy, ys);
            mspmv_contig(UPLO == 'U', n, ap, xs, alpha, ys);
            mf_kernels::scatter_strided(n, y, incy, ys);
            std::free(xs); std::free(ys);
            return;
        }
        std::free(xs); std::free(ys);
    }

    {
        std::ptrdiff_t kk = 0;
        std::ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        std::ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        if (UPLO == 'U') {
            std::ptrdiff_t jx = kx, jy = ky;
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const TR t1 = alpha * x[jx];
                TR t2 = zero_dd;
                std::ptrdiff_t ix = kx, iy = ky;
                for (std::ptrdiff_t k = kk; k < kk + j; ++k) {
                    y[iy] = y[iy] + t1 * ap[k];
                    t2 = t2 + ap[k] * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] = y[jy] + t1 * ap[kk + j] + alpha * t2;
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            std::ptrdiff_t jx = kx, jy = ky;
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const TR t1 = alpha * x[jx];
                TR t2 = zero_dd;
                y[jy] = y[jy] + t1 * ap[kk];
                std::ptrdiff_t ix = jx, iy = jy;
                for (std::ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                    ix += incx; iy += incy;
                    y[iy] = y[iy] + t1 * ap[k];
                    t2 = t2 + ap[k] * x[ix];
                }
                y[jy] = y[jy] + alpha * t2;
                jx += incx; jy += incy;
                kk += n - j;
            }
        }
    }
}

extern "C" {
EPBLAS_FACADE_SPMV(mspmv, TR)
}
