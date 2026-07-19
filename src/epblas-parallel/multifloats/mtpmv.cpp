/* mtpmv — multifloats real DD triangular packed matrix-vector.
 *   x := A*x or A^T*x
 *
 * Serial — data dependencies in x.
 */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#include "mf_packed.h"     /* mf_packed::kk_upper / kk_lower — packed column offsets */
#ifdef _OPENMP
#include <cmath>
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#define MTPMV_OMP_MIN 128
#define MTPMV_MAX_CPUS 256
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::eq0;

using mf_util::up;  /* char flag uppercase — mf_util.h */
namespace {

using mf_packed::kk_upper;   /* packed column base offsets — see mf_packed.h */
using mf_packed::kk_lower;

/* In-place contiguous (incx==1) triangular packed matvec. Per column j, &ap[kk]
 * is a contiguous run, so this is the packed twin of mtrmv_contig: NoTrans is a
 * column AXPY scaled by x[j] (mf_kernels::axpy_add, order-free -> bit-exact); Trans
 * is a column dot against the off-diagonal x (mf_kernels::dot, vector accumulate +
 * hreduce -> within tolerance). The diagonal scaling matches the reference:
 * after the axpy for NoTrans, folded into the dot seed for Trans. The strided
 * entry gathers into contiguous scratch and reuses this. */
void mtpmv_serial_contig(bool upper, bool trans, bool nounit,
                         std::ptrdiff_t n, const TR *ap, TR *x)
{
    if (!trans) {
        if (upper) {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const TR *aj = &ap[kk_upper(j)];
                const TR tmp = x[j];
                if (!eq0(tmp)) mf_kernels::axpy_add(j, &x[0], &aj[0], tmp);
                if (nounit) x[j] = x[j] * aj[j];
            }
        } else {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                const TR *aj = &ap[kk_lower(j, n)];
                const TR tmp = x[j];
                if (!eq0(tmp)) mf_kernels::axpy_add(n - 1 - j, &x[j + 1], &aj[1], tmp);
                if (nounit) x[j] = x[j] * aj[0];
            }
        }
    } else {
        if (upper) {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                const TR *aj = &ap[kk_upper(j)];
                TR tmp = nounit ? (aj[j] * x[j]) : x[j];
                x[j] = tmp + mf_kernels::dot(j, &aj[0], &x[0]);
            }
        } else {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const TR *aj = &ap[kk_lower(j, n)];
                TR tmp = nounit ? (aj[0] * x[j]) : x[j];
                x[j] = tmp + mf_kernels::dot(n - 1 - j, &aj[1], &x[j + 1]);
            }
        }
    }
}
}

#ifdef _OPENMP
/* Threaded contiguous-x core, mirroring mtrmv_omp_contig but with packed column
 * offsets. The earlier row-gather threaded poorly: NoTrans walked a column-
 * JUMPING run (offset += c+1 per step) — cache-hostile — and the contiguous-
 * block row partition load-imbalanced the triangular work. This keeps packed-
 * column access contiguous:
 *   - Trans: each x[j] is an independent contiguous-column dot (disjoint writes),
 *     schedule(static,1) cyclic balancing.
 *   - NoTrans: area-balanced contiguous COLUMN partition (packed twin of the
 *     mspmv axpydot) + per-thread slot + BOUNDED reduction, escaping the old
 *     cyclic O(nthreads*n) full fold that floored par4/par1 at ~0.47.
 * DD addition reorders vs serial → within fuzz tol; serial stays bit-exact.
 * Returns true on success, false if a scratch alloc failed. */
static bool mtpmv_omp_contig(bool upper, bool trans, bool nounit,
                             std::ptrdiff_t n, const TR *ap, TR *x, std::ptrdiff_t nthreads)
{
    if (trans) {
        TR *y_buf = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
        if (!y_buf) return false;
        #pragma omp parallel num_threads(nthreads)
        {
            #pragma omp for schedule(static, 1)
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const TR *aj = upper ? &ap[kk_upper(j)] : &ap[kk_lower(j, n)];
                if (upper) {
                    TR temp = nounit ? (aj[j] * x[j]) : x[j];
                    y_buf[j] = temp + mf_kernels::dot(j, &aj[0], &x[0]);
                } else {
                    TR temp = nounit ? (aj[0] * x[j]) : x[j];
                    y_buf[j] = temp + mf_kernels::dot(n - 1 - j, &aj[1], &x[j + 1]);
                }
            }
            #pragma omp for schedule(static)
            for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = y_buf[i];
        }
        std::free(y_buf);
        return true;
    } else {
        using mf_pred::one_dd;   /* shared DD constant — mf_pred.h */
        std::ptrdiff_t range[MTPMV_MAX_CPUS + 1];
        /* per-column work ~j (upper) / ~(n-j) (lower) -> heavy_high=upper. */
        std::ptrdiff_t ncpu = mf_omp::tri_area_bounds(n, nthreads, 3, 4, upper,
                                           MTPMV_MAX_CPUS, range);
        if (ncpu <= 1) return false;
        TR *buf = static_cast<TR *>(std::calloc((std::size_t)ncpu * n, sizeof(TR)));
        if (!buf) return false;
        /* Each thread folds its disjoint column range's AXPY into a private slot,
         * reading the ORIGINAL x (x is overwritten only in the reduction). */
        #pragma omp parallel for schedule(static, 1) num_threads(ncpu)
        for (std::ptrdiff_t t = 0; t < ncpu; ++t)
        {
            std::ptrdiff_t c_from = range[t], c_to = range[t + 1];
            TR *slot = buf + (std::size_t)t * n;
            if (upper) {
                for (std::ptrdiff_t j = c_from; j < c_to; ++j) {
                    const TR xj = x[j];
                    const TR *aj = &ap[kk_upper(j)];
                    if (!eq0(xj)) mf_kernels::axpy_add(j, &slot[0], &aj[0], xj);
                    slot[j] = slot[j] + xj * (nounit ? aj[j] : one_dd);
                }
            } else {
                for (std::ptrdiff_t j = c_from; j < c_to; ++j) {
                    const TR xj = x[j];
                    const TR *aj = &ap[kk_lower(j, n)];
                    slot[j] = slot[j] + xj * (nounit ? aj[0] : one_dd);
                    if (!eq0(xj)) mf_kernels::axpy_add(n - 1 - j, &slot[j + 1], &aj[1], xj);
                }
            }
        }
        /* Bounded reduction: x aliases the input, so sum the other slots' row
         * windows into the widest slot (last for upper / first for lower, which
         * spans all of [0,n)) and then overwrite x in one pass. */
        TR *target = buf + (std::size_t)(upper ? ncpu - 1 : 0) * n;
        for (std::ptrdiff_t i = upper ? 0 : 1; i < (upper ? ncpu - 1 : ncpu); ++i) {
            const TR *src = buf + (std::size_t)i * n;
            std::ptrdiff_t from, to;
            mf_omp::tri_row_window(i, upper, range, n, from, to);
            for (std::ptrdiff_t k = from; k < to; ++k) target[k] = target[k] + src[k];
        }
        for (std::ptrdiff_t i = 0; i < n; ++i) x[i] = target[i];
        std::free(buf);
        return true;
    }
}

/* Threaded in-place triangular packed matvec. incx==1 drives the contiguous
 * core directly; strided gathers/scatters around it. Returns true if handled. */
__attribute__((noinline)) static bool mtpmv_omp(
    bool upper, bool trans, bool nounit, std::ptrdiff_t n,
    const TR *ap, TR *x, std::ptrdiff_t incx)
{
    if (n < MTPMV_OMP_MIN || !blas_omp_should_thread())
        return false;
    std::ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > MTPMV_MAX_CPUS) nthreads = MTPMV_MAX_CPUS;

    if (incx == 1)
        return mtpmv_omp_contig(upper, trans, nounit, n, ap, x, nthreads);

    TR *xbase = (incx < 0) ? x - (std::ptrdiff_t)(n - 1) * incx : x;
    TR *xbuf = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR)));
    if (!xbuf) return false;
    for (std::ptrdiff_t i = 0; i < n; ++i) xbuf[i] = xbase[(std::ptrdiff_t)i * incx];
    bool ok = mtpmv_omp_contig(upper, trans, nounit, n, ap, xbuf, nthreads);
    if (ok)
        for (std::ptrdiff_t i = 0; i < n; ++i) xbase[(std::ptrdiff_t)i * incx] = xbuf[i];
    std::free(xbuf);
    return ok;
}
#endif

static void mtpmv_core(
    char uplo, char trans, char diag,
    std::ptrdiff_t n,
    const TR *ap,
    TR *x, std::ptrdiff_t incx)
{
    const char UPLO = up(&uplo);
    char TRANS = up(&trans);
    if (TRANS == 'C') TRANS = 'T';
    const bool nounit = (up(&diag) != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    if (n >= MTPMV_OMP_MIN && blas_omp_available()
        && mtpmv_omp(UPLO == 'U', TRANS != 'N', nounit != 0, n, ap, x, incx))
        return;
#endif

    const bool is_upper = (UPLO == 'U');
    const bool is_trans = (TRANS != 'N');

    if (incx == 1) {
        mtpmv_serial_contig(is_upper, is_trans, nounit != 0, n, ap, x);
        return;
    }

    /* Strided: gather x into contiguous scratch, run the stride-1 core (which
     * beats the ob clone's indexed strided walk), scatter back. O(N)
     * gather/scatter against O(N^2) work, bit-identical column order. Stack
     * scratch for the common small-N case dodges malloc latency; spill to the
     * heap past it. Falls back to the direct strided walk only if the heap
     * alloc fails. */
    {
        const std::ptrdiff_t sx = incx;
        const std::ptrdiff_t kx = (sx < 0) ? -(n - 1) * sx : 0;
        TR stackbuf[512];
        TR *heap = NULL;
        TR *xc = (n <= 512) ? stackbuf
                           : (heap = static_cast<TR *>(std::malloc((std::size_t)n * sizeof(TR))));
        if (xc) {
            std::ptrdiff_t ix = kx;
            for (std::ptrdiff_t k = 0; k < n; ++k) { xc[k] = x[ix]; ix += sx; }
            mtpmv_serial_contig(is_upper, is_trans, nounit != 0, n, ap, xc);
            ix = kx;
            for (std::ptrdiff_t k = 0; k < n; ++k) { x[ix] = xc[k]; ix += sx; }
            std::free(heap);
            return;
        }
        std::free(heap);
    }

    {
        /* Direct strided fallback (heap alloc failed). ptrdiff_t indices so the
         * packed (ap[k]) and strided x (x[ix]) walks need no per-element sign
         * extension. */
        const std::ptrdiff_t sx = incx;
        std::ptrdiff_t kx = (sx < 0) ? -(n - 1) * sx : 0;
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                std::ptrdiff_t kk = 0;
                std::ptrdiff_t jx = kx;
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    if (!eq0(x[jx])) {
                        const TR tmp = x[jx];
                        std::ptrdiff_t ix = kx;
                        for (std::ptrdiff_t k = kk; k < kk + j; ++k) {
                            x[ix] = x[ix] + tmp * ap[k];
                            ix += sx;
                        }
                        if (nounit) x[jx] = x[jx] * ap[kk + j];
                    }
                    jx += sx;
                    kk += j + 1;
                }
            } else {
                std::ptrdiff_t kk = n * (n + 1) / 2 - 1;
                kx += (n - 1) * sx;
                std::ptrdiff_t jx = kx;
                for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (!eq0(x[jx])) {
                        const TR tmp = x[jx];
                        std::ptrdiff_t ix = kx;
                        for (std::ptrdiff_t k = kk; k > kk - (n - 1 - j); --k) {
                            x[ix] = x[ix] + tmp * ap[k];
                            ix -= sx;
                        }
                        if (nounit) x[jx] = x[jx] * ap[kk - (n - 1 - j)];
                    }
                    jx -= sx;
                    kk -= (n - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                std::ptrdiff_t kk = n * (n + 1) / 2 - 1;
                std::ptrdiff_t jx = kx + (n - 1) * sx;
                for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR tmp = x[jx];
                    std::ptrdiff_t ix = jx;
                    if (nounit) tmp = tmp * ap[kk];
                    for (std::ptrdiff_t k = kk - 1; k >= kk - j; --k) {
                        ix -= sx;
                        tmp = tmp + ap[k] * x[ix];
                    }
                    x[jx] = tmp;
                    jx -= sx;
                    kk -= j + 1;
                }
            } else {
                std::ptrdiff_t kk = 0;
                std::ptrdiff_t jx = kx;
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    TR tmp = x[jx];
                    std::ptrdiff_t ix = jx;
                    if (nounit) tmp = tmp * ap[kk];
                    for (std::ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                        ix += sx;
                        tmp = tmp + ap[k] * x[ix];
                    }
                    x[jx] = tmp;
                    jx += sx;
                    kk += n - j;
                }
            }
        }
    }
}

extern "C" {
EPBLAS_FACADE_TPMV(mtpmv, TR)
}
