/* mtbsv — multifloats real DD triangular band solve.
 *   x := inv(A)*x or inv(A^T)*x, A triangular band with K+1 diagonals.
 *
 * Serial — back/forward substitution. tbsv is O(N*K) with a K-deep
 * loop-carried recurrence (OpenBLAS does not thread it either), so there is
 * no parallel path.
 *
 * The contiguous (incx==1) core mtbsv_contig() is a faithful port of the
 * OpenBLAS reference (trans -> uplo nesting, col-base pointer hoisted once per
 * column); identical source + flags -> identical codegen -> par/ob parity.
 *
 * A strided x is *not* solved in place: the reference's strided leaves drift
 * ~4% behind ob on byte-identical source (codegen jitter on the kx/jx/ix
 * stepping), so instead we linearize x into a contiguous scratch, run the
 * parity-winning contiguous core, and scatter back. The gather/scatter is 2N
 * copies against O(N*K) band work, so even for a thin band it is repaid by the
 * core no longer paying the strided-access penalty. The in-place strided code
 * is kept only as the (essentially unreachable) scratch-alloc-failure fallback.
 */

#include <cstddef>
#include <cctype>
#include <cstdlib>
#include <multifloats.h>
#include "mf_util.h"
#include "mf_kernels.h"
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using T = mf::float64x2;


using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {

/* NoTrans band update is mf_kernels::axpy_sub (xp -= temp*cp), an order-free DD
 * axpy -> bit-exact. The Trans per-column band dot is mf_kernels::dot (vector
 * accumulate + hreduce) -> within tolerance, since the reduce reorders; the
 * cross-column recurrence stays scalar/exact in the caller below. */

/* Contiguous (unit-stride) triangular band solve, x[0..n-1] in logical order. */
void mtbsv_contig(bool upper, std::ptrdiff_t trans_, bool nounit,
                  std::ptrdiff_t n, std::ptrdiff_t k,
                  const T *a, std::ptrdiff_t lda, T *x)
{
    if (!trans_) {
        if (upper) {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                if (x[j] != 0.0) {
                    const T *col = &a[static_cast<std::size_t>(j) * lda];
                    const std::ptrdiff_t off = k - j;
                    if (nounit) x[j] /= col[k];
                    const T temp = x[j];
                    const std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    mf_kernels::axpy_sub(j - i_lo, &x[i_lo], &col[off + i_lo], temp);
                }
            }
        } else {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                if (x[j] != 0.0) {
                    const T *col = &a[static_cast<std::size_t>(j) * lda];
                    const std::ptrdiff_t off = -j;
                    if (nounit) x[j] /= col[0];
                    const T temp = x[j];
                    const std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    mf_kernels::axpy_sub(i_hi - j, &x[j + 1], &col[off + j + 1], temp);
                }
            }
        }
    } else {
        if (upper) {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const T *col = &a[static_cast<std::size_t>(j) * lda];
                const std::ptrdiff_t off = k - j;
                const std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                T temp = x[j] - mf_kernels::dot(j - i_lo, &col[off + i_lo], &x[i_lo]);
                if (nounit) temp /= col[k];
                x[j] = temp;
            }
        } else {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                const T *col = &a[static_cast<std::size_t>(j) * lda];
                const std::ptrdiff_t off = -j;
                const std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                T temp = x[j] - mf_kernels::dot(i_hi - j, &col[off + j + 1], &x[j + 1]);
                if (nounit) temp /= col[0];
                x[j] = temp;
            }
        }
    }
}

/* In-place strided solve — faithful OpenBLAS reference, used only when the
 * gather scratch cannot be allocated. */
void mtbsv_strided(bool upper, std::ptrdiff_t trans_, bool nounit,
                   std::ptrdiff_t n, std::ptrdiff_t k,
                   const T *a, std::ptrdiff_t lda, T *x, std::ptrdiff_t incx)
{
    std::ptrdiff_t kx = (incx <= 0) ? -(n - 1) * incx : 0;

    if (!trans_) {
        if (upper) {
            kx += (n - 1) * incx;
            std::ptrdiff_t jx = kx;
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                kx -= incx;
                if (x[jx] != 0.0) {
                    std::ptrdiff_t ix = kx;
                    const T *col = &a[static_cast<std::size_t>(j) * lda];
                    const std::ptrdiff_t off = k - j;
                    if (nounit) x[jx] /= col[k];
                    const T temp = x[jx];
                    const std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (std::ptrdiff_t i = j - 1; i >= i_lo; --i) {
                        x[ix] -= temp * col[off + i];
                        ix -= incx;
                    }
                }
                jx -= incx;
            }
        } else {
            std::ptrdiff_t jx = kx;
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                kx += incx;
                if (x[jx] != 0.0) {
                    std::ptrdiff_t ix = kx;
                    const T *col = &a[static_cast<std::size_t>(j) * lda];
                    const std::ptrdiff_t off = -j;
                    if (nounit) x[jx] /= col[0];
                    const T temp = x[jx];
                    const std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (std::ptrdiff_t i = j + 1; i <= i_hi; ++i) {
                        x[ix] -= temp * col[off + i];
                        ix += incx;
                    }
                }
                jx += incx;
            }
        }
    } else {
        if (upper) {
            std::ptrdiff_t jx = kx;
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                T temp = x[jx];
                std::ptrdiff_t ix = kx;
                const T *col = &a[static_cast<std::size_t>(j) * lda];
                const std::ptrdiff_t off = k - j;
                const std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                for (std::ptrdiff_t i = i_lo; i < j; ++i) {
                    temp -= col[off + i] * x[ix];
                    ix += incx;
                }
                if (nounit) temp /= col[k];
                x[jx] = temp;
                jx += incx;
                if (j >= k) kx += incx;
            }
        } else {
            kx += (n - 1) * incx;
            std::ptrdiff_t jx = kx;
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                T temp = x[jx];
                std::ptrdiff_t ix = kx;
                const T *col = &a[static_cast<std::size_t>(j) * lda];
                const std::ptrdiff_t off = -j;
                const std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                for (std::ptrdiff_t i = i_hi; i > j; --i) {
                    temp -= col[off + i] * x[ix];
                    ix -= incx;
                }
                if (nounit) temp /= col[0];
                x[jx] = temp;
                jx -= incx;
                if (n - 1 - j >= k) kx -= incx;
            }
        }
    }
}
}  // namespace

static void mtbsv_core(
    char uplo, char trans, char diag,
    std::ptrdiff_t n, std::ptrdiff_t k,
    const T *a, std::ptrdiff_t lda,
    T *x, std::ptrdiff_t incx)
{

    if (n == 0) return;

    const bool upper  = (up(&uplo) == 'U');
    const char TR   = up(&trans);
    const std::ptrdiff_t trans_ = (TR == 'T' || TR == 'C') ? 1 : 0;
    const bool nounit = (up(&diag) == 'N');

    if (incx == 1) {
        mtbsv_contig(upper, trans_, nounit, n, k, a, lda, x);
        return;
    }

    /* Strided x: linearize to a contiguous scratch in logical order, solve on
     * the parity-winning contiguous core, scatter back. */
    const std::ptrdiff_t base = (incx <= 0) ? -(n - 1) * incx : 0;
    T *xs = static_cast<T *>(std::malloc(static_cast<std::size_t>(n) * sizeof(T)));
    if (xs) {
        for (std::ptrdiff_t i = 0; i < n; ++i) xs[i] = x[base + i * incx];
        mtbsv_contig(upper, trans_, nounit, n, k, a, lda, xs);
        for (std::ptrdiff_t i = 0; i < n; ++i) x[base + i * incx] = xs[i];
        std::free(xs);
        return;
    }

    mtbsv_strided(upper, trans_, nounit, n, k, a, lda, x, incx);  /* alloc-failure fallback */
}

extern "C" {
EPBLAS_FACADE_TBMV(mtbsv, T)
}
