/* mtbsv — multifloats real DD triangular band solve.
 *   x := inv(A)*x or inv(A^T)*x, A triangular band with K+1 diagonals.
 *
 * Serial — back/forward substitution. tbsv is O(N*K) with a K-deep
 * loop-carried recurrence (OpenBLAS does not thread it either), so there is
 * no parallel path and no reason for this overlay to diverge from the
 * faithful OpenBLAS port: the body below mirrors src/epblas-openblas's mtbsv
 * structure exactly (trans -> uplo -> incx nesting, contiguous and strided
 * as separate leaves, col-base pointer hoisted once per column, running kx
 * for the strided index). Identical source + identical flags (-O3
 * -ffp-contract=on, -march=native) -> identical codegen -> par/ob parity.
 * An earlier incx-first structure grouped all four shapes under one incx
 * branch, which coupled the contiguous and strided register allocation and
 * left the strided unit-diag NoTrans cells ~2-5% behind ob.
 */

#include <cstddef>
#include <cctype>
#include <multifloats.h>

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}
}

extern "C" void mtbsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_, const int *k_,
    const T *a, const int *lda_,
    T *x, const int *incx_,
    std::size_t uplo_len, std::size_t trans_len, std::size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const std::ptrdiff_t n    = static_cast<std::ptrdiff_t>(*n_);
    const std::ptrdiff_t k    = static_cast<std::ptrdiff_t>(*k_);
    const std::ptrdiff_t lda  = static_cast<std::ptrdiff_t>(*lda_);
    const std::ptrdiff_t incx = static_cast<std::ptrdiff_t>(*incx_);

    if (n == 0) return;

    const int upper  = (up(uplo) == 'U');
    char trc         = up(trans);
    const int trans_ = (trc == 'T' || trc == 'C') ? 1 : 0;
    const int nounit = (up(diag) == 'N');

    std::ptrdiff_t kx = (incx <= 0) ? -(n - 1) * incx : 0;

    if (!trans_) {
        if (upper) {
            if (incx == 1) {
                for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0) {
                        const T *col = &a[static_cast<std::size_t>(j) * lda];
                        const std::ptrdiff_t off = k - j;
                        if (nounit) x[j] /= col[k];
                        const T temp = x[j];
                        const std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        for (std::ptrdiff_t i = j - 1; i >= i_lo; --i)
                            x[i] -= temp * col[off + i];
                    }
                }
            } else {
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
            }
        } else {
            if (incx == 1) {
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0) {
                        const T *col = &a[static_cast<std::size_t>(j) * lda];
                        const std::ptrdiff_t off = -j;
                        if (nounit) x[j] /= col[0];
                        const T temp = x[j];
                        const std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        for (std::ptrdiff_t i = j + 1; i <= i_hi; ++i)
                            x[i] -= temp * col[off + i];
                    }
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
        }
    } else {
        if (upper) {
            if (incx == 1) {
                for (std::ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[j];
                    const T *col = &a[static_cast<std::size_t>(j) * lda];
                    const std::ptrdiff_t off = k - j;
                    const std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (std::ptrdiff_t i = i_lo; i < j; ++i)
                        temp -= col[off + i] * x[i];
                    if (nounit) temp /= col[k];
                    x[j] = temp;
                }
            } else {
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
            }
        } else {
            if (incx == 1) {
                for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[j];
                    const T *col = &a[static_cast<std::size_t>(j) * lda];
                    const std::ptrdiff_t off = -j;
                    const std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (std::ptrdiff_t i = i_hi; i > j; --i)
                        temp -= col[off + i] * x[i];
                    if (nounit) temp /= col[0];
                    x[j] = temp;
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
}
