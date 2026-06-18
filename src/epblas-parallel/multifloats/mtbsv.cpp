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
#include "mf_simd_dd.h"   /* faithful SoA dd_mul/dd_sub/load_dd4/store_dd4 */

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

/* xp[t] -= t_scalar * cp[t], t = 0..len-1 — the NoTrans band update. Each row
 * is an independent write (temp is the finalized x[j]), so the run is a plain
 * DD axpy: 4-wide SoA (bit-identical to the scalar operators) with a scalar
 * tail. The Trans path is a per-column dot reduction, vectorized below in
 * band_dot (within tolerance, not bit-exact, since the reduce reorders). */
#ifdef MBLAS_SIMD_DD
inline void band_axpy_sub(std::ptrdiff_t len, T *xp, const T *cp, const T &t) {
    const __m256d thb = _mm256_set1_pd(t.limbs[0]);
    const __m256d tlb = _mm256_set1_pd(t.limbs[1]);
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256d mh, ml; mf_simd::load_dd4(cp + i, mh, ml);
        __m256d ph, pl; mf_simd::dd_mul(mh, ml, thb, tlb, ph, pl);   /* t*cp */
        __m256d xh, xl; mf_simd::load_dd4(xp + i, xh, xl);
        __m256d rh, rl; mf_simd::dd_sub(xh, xl, ph, pl, rh, rl);     /* x - t*cp */
        mf_simd::store_dd4(xp + i, rh, rl);
    }
    for (; i < len; ++i) xp[i] -= t * cp[i];
}
#else
inline void band_axpy_sub(std::ptrdiff_t len, T *xp, const T *cp, const T &t) {
    for (std::ptrdiff_t i = 0; i < len; ++i) xp[i] -= t * cp[i];
}
#endif

/* sum_{t=0}^{len-1} cp[t]*xp[t] — the band dot the Trans solve subtracts from
 * x[j]. The reduction is loop-carried within a column but every column's dot
 * reads only already-solved x (rows below j for upper, above j for lower), so
 * the dot itself is a plain reduction: 4-wide vector accumulator + one faithful
 * hreduce, with a scalar tail. The earlier attempt buffered the 4 products and
 * reduced them in scalar (keeping the full scalar fold + paying store/reload) —
 * that regressed; accumulating into the vector and reducing once does not. The
 * reduce reorders vs the scalar left-fold, so this matches the reference within
 * the consistency tolerance (the cross-column recurrence stays scalar/exact). */
#ifdef MBLAS_SIMD_DD
inline T band_dot(std::ptrdiff_t len, const T *cp, const T *xp) {
    __m256d sh = _mm256_setzero_pd(), sl = _mm256_setzero_pd();
    std::ptrdiff_t i = 0;
    for (; i + 4 <= len; i += 4) {
        __m256d mh, ml; mf_simd::load_dd4(cp + i, mh, ml);
        __m256d xh, xl; mf_simd::load_dd4(xp + i, xh, xl);
        __m256d ph, pl; mf_simd::dd_mul(mh, ml, xh, xl, ph, pl);
        mf_simd::dd_add(sh, sl, ph, pl, sh, sl);
    }
    T s = (i >= 4) ? mf_simd::hreduce(sh, sl) : T{0.0, 0.0};
    for (; i < len; ++i) s = s + cp[i] * xp[i];
    return s;
}
#else
inline T band_dot(std::ptrdiff_t len, const T *cp, const T *xp) {
    T s{0.0, 0.0};
    for (std::ptrdiff_t i = 0; i < len; ++i) s = s + cp[i] * xp[i];
    return s;
}
#endif

/* Contiguous (unit-stride) triangular band solve, x[0..n-1] in logical order. */
void mtbsv_contig(int upper, int trans_, int nounit,
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
                    band_axpy_sub(j - i_lo, &x[i_lo], &col[off + i_lo], temp);
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
                    band_axpy_sub(i_hi - j, &x[j + 1], &col[off + j + 1], temp);
                }
            }
        }
    } else {
        if (upper) {
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                const T *col = &a[static_cast<std::size_t>(j) * lda];
                const std::ptrdiff_t off = k - j;
                const std::ptrdiff_t i_lo = (j > k) ? j - k : 0;
                T temp = x[j] - band_dot(j - i_lo, &col[off + i_lo], &x[i_lo]);
                if (nounit) temp /= col[k];
                x[j] = temp;
            }
        } else {
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                const T *col = &a[static_cast<std::size_t>(j) * lda];
                const std::ptrdiff_t off = -j;
                const std::ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                T temp = x[j] - band_dot(i_hi - j, &col[off + j + 1], &x[j + 1]);
                if (nounit) temp /= col[0];
                x[j] = temp;
            }
        }
    }
}

/* In-place strided solve — faithful OpenBLAS reference, used only when the
 * gather scratch cannot be allocated. */
void mtbsv_strided(int upper, int trans_, int nounit,
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
    const char trc   = up(trans);
    const int trans_ = (trc == 'T' || trc == 'C') ? 1 : 0;
    const int nounit = (up(diag) == 'N');

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
