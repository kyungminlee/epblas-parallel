/*
 * ytbsv — kind10 complex triangular band solve.
 *   x := inv(op(A)) * x   (op = A, A^T, or A^H), A triangular band, K+1 diags.
 *
 * Serial — back/forward substitution; O(N*K) with a K-deep loop-carried
 * recurrence (OpenBLAS does not thread it either), so there is no parallel
 * path. The NoTrans paths (contiguous and strided) keep the hand-tuned A_-macro
 * form, which beats both ob and netlib here (complex strided NoTrans: par/ob
 * ~0.92-0.94, par/mig ~0.98-1.01). Only the Trans/Conj CONTIGUOUS leaves carry
 * the col-base-pointer hoist + (j>K) compare-only window-start that ob uses:
 * the old A_(L+i,j) form re-derived (size_t)j*lda at every band access and split
 * the conj branch into two loops, leaving those cells ~3-5% behind ob. Hoisting
 * col=&a[j*lda] once per column (so col[L+i] rides one band-diagonal induction)
 * and folding the conj into CONJIF brings UTN/UTU/LTU/LCU back to parity without
 * disturbing the fast NoTrans or the (already-passing) strided Trans paths.
 * The Lower-Trans contiguous leaf is additionally phase-split (final triangle
 * + steady fixed-k band, pure pointer walks) in a noinline helper: the band
 * dot is only ~k iterations, so the generic column loop's min(j+k+1,n) clamp
 * and per-column base-pointer rebuild never amortize (LTN trailed ob ~1.02
 * with a byte-identical dot loop; same fix as the qtbsv contiguous arms).
 */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"

typedef _Complex long double TC;
static inline TC cconj(TC z) { return ~z; }


#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define CONJIF(z) (noconj ? (z) : cconj(z))

/* Contiguous Lower Trans/ConjTrans solve, phase-split: final triangle
 * (j > n-k-1, shrinking dot off the column base) then steady band (fixed
 * k-length dot), both pure pointer walks.  Descending i order kept =>
 * bit-exact.  noinline keeps this leaf's layout independent of the
 * dispatch body. */
__attribute__((noinline, noclone))
static void ytbsv_trans_lower_contig(
    ptrdiff_t n, ptrdiff_t k, bool noconj, bool nounit,
    const TC *restrict a, ptrdiff_t lda, TC *restrict x)
{
    const ptrdiff_t jt = n - k - 1;              /* j <= jt: full k-length dot */
    const ptrdiff_t tri_lo = (jt + 1 > 0) ? (jt + 1) : 0;
    const TC *col = a + (size_t)(n - 1) * lda;   /* col[t] = A_(t, j); col[0] = diag */
    ptrdiff_t j = n - 1;
    for (; j >= tri_lo; --j, col -= lda) {
        const ptrdiff_t len = n - 1 - j;
        TC tmp = x[j];
        const TC *xj = x + j;
        for (ptrdiff_t t = len; t >= 1; --t) tmp -= CONJIF(col[t]) * xj[t];
        if (nounit) tmp /= CONJIF(col[0]);
        x[j] = tmp;
    }
    for (; j >= 0; --j, col -= lda) {
        TC tmp = x[j];
        const TC *xj = x + j;
        for (ptrdiff_t t = k; t >= 1; --t) tmp -= CONJIF(col[t]) * xj[t];
        if (nounit) tmp /= CONJIF(col[0]);
        x[j] = tmp;
    }
}

/* Strided Lower Trans/ConjTrans solve, body verbatim from the dispatch
 * loop.  noinline quarantines its layout: inline it sat downstream of the
 * contiguous arm and lost ~2-4% (LTU strided) whenever that code moved. */
__attribute__((noinline, noclone))
static void ytbsv_trans_lower_strided(
    ptrdiff_t n, ptrdiff_t k, bool noconj, bool nounit,
    const TC *restrict a, ptrdiff_t lda,
    TC *restrict x, ptrdiff_t incx, ptrdiff_t kx)
{
    kx += (n - 1) * incx;
    ptrdiff_t jx = kx;
    for (ptrdiff_t j = n - 1; j >= 0; --j) {
        TC tmp = x[jx];
        ptrdiff_t ix = kx;
        const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
        for (ptrdiff_t i = i_hi - 1; i > j; --i) {
            const TC aij = noconj ? A_(i - j, j) : cconj(A_(i - j, j));
            tmp -= aij * x[ix];
            ix -= incx;
        }
        if (nounit) tmp /= (noconj ? A_(0, j) : cconj(A_(0, j)));
        x[jx] = tmp;
        jx -= incx;
        if ((n - 1 - j) >= k) kx -= incx;
    }
}

static void ytbsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n, ptrdiff_t k,
    const TC *restrict a, ptrdiff_t lda,
    TC *restrict x, ptrdiff_t incx)
{
    const TC zero = 0.0L + 0.0Li;
    const char UPLO = blas_up(uplo);
    const char TRANS = blas_up(trans);
    const bool noconj = (TRANS == 'T');
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

    if (incx == 1) {
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const ptrdiff_t L = k - j;
                        if (nounit) x[j] /= A_(k, j);
                        const TC tmp = x[j];
                        const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i) x[i] -= tmp * A_(L + i, j);
                    }
                }
            } else {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= A_(0, j);
                        const TC tmp = x[j];
                        const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                        for (ptrdiff_t i = j + 1; i < i_hi; ++i) x[i] -= tmp * A_(i - j, j);
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const TC *restrict col = &a[(size_t)j * lda];
                    const ptrdiff_t L = k - j;
                    const ptrdiff_t i_lo = (j > k) ? (j - k) : 0;
                    TC tmp = x[j];
                    for (ptrdiff_t i = i_lo; i < j; ++i) tmp -= CONJIF(col[L + i]) * x[i];
                    if (nounit) tmp /= CONJIF(col[k]);
                    x[j] = tmp;
                }
            } else {
                ytbsv_trans_lower_contig(n, k, noconj, nounit, a, lda, x);
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    kx -= incx;
                    if (x[jx] != zero) {
                        ptrdiff_t ix = kx;
                        const ptrdiff_t L = k - j;
                        if (nounit) x[jx] /= A_(k, j);
                        const TC tmp = x[jx];
                        const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i) {
                            x[ix] -= tmp * A_(L + i, j);
                            ix -= incx;
                        }
                    }
                    jx -= incx;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    kx += incx;
                    if (x[jx] != zero) {
                        ptrdiff_t ix = kx;
                        if (nounit) x[jx] /= A_(0, j);
                        const TC tmp = x[jx];
                        const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                        for (ptrdiff_t i = j + 1; i < i_hi; ++i) {
                            x[ix] -= tmp * A_(i - j, j);
                            ix += incx;
                        }
                    }
                    jx += incx;
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const ptrdiff_t L = k - j;
                    const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                    for (ptrdiff_t i = i_lo; i < j; ++i) {
                        const TC aij = noconj ? A_(L + i, j) : cconj(A_(L + i, j));
                        tmp -= aij * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= (noconj ? A_(k, j) : cconj(A_(k, j)));
                    x[jx] = tmp;
                    jx += incx;
                    if (j >= k) kx += incx;
                }
            } else {
                ytbsv_trans_lower_strided(n, k, noconj, nounit, a, lda, x, incx, kx);
            }
        }
    }
}

EPBLAS_FACADE_TBMV(ytbsv, TC)

#undef CONJIF
#undef A_
