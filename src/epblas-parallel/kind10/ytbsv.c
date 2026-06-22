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
 * LTN contiguous still trails ob ~1.022: the hot dot loop is byte-identical to
 * ob (fldt=6/fmul=4/fsub=3, both 16-aligned) and par beats netlib there
 * (par/mig ~0.97), so the residual is a whole-function layout artifact, not a
 * loop-body deficit -- the same alignment floor the real twin etbsv sits at.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"

typedef _Complex long double T;
static inline T cconj(T z) { return ~z; }


#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define CONJIF(z) (noconj ? (z) : cconj(z))

static void ytbsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx)
{
    const T zero = 0.0L + 0.0Li;
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
                        const T tmp = x[j];
                        const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i) x[i] -= tmp * A_(L + i, j);
                    }
                }
            } else {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= A_(0, j);
                        const T tmp = x[j];
                        const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                        for (ptrdiff_t i = j + 1; i < i_hi; ++i) x[i] -= tmp * A_(i - j, j);
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const T *restrict col = &a[(size_t)j * lda];
                    const ptrdiff_t L = k - j;
                    const ptrdiff_t i_lo = (j > k) ? (j - k) : 0;
                    T tmp = x[j];
                    for (ptrdiff_t i = i_lo; i < j; ++i) tmp -= CONJIF(col[L + i]) * x[i];
                    if (nounit) tmp /= CONJIF(col[k]);
                    x[j] = tmp;
                }
            } else {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    const T *restrict col = &a[(size_t)j * lda];
                    const ptrdiff_t off = -j;
                    const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                    T tmp = x[j];
                    for (ptrdiff_t i = i_hi - 1; i > j; --i) tmp -= CONJIF(col[off + i]) * x[i];
                    if (nounit) tmp /= CONJIF(col[0]);
                    x[j] = tmp;
                }
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
                        const T tmp = x[jx];
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
                        const T tmp = x[jx];
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
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const ptrdiff_t L = k - j;
                    const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                    for (ptrdiff_t i = i_lo; i < j; ++i) {
                        const T aij = noconj ? A_(L + i, j) : cconj(A_(L + i, j));
                        tmp -= aij * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= (noconj ? A_(k, j) : cconj(A_(k, j)));
                    x[jx] = tmp;
                    jx += incx;
                    if (j >= k) kx += incx;
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                    for (ptrdiff_t i = i_hi - 1; i > j; --i) {
                        const T aij = noconj ? A_(i - j, j) : cconj(A_(i - j, j));
                        tmp -= aij * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= (noconj ? A_(0, j) : cconj(A_(0, j)));
                    x[jx] = tmp;
                    jx -= incx;
                    if ((n - 1 - j) >= k) kx -= incx;
                }
            }
        }
    }
}

EPBLAS_FACADE_TBMV(ytbsv, T)

#undef CONJIF
#undef A_
