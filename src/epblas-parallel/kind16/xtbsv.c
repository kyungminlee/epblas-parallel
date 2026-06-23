/*
 * xtbsv — kind16 complex triangular band solve.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#include "../common/epblas_facade.h"

typedef __complex128 TC;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xtbsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n, ptrdiff_t k,
    const TC *restrict a, ptrdiff_t lda,
    TC *restrict x, ptrdiff_t incx)
{
    const TC zero = 0.0Q + 0.0Qi;
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
                    TC tmp = x[j];
                    const ptrdiff_t L = k - j;
                    const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                    if (noconj) for (ptrdiff_t i = i_lo; i < j; ++i) tmp -= A_(L + i, j) * x[i];
                    else        for (ptrdiff_t i = i_lo; i < j; ++i) tmp -= conjq(A_(L + i, j)) * x[i];
                    if (nounit) tmp /= (noconj ? A_(k, j) : conjq(A_(k, j)));
                    x[j] = tmp;
                }
            } else {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TC tmp = x[j];
                    const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                    if (noconj) for (ptrdiff_t i = i_hi - 1; i > j; --i) tmp -= A_(i - j, j) * x[i];
                    else        for (ptrdiff_t i = i_hi - 1; i > j; --i) tmp -= conjq(A_(i - j, j)) * x[i];
                    if (nounit) tmp /= (noconj ? A_(0, j) : conjq(A_(0, j)));
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
                        const TC aij = noconj ? A_(L + i, j) : conjq(A_(L + i, j));
                        tmp -= aij * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= (noconj ? A_(k, j) : conjq(A_(k, j)));
                    x[jx] = tmp;
                    jx += incx;
                    if (j >= k) kx += incx;
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TC tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                    for (ptrdiff_t i = i_hi - 1; i > j; --i) {
                        const TC aij = noconj ? A_(i - j, j) : conjq(A_(i - j, j));
                        tmp -= aij * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= (noconj ? A_(0, j) : conjq(A_(0, j)));
                    x[jx] = tmp;
                    jx -= incx;
                    if ((n - 1 - j) >= k) kx -= incx;
                }
            }
        }
    }
}

EPBLAS_FACADE_TBMV(xtbsv, TC)

#undef A_
