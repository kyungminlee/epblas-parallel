/*
 * xtbsv — kind16 complex triangular band solve.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#include "../common/epblas_facade.h"

typedef __complex128 T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xtbsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N, ptrdiff_t K,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx)
{
    const T zero = 0.0Q + 0.0Qi;
    const char UPLO = blas_up(uplo);
    const char TR = blas_up(trans);
    const bool noconj = (TR == 'T');
    const bool nounit = (blas_up(diag) != 'U');

    if (N == 0) return;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const ptrdiff_t L = K - j;
                        if (nounit) x[j] /= A_(K, j);
                        const T tmp = x[j];
                        const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i) x[i] -= tmp * A_(L + i, j);
                    }
                }
            } else {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= A_(0, j);
                        const T tmp = x[j];
                        const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (ptrdiff_t i = j + 1; i < i_hi; ++i) x[i] -= tmp * A_(i - j, j);
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T tmp = x[j];
                    const ptrdiff_t L = K - j;
                    const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                    if (noconj) for (ptrdiff_t i = i_lo; i < j; ++i) tmp -= A_(L + i, j) * x[i];
                    else        for (ptrdiff_t i = i_lo; i < j; ++i) tmp -= conjq(A_(L + i, j)) * x[i];
                    if (nounit) tmp /= (noconj ? A_(K, j) : conjq(A_(K, j)));
                    x[j] = tmp;
                }
            } else {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    if (noconj) for (ptrdiff_t i = i_hi - 1; i > j; --i) tmp -= A_(i - j, j) * x[i];
                    else        for (ptrdiff_t i = i_hi - 1; i > j; --i) tmp -= conjq(A_(i - j, j)) * x[i];
                    if (nounit) tmp /= (noconj ? A_(0, j) : conjq(A_(0, j)));
                    x[j] = tmp;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                kx += (N - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    kx -= incx;
                    if (x[jx] != zero) {
                        ptrdiff_t ix = kx;
                        const ptrdiff_t L = K - j;
                        if (nounit) x[jx] /= A_(K, j);
                        const T tmp = x[jx];
                        const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                        for (ptrdiff_t i = j - 1; i >= i_lo; --i) {
                            x[ix] -= tmp * A_(L + i, j);
                            ix -= incx;
                        }
                    }
                    jx -= incx;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    kx += incx;
                    if (x[jx] != zero) {
                        ptrdiff_t ix = kx;
                        if (nounit) x[jx] /= A_(0, j);
                        const T tmp = x[jx];
                        const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
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
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const ptrdiff_t L = K - j;
                    const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                    for (ptrdiff_t i = i_lo; i < j; ++i) {
                        const T aij = noconj ? A_(L + i, j) : conjq(A_(L + i, j));
                        tmp -= aij * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= (noconj ? A_(K, j) : conjq(A_(K, j)));
                    x[jx] = tmp;
                    jx += incx;
                    if (j >= K) kx += incx;
                }
            } else {
                kx += (N - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (ptrdiff_t i = i_hi - 1; i > j; --i) {
                        const T aij = noconj ? A_(i - j, j) : conjq(A_(i - j, j));
                        tmp -= aij * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= (noconj ? A_(0, j) : conjq(A_(0, j)));
                    x[jx] = tmp;
                    jx -= incx;
                    if ((N - 1 - j) >= K) kx -= incx;
                }
            }
        }
    }
}

EPBLAS_FACADE_TBMV(xtbsv, T)

#undef A_
