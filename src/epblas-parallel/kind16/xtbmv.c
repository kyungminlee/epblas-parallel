/*
 * xtbmv — kind16 complex triangular band matrix-vector.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>

typedef __complex128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xtbmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_, const int *k_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_;
    const T zero = 0.0Q + 0.0Qi;
    const char UPLO = up(uplo);
    const char TR = up(trans);
    const int noconj = (TR == 'T');
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        const T tmp = x[j];
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) x[i] += tmp * A_(L + i, j);
                        if (nounit) x[j] *= A_(K, j);
                    }
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const T tmp = x[j];
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) x[i] += tmp * A_(i - j, j);
                        if (nounit) x[j] *= A_(0, j);
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const int L = K - j;
                    if (nounit) tmp *= (noconj ? A_(K, j) : conjq(A_(K, j)));
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    if (noconj) for (int i = j - 1; i >= i_lo; --i) tmp += A_(L + i, j) * x[i];
                    else        for (int i = j - 1; i >= i_lo; --i) tmp += conjq(A_(L + i, j)) * x[i];
                    x[j] = tmp;
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp *= (noconj ? A_(0, j) : conjq(A_(0, j)));
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    if (noconj) for (int i = j + 1; i < i_hi; ++i) tmp += A_(i - j, j) * x[i];
                    else        for (int i = j + 1; i < i_hi; ++i) tmp += conjq(A_(i - j, j)) * x[i];
                    x[j] = tmp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (x[jx] != zero) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int L = K - j;
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = i_lo; i < j; ++i) {
                            x[ix] += tmp * A_(L + i, j);
                            ix += incx;
                        }
                        if (nounit) x[jx] *= A_(K, j);
                    }
                    jx += incx;
                    if (j >= K) kx += incx;
                }
            } else {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        const T tmp = x[jx];
                        int ix = kx;
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = i_hi - 1; i > j; --i) {
                            x[ix] += tmp * A_(i - j, j);
                            ix -= incx;
                        }
                        if (nounit) x[jx] *= A_(0, j);
                    }
                    jx -= incx;
                    if ((N - 1 - j) >= K) kx -= incx;
                }
            }
        } else {
            if (UPLO == 'U') {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    kx -= incx;
                    int ix = kx;
                    const int L = K - j;
                    if (nounit) tmp *= (noconj ? A_(K, j) : conjq(A_(K, j)));
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = j - 1; i >= i_lo; --i) {
                        const T aij = noconj ? A_(L + i, j) : conjq(A_(L + i, j));
                        tmp += aij * x[ix];
                        ix -= incx;
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            } else {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    kx += incx;
                    int ix = kx;
                    if (nounit) tmp *= (noconj ? A_(0, j) : conjq(A_(0, j)));
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = j + 1; i < i_hi; ++i) {
                        const T aij = noconj ? A_(i - j, j) : conjq(A_(i - j, j));
                        tmp += aij * x[ix];
                        ix += incx;
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
    }
}

#undef A_
