/*
 * ytbsv — kind10 complex triangular band solve.
 */

#include <stddef.h>
#include <ctype.h>

typedef _Complex long double T;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void ytbsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_, const int *k_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_;
    const T zero = 0.0L + 0.0Li;
    const char UPLO = up(uplo);
    const char TR = up(trans);
    const int noconj = (TR == 'T');
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                for (int j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const int L = K - j;
                        if (nounit) x[j] /= A_(K, j);
                        const T tmp = x[j];
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = j - 1; i >= i_lo; --i) x[i] -= tmp * A_(L + i, j);
                    }
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= A_(0, j);
                        const T tmp = x[j];
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = j + 1; i < i_hi; ++i) x[i] -= tmp * A_(i - j, j);
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    const int L = K - j;
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    if (noconj) for (int i = i_lo; i < j; ++i) tmp -= A_(L + i, j) * x[i];
                    else        for (int i = i_lo; i < j; ++i) tmp -= cconj(A_(L + i, j)) * x[i];
                    if (nounit) tmp /= (noconj ? A_(K, j) : cconj(A_(K, j)));
                    x[j] = tmp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    if (noconj) for (int i = i_hi - 1; i > j; --i) tmp -= A_(i - j, j) * x[i];
                    else        for (int i = i_hi - 1; i > j; --i) tmp -= cconj(A_(i - j, j)) * x[i];
                    if (nounit) tmp /= (noconj ? A_(0, j) : cconj(A_(0, j)));
                    x[j] = tmp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    kx -= incx;
                    if (x[jx] != zero) {
                        int ix = kx;
                        const int L = K - j;
                        if (nounit) x[jx] /= A_(K, j);
                        const T tmp = x[jx];
                        const int i_lo = (j - K > 0) ? (j - K) : 0;
                        for (int i = j - 1; i >= i_lo; --i) {
                            x[ix] -= tmp * A_(L + i, j);
                            ix -= incx;
                        }
                    }
                    jx -= incx;
                }
            } else {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    kx += incx;
                    if (x[jx] != zero) {
                        int ix = kx;
                        if (nounit) x[jx] /= A_(0, j);
                        const T tmp = x[jx];
                        const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (int i = j + 1; i < i_hi; ++i) {
                            x[ix] -= tmp * A_(i - j, j);
                            ix += incx;
                        }
                    }
                    jx += incx;
                }
            }
        } else {
            if (UPLO == 'U') {
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    int ix = kx;
                    const int L = K - j;
                    const int i_lo = (j - K > 0) ? (j - K) : 0;
                    for (int i = i_lo; i < j; ++i) {
                        const T aij = noconj ? A_(L + i, j) : cconj(A_(L + i, j));
                        tmp -= aij * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= (noconj ? A_(K, j) : cconj(A_(K, j)));
                    x[jx] = tmp;
                    jx += incx;
                    if (j >= K) kx += incx;
                }
            } else {
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    int ix = kx;
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = i_hi - 1; i > j; --i) {
                        const T aij = noconj ? A_(i - j, j) : cconj(A_(i - j, j));
                        tmp -= aij * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= (noconj ? A_(0, j) : cconj(A_(0, j)));
                    x[jx] = tmp;
                    jx -= incx;
                    if ((N - 1 - j) >= K) kx -= incx;
                }
            }
        }
    }
}

#undef A_
