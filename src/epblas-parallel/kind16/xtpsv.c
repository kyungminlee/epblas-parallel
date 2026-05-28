/*
 * xtpsv — kind16 complex triangular packed solve.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>

typedef __complex128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

void xtpsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict ap,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_;
    const int incx = *incx_;
    const T zero = 0.0Q + 0.0Qi;
    const char UPLO = up(uplo);
    const char TR = up(trans);
    const int noconj = (TR == 'T');
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const T tmp = x[j];
                        int k = kk - 1;
                        for (int i = j - 1; i >= 0; --i) { x[i] -= tmp * ap[k]; --k; }
                    }
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const T tmp = x[j];
                        int k = kk + 1;
                        for (int i = j + 1; i < N; ++i) { x[i] -= tmp * ap[k]; ++k; }
                    }
                    kk += N - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    int k = kk;
                    if (noconj) for (int i = 0; i < j; ++i) { tmp -= ap[k] * x[i]; ++k; }
                    else        for (int i = 0; i < j; ++i) { tmp -= conjq(ap[k]) * x[i]; ++k; }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : conjq(ap[kk + j]));
                    x[j] = tmp;
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    int k = kk;
                    if (noconj) for (int i = N - 1; i > j; --i) { tmp -= ap[k] * x[i]; --k; }
                    else        for (int i = N - 1; i > j; --i) { tmp -= conjq(ap[k]) * x[i]; --k; }
                    if (nounit) tmp /= (noconj ? ap[kk - (N - 1 - j)] : conjq(ap[kk - (N - 1 - j)]));
                    x[j] = tmp;
                    kk -= (N - j);
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                int jx = kx + (N - 1) * incx;
                for (int j = N - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const T tmp = x[jx];
                        int ix = jx;
                        for (int k = kk - 1; k >= kk - j; --k) {
                            ix -= incx;
                            x[ix] -= tmp * ap[k];
                        }
                    }
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const T tmp = x[jx];
                        int ix = jx;
                        for (int k = kk + 1; k < kk + N - j; ++k) {
                            ix += incx;
                            x[ix] -= tmp * ap[k];
                        }
                    }
                    jx += incx;
                    kk += N - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    int ix = kx;
                    for (int k = kk; k < kk + j; ++k) {
                        tmp -= (noconj ? ap[k] : conjq(ap[k])) * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : conjq(ap[kk + j]));
                    x[jx] = tmp;
                    jx += incx;
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    int ix = kx;
                    for (int k = kk; k > kk - (N - 1 - j); --k) {
                        tmp -= (noconj ? ap[k] : conjq(ap[k])) * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk - (N - 1 - j)] : conjq(ap[kk - (N - 1 - j)]));
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= (N - j);
                }
            }
        }
    }
}
