/*
 * qtpmv — kind16 (__float128) triangular packed matrix-vector.
 *   x := A*x or A^T*x
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

void qtpmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict ap,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_;
    const int incx = *incx_;
    const T zero = 0.0Q;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (N == 0) return;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        const T tmp = x[j];
                        int k = kk;
                        for (int i = 0; i < j; ++i) { x[i] += tmp * ap[k]; ++k; }
                        if (nounit) x[j] *= ap[kk + j];
                    }
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const T tmp = x[j];
                        int k = kk;
                        for (int i = N - 1; i > j; --i) { x[i] += tmp * ap[k]; --k; }
                        if (nounit) x[j] *= ap[kk - (N - 1 - j)];
                    }
                    kk -= (N - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    if (nounit) tmp *= ap[kk];
                    int k = kk - 1;
                    for (int i = j - 1; i >= 0; --i) { tmp += ap[k] * x[i]; --k; }
                    x[j] = tmp;
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp *= ap[kk];
                    int k = kk + 1;
                    for (int i = j + 1; i < N; ++i) { tmp += ap[k] * x[i]; ++k; }
                    x[j] = tmp;
                    kk += N - j;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    if (x[jx] != zero) {
                        const T tmp = x[jx];
                        int ix = kx;
                        for (int k = kk; k < kk + j; ++k) {
                            x[ix] += tmp * ap[k];
                            ix += incx;
                        }
                        if (nounit) x[jx] *= ap[kk + j];
                    }
                    jx += incx;
                    kk += j + 1;
                }
            } else {
                int kk = (N * (N + 1)) / 2 - 1;
                kx += (N - 1) * incx;
                int jx = kx;
                for (int j = N - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        const T tmp = x[jx];
                        int ix = kx;
                        for (int k = kk; k > kk - (N - 1 - j); --k) {
                            x[ix] += tmp * ap[k];
                            ix -= incx;
                        }
                        if (nounit) x[jx] *= ap[kk - (N - 1 - j)];
                    }
                    jx -= incx;
                    kk -= (N - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                int kk = (N * (N + 1)) / 2 - 1;
                int jx = kx + (N - 1) * incx;
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    int ix = jx;
                    if (nounit) tmp *= ap[kk];
                    for (int k = kk - 1; k >= kk - j; --k) {
                        ix -= incx;
                        tmp += ap[k] * x[ix];
                    }
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                int kk = 0;
                int jx = kx;
                for (int j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    int ix = jx;
                    if (nounit) tmp *= ap[kk];
                    for (int k = kk + 1; k < kk + N - j; ++k) {
                        ix += incx;
                        tmp += ap[k] * x[ix];
                    }
                    x[jx] = tmp;
                    jx += incx;
                    kk += N - j;
                }
            }
        }
    }
}
