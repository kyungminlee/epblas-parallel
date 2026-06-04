/*
 * ytpsv — kind10 complex triangular packed solve.
 */

#include <stddef.h>
#include <ctype.h>

typedef _Complex long double T;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

void ytpsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict ap,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const ptrdiff_t N = *n_;
    const ptrdiff_t incx = *incx_;
    const T zero = 0.0L + 0.0Li;
    const char UPLO = up(uplo);
    const char TR = up(trans);
    const ptrdiff_t noconj = (TR == 'T');
    const ptrdiff_t nounit = (up(diag) != 'U');

    if (N == 0) return;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = (N * (N + 1)) / 2 - 1;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const T tmp = x[j];
                        ptrdiff_t k = kk - 1;
                        for (ptrdiff_t i = j - 1; i >= 0; --i) { x[i] -= tmp * ap[k]; --k; }
                    }
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        if (nounit) x[j] /= ap[kk];
                        const T tmp = x[j];
                        ptrdiff_t k = kk + 1;
                        for (ptrdiff_t i = j + 1; i < N; ++i) { x[i] -= tmp * ap[k]; ++k; }
                    }
                    kk += N - j;
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T tmp = x[j];
                    ptrdiff_t k = kk;
                    if (noconj) for (ptrdiff_t i = 0; i < j; ++i) { tmp -= ap[k] * x[i]; ++k; }
                    else        for (ptrdiff_t i = 0; i < j; ++i) { tmp -= cconj(ap[k]) * x[i]; ++k; }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : cconj(ap[kk + j]));
                    x[j] = tmp;
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (N * (N + 1)) / 2 - 1;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    ptrdiff_t k = kk;
                    if (noconj) for (ptrdiff_t i = N - 1; i > j; --i) { tmp -= ap[k] * x[i]; --k; }
                    else        for (ptrdiff_t i = N - 1; i > j; --i) { tmp -= cconj(ap[k]) * x[i]; --k; }
                    if (nounit) tmp /= (noconj ? ap[kk - (N - 1 - j)] : cconj(ap[kk - (N - 1 - j)]));
                    x[j] = tmp;
                    kk -= (N - j);
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = (N * (N + 1)) / 2 - 1;
                ptrdiff_t jx = kx + (N - 1) * incx;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const T tmp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t k = kk - 1; k >= kk - j; --k) {
                            ix -= incx;
                            x[ix] -= tmp * ap[k];
                        }
                    }
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    if (x[jx] != zero) {
                        if (nounit) x[jx] /= ap[kk];
                        const T tmp = x[jx];
                        ptrdiff_t ix = jx;
                        for (ptrdiff_t k = kk + 1; k < kk + N - j; ++k) {
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
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k < kk + j; ++k) {
                        tmp -= (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk + j] : cconj(ap[kk + j]));
                    x[jx] = tmp;
                    jx += incx;
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (N * (N + 1)) / 2 - 1;
                kx += (N - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k > kk - (N - 1 - j); --k) {
                        tmp -= (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= (noconj ? ap[kk - (N - 1 - j)] : cconj(ap[kk - (N - 1 - j)]));
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= (N - j);
                }
            }
        }
    }
}
