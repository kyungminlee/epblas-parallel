/*
 * etbsv — kind10 (long double) triangular band solve.
 *   x := inv(A)*x or inv(A^T)*x, A triangular band with K+1 diagonals.
 */

#include <stddef.h>
#include <ctype.h>

typedef long double T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void etbsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_, const int *k_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_;
    const T zero = 0.0L;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
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
                        /* Inner iterations are independent (each writes a
                         * distinct x[i] using a constant tmp); walk forward
                         * for the hardware prefetcher. The migrated walks
                         * backward but hits the same ~0.84× floor. */
                        for (int i = i_lo; i < j; ++i) x[i] -= tmp * A_(L + i, j);
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
                    for (int i = i_lo; i < j; ++i) tmp -= A_(L + i, j) * x[i];
                    if (nounit) tmp /= A_(K, j);
                    x[j] = tmp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (int i = i_hi - 1; i > j; --i) tmp -= A_(i - j, j) * x[i];
                    if (nounit) tmp /= A_(0, j);
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
                        tmp -= A_(L + i, j) * x[ix];
                        ix += incx;
                    }
                    if (nounit) tmp /= A_(K, j);
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
                        tmp -= A_(i - j, j) * x[ix];
                        ix -= incx;
                    }
                    if (nounit) tmp /= A_(0, j);
                    x[jx] = tmp;
                    jx -= incx;
                    if ((N - 1 - j) >= K) kx -= incx;
                }
            }
        }
    }
}

#undef A_
