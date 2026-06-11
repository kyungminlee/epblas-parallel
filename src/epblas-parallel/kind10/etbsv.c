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
    const ptrdiff_t N = *n_, K = *k_;
    const ptrdiff_t lda = *lda_, incx = *incx_;
    const T zero = 0.0L;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';
    const ptrdiff_t nounit = (up(diag) != 'U');

    if (N == 0) return;

    if (incx == 1) {
        /* incx==1 hot path: hoist the column base pointer so the two array
         * walks (band column + x) collapse onto a SINGLE byte-offset
         * induction (col[off+i] and x[i] share i). Verified in-process to be
         * parity-or-better than the A_() macro form in every U/L × N/T cell,
         * with a reproducible ~0.5-1% edge on the Upper shapes (where the
         * macro's L=K-j offset otherwise lengthens the per-element address
         * arithmetic). Walk direction is irrelevant to the codegen — keep the
         * forward inner walks (hardware-prefetcher friendly). The strided
         * branch below cannot collapse (its ix induction is independent of the
         * band index) so it stays on the macro form. */
        if (TR == 'N') {
            if (UPLO == 'U') {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const T *restrict col = &a[(size_t)j * lda];
                        const ptrdiff_t off = K - j;
                        if (nounit) x[j] /= col[K];
                        const T tmp = x[j];
                        const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                        for (ptrdiff_t i = i_lo; i < j; ++i) x[i] -= tmp * col[off + i];
                    }
                }
            } else {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    if (x[j] != zero) {
                        const T *restrict col = &a[(size_t)j * lda];
                        const ptrdiff_t off = -j;
                        if (nounit) x[j] /= col[0];
                        const T tmp = x[j];
                        const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                        for (ptrdiff_t i = j + 1; i < i_hi; ++i) x[i] -= tmp * col[off + i];
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T tmp = x[j];
                    const T *restrict col = &a[(size_t)j * lda];
                    const ptrdiff_t off = K - j;
                    const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                    for (ptrdiff_t i = i_lo; i < j; ++i) tmp -= col[off + i] * x[i];
                    if (nounit) tmp /= col[K];
                    x[j] = tmp;
                }
            } else {
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const T *restrict col = &a[(size_t)j * lda];
                    const ptrdiff_t off = -j;
                    const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (ptrdiff_t i = i_hi - 1; i > j; --i) tmp -= col[off + i] * x[i];
                    if (nounit) tmp /= col[0];
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
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = N - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    ptrdiff_t ix = kx;
                    const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                    for (ptrdiff_t i = i_hi - 1; i > j; --i) {
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
