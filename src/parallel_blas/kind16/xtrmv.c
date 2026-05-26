/*
 * xtrmv — kind16 complex triangular matrix-vector.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>

typedef __complex128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xtrmv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n_,
    const T *restrict a, const int *lda_,
    T *restrict x, const int *incx_,
    size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    const int N = *n_;
    const int lda = *lda_, incx = *incx_;
    const char UPLO = up(uplo);
    const char TR   = up(trans);
    const char DIAG = up(diag);
    const int nounit = (DIAG != 'U');

    if (N == 0) return;
    const T zero = 0.0Q + 0.0Qi;

    if (incx == 1) {
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int j = N - 1; j >= 0; --j) {
                    const T temp = x[j];
                    if (temp != zero) {
                        const T *aj = &A_(0, j);
                        for (int i = j + 1; i < N; ++i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    const T temp = x[j];
                    if (temp != zero) {
                        const T *aj = &A_(0, j);
                        for (int i = 0; i < j; ++i) x[i] += temp * aj[i];
                    }
                    if (nounit) x[j] *= A_(j, j);
                }
            }
        } else {
            const int conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int j = 0; j < N; ++j) {
                    T temp = x[j];
                    if (nounit) temp *= (conj_a ? conjq(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) {
                        for (int i = j + 1; i < N; ++i) temp += conjq(aj[i]) * x[i];
                    } else {
                        for (int i = j + 1; i < N; ++i) temp += aj[i] * x[i];
                    }
                    x[j] = temp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T temp = x[j];
                    if (nounit) temp *= (conj_a ? conjq(A_(j, j)) : A_(j, j));
                    const T *aj = &A_(0, j);
                    if (conj_a) {
                        for (int i = 0; i < j; ++i) temp += conjq(aj[i]) * x[i];
                    } else {
                        for (int i = 0; i < j; ++i) temp += aj[i] * x[i];
                    }
                    x[j] = temp;
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        if (TR == 'N') {
            if (UPLO == 'L') {
                for (int j = N - 1; j >= 0; --j) {
                    const T temp = x[kx + j * incx];
                    if (temp != zero)
                        for (int i = j + 1; i < N; ++i) x[kx + i * incx] += temp * A_(i, j);
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            } else {
                for (int j = 0; j < N; ++j) {
                    const T temp = x[kx + j * incx];
                    if (temp != zero)
                        for (int i = 0; i < j; ++i) x[kx + i * incx] += temp * A_(i, j);
                    if (nounit) x[kx + j * incx] *= A_(j, j);
                }
            }
        } else {
            const int conj_a = (TR == 'C');
            if (UPLO == 'L') {
                for (int j = 0; j < N; ++j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp *= (conj_a ? conjq(A_(j, j)) : A_(j, j));
                    for (int i = j + 1; i < N; ++i) {
                        const T aij = conj_a ? conjq(A_(i, j)) : A_(i, j);
                        temp += aij * x[kx + i * incx];
                    }
                    x[kx + j * incx] = temp;
                }
            } else {
                for (int j = N - 1; j >= 0; --j) {
                    T temp = x[kx + j * incx];
                    if (nounit) temp *= (conj_a ? conjq(A_(j, j)) : A_(j, j));
                    for (int i = 0; i < j; ++i) {
                        const T aij = conj_a ? conjq(A_(i, j)) : A_(i, j);
                        temp += aij * x[kx + i * incx];
                    }
                    x[kx + j * incx] = temp;
                }
            }
        }
    }
}

#undef A_
