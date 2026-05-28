/*
 * xhemv — kind16 Hermitian matrix-vector multiply.
 *   y := alpha · A · x + beta · y     A Hermitian
 * Diagonal of A treated as real (Hermitian convention).
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>

typedef __complex128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xhemv_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    const T zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;

    if (N == 0) return;

    if (beta != one) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        for (int i = 0; i < N; ++i) {
            if (beta == zero) y[iy] = zero;
            else              y[iy] *= beta;
            iy += incy;
        }
    }
    if (alpha == zero) return;

    if (incx == 1 && incy == 1) {
        if (UPLO == 'L') {
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[i];
                T temp2 = zero;
                const T *ai = &A_(0, i);
                y[i] += temp1 * crealq(ai[i]);
                for (int k = i + 1; k < N; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += conjq(ai[k]) * x[k];
                }
                y[i] += alpha * temp2;
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[i];
                T temp2 = zero;
                const T *ai = &A_(0, i);
                for (int k = 0; k < i; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += conjq(ai[k]) * x[k];
                }
                y[i] += temp1 * crealq(ai[i]) + alpha * temp2;
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'L') {
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[kx + i * incx];
                T temp2 = zero;
                y[ky + i * incy] += temp1 * crealq(A_(i, i));
                for (int k = i + 1; k < N; ++k) {
                    y[ky + k * incy] += temp1 * A_(k, i);
                    temp2 += conjq(A_(k, i)) * x[kx + k * incx];
                }
                y[ky + i * incy] += alpha * temp2;
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[kx + i * incx];
                T temp2 = zero;
                for (int k = 0; k < i; ++k) {
                    y[ky + k * incy] += temp1 * A_(k, i);
                    temp2 += conjq(A_(k, i)) * x[kx + k * incx];
                }
                y[ky + i * incy] += temp1 * crealq(A_(i, i)) + alpha * temp2;
            }
        }
    }
}

#undef A_
