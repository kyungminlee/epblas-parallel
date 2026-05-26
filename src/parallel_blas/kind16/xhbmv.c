/*
 * xhbmv — kind16 complex Hermitian band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A Hermitian with K super-(sub-)diagonals.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>

typedef __complex128 T;
typedef __float128 TR;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xhbmv_(
    const char *uplo,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;
    const char UPLO = up(uplo);

    if (N == 0 || (alpha == zero && beta == one)) return;

    if (beta != one) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (beta == zero) for (int i = 0; i < N; ++i) { y[iy] = zero; iy += incy; }
        else              for (int i = 0; i < N; ++i) { y[iy] = beta * y[iy]; iy += incy; }
    }
    if (alpha == zero) return;

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[i] += t1 * A_(L + i, j);
                    t2 += conjq(A_(L + i, j)) * x[i];
                }
                y[j] += t1 * (TR)crealq(A_(K, j)) + alpha * t2;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                y[j] += t1 * (TR)crealq(A_(0, j));
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    y[i] += t1 * A_(i - j, j);
                    t2 += conjq(A_(i - j, j)) * x[i];
                }
                y[j] += alpha * t2;
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'U') {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                int ix = kx, iy = ky;
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[iy] += t1 * A_(L + i, j);
                    t2 += conjq(A_(L + i, j)) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] += t1 * (TR)crealq(A_(K, j)) + alpha * t2;
                jx += incx; jy += incy;
                if (j >= K) { kx += incx; ky += incy; }
            }
        } else {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                y[jy] += t1 * (TR)crealq(A_(0, j));
                int ix = jx, iy = jy;
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    ix += incx; iy += incy;
                    y[iy] += t1 * A_(i - j, j);
                    t2 += conjq(A_(i - j, j)) * x[ix];
                }
                y[jy] += alpha * t2;
                jx += incx; jy += incy;
            }
        }
    }
}

#undef A_
