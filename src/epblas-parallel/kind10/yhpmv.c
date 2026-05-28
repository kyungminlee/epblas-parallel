/*
 * yhpmv — kind10 complex Hermitian packed matrix-vector multiply.
 */

#include <stddef.h>
#include <ctype.h>

typedef _Complex long double T;
typedef long double TR;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

void yhpmv_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict ap,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0L + 0.0Li, one = 1.0L + 0.0Li;
    const char UPLO = up(uplo);

    if (N == 0 || (alpha == zero && beta == one)) return;

    if (beta != one) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (beta == zero) for (int i = 0; i < N; ++i) { y[iy] = zero; iy += incy; }
        else              for (int i = 0; i < N; ++i) { y[iy] = beta * y[iy]; iy += incy; }
    }
    if (alpha == zero) return;

    int kk = 0;
    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                int k = kk;
                for (int i = 0; i < j; ++i) {
                    y[i] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[i];
                    ++k;
                }
                y[j] += t1 * (TR)__real__ ap[kk + j] + alpha * t2;
                kk += j + 1;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                y[j] += t1 * (TR)__real__ ap[kk];
                int k = kk + 1;
                for (int i = j + 1; i < N; ++i) {
                    y[i] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[i];
                    ++k;
                }
                y[j] += alpha * t2;
                kk += N - j;
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
                for (int k = kk; k < kk + j; ++k) {
                    y[iy] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] += t1 * (TR)__real__ ap[kk + j] + alpha * t2;
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                y[jy] += t1 * (TR)__real__ ap[kk];
                int ix = jx, iy = jy;
                for (int k = kk + 1; k < kk + N - j; ++k) {
                    ix += incx; iy += incy;
                    y[iy] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[ix];
                }
                y[jy] += alpha * t2;
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}
