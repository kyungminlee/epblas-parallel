/*
 * xgbmv — kind16 complex general band matrix-vector multiply.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XGBMV_OMP_MIN 64

typedef __complex128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xgbmv_(
    const char *trans,
    const int *m_, const int *n_,
    const int *kl_, const int *ku_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t trans_len)
{
    (void)trans_len;
    const int M = *m_, N = *n_;
    const int KL = *kl_, KU = *ku_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;
    const char TR = up(trans);
    const int noconj = (TR == 'T');

    if (M == 0 || N == 0 || (alpha == zero && beta == one)) return;

    const int leny = (TR == 'N') ? M : N;
    const int lenx = (TR == 'N') ? N : M;

    if (beta != one) {
        int iy = (incy < 0) ? -(leny - 1) * incy : 0;
        if (beta == zero) for (int i = 0; i < leny; ++i) { y[iy] = zero; iy += incy; }
        else              for (int i = 0; i < leny; ++i) { y[iy] = beta * y[iy]; iy += incy; }
    }
    if (alpha == zero) return;

    if (TR == 'N' && incx == 1 && incy == 1) {
        for (int j = 0; j < N; ++j) {
            const T tmp = alpha * x[j];
            const int i_lo = (j - KU > 0) ? (j - KU) : 0;
            const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const int k = KU - j;
            for (int i = i_lo; i < i_hi; ++i) y[i] += tmp * A_(k + i, j);
        }
    } else if (TR != 'N' && incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= XGBMV_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T s = zero;
            const int i_lo = (j - KU > 0) ? (j - KU) : 0;
            const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const int k = KU - j;
            if (noconj) for (int i = i_lo; i < i_hi; ++i) s += A_(k + i, j) * x[i];
            else        for (int i = i_lo; i < i_hi; ++i) s += conjq(A_(k + i, j)) * x[i];
            y[j] += alpha * s;
        }
    } else {
        int kx = (incx < 0) ? -(lenx - 1) * incx : 0;
        int ky = (incy < 0) ? -(leny - 1) * incy : 0;
        if (TR == 'N') {
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                const T tmp = alpha * x[jx];
                int iy = ky;
                const int i_lo = (j - KU > 0) ? (j - KU) : 0;
                const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
                const int k = KU - j;
                for (int i = i_lo; i < i_hi; ++i) {
                    y[iy] += tmp * A_(k + i, j);
                    iy += incy;
                }
                jx += incx;
                if (j >= KU) ky += incy;
            }
        } else {
            int jy = ky;
            for (int j = 0; j < N; ++j) {
                T s = zero;
                int ix = kx;
                const int i_lo = (j - KU > 0) ? (j - KU) : 0;
                const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
                const int k = KU - j;
                if (noconj) {
                    for (int i = i_lo; i < i_hi; ++i) {
                        s += A_(k + i, j) * x[ix];
                        ix += incx;
                    }
                } else {
                    for (int i = i_lo; i < i_hi; ++i) {
                        s += conjq(A_(k + i, j)) * x[ix];
                        ix += incx;
                    }
                }
                y[jy] += alpha * s;
                jy += incy;
                if (j >= KU) kx += incx;
            }
        }
    }
}

#undef A_
