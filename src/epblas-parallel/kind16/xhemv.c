/*
 * xhemv — kind16 Hermitian matrix-vector multiply.
 *   y := alpha · A · x + beta · y     A Hermitian
 * Diagonal of A treated as real (Hermitian convention).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XHEMV_OMP_MIN 128

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
#ifdef _OPENMP
        const int nt = blas_omp_max_threads();
        if (N >= XHEMV_OMP_MIN && nt > 1 && !omp_in_parallel()) {
            /* Parallel column-walk with per-thread private y, then reduce
             * (same pattern as qsymv; Hermitian conjugation stays inside the
             * column loop unchanged). Faithful port of kind10 yhemv. */
            T *y_priv_all = (T *)calloc((size_t)nt * (size_t)N, sizeof(T));
            if (y_priv_all) {
                #pragma omp parallel num_threads(nt)
                {
                    const int tid = omp_get_thread_num();
                    T *y_priv = &y_priv_all[(size_t)tid * N];  /* calloc-zeroed */

                    if (UPLO == 'L') {
                        #pragma omp for schedule(static, 1)
                        for (int j = 0; j < N; ++j) {
                            const T temp1 = alpha * x[j];
                            T temp2 = zero;
                            const T *aj = &A_(0, j);
                            y_priv[j] += temp1 * crealq(aj[j]);
                            for (int k = j + 1; k < N; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += conjq(aj[k]) * x[k];
                            }
                            y_priv[j] += alpha * temp2;
                        }
                    } else {
                        #pragma omp for schedule(static, 1)
                        for (int j = 0; j < N; ++j) {
                            const T temp1 = alpha * x[j];
                            T temp2 = zero;
                            const T *aj = &A_(0, j);
                            for (int k = 0; k < j; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += conjq(aj[k]) * x[k];
                            }
                            y_priv[j] += temp1 * crealq(aj[j]) + alpha * temp2;
                        }
                    }
                    #pragma omp for schedule(static)
                    for (int i = 0; i < N; ++i) {
                        T s = zero;
                        for (int t = 0; t < nt; ++t)
                            s += y_priv_all[(size_t)t * N + i];
                        y[i] += s;
                    }
                }
                free(y_priv_all);
                return;
            }
        }
#endif
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
