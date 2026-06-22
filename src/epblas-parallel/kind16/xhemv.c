/*
 * xhemv — kind16 Hermitian matrix-vector multiply.
 *   y := alpha · A · x + beta · y     A Hermitian
 * Diagonal of A treated as real (Hermitian convention).
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define XHEMV_OMP_MIN 128

typedef __complex128 T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xhemv_core(
    char uplo,
    ptrdiff_t N,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const T zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;

    if (N == 0) return;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(N - 1) * incy : 0;
        for (ptrdiff_t i = 0; i < N; ++i) {
            if (beta == zero) y[iy] = zero;
            else              y[iy] *= beta;
            iy += incy;
        }
    }
    if (alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const ptrdiff_t nt = blas_omp_max_threads();
        if (N >= XHEMV_OMP_MIN && blas_omp_should_thread()) {
            /* Parallel column-walk with per-thread private y, then reduce
             * (same pattern as qsymv; Hermitian conjugation stays inside the
             * column loop unchanged). Faithful port of kind10 yhemv. */
            T *y_priv_all = (T *)calloc((size_t)nt * (size_t)N, sizeof(T));
            if (y_priv_all) {
                #pragma omp parallel num_threads(nt)
                {
                    const ptrdiff_t tid = omp_get_thread_num();
                    T *y_priv = &y_priv_all[(size_t)tid * N];  /* calloc-zeroed */

                    if (UPLO == 'L') {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < N; ++j) {
                            const T temp1 = alpha * x[j];
                            T temp2 = zero;
                            const T *aj = &A_(0, j);
                            y_priv[j] += temp1 * crealq(aj[j]);
                            for (ptrdiff_t k = j + 1; k < N; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += conjq(aj[k]) * x[k];
                            }
                            y_priv[j] += alpha * temp2;
                        }
                    } else {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < N; ++j) {
                            const T temp1 = alpha * x[j];
                            T temp2 = zero;
                            const T *aj = &A_(0, j);
                            for (ptrdiff_t k = 0; k < j; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += conjq(aj[k]) * x[k];
                            }
                            y_priv[j] += temp1 * crealq(aj[j]) + alpha * temp2;
                        }
                    }
                    #pragma omp for schedule(static)
                    for (ptrdiff_t i = 0; i < N; ++i) {
                        T s = zero;
                        for (ptrdiff_t t = 0; t < nt; ++t)
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
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[i];
                T temp2 = zero;
                const T *ai = &A_(0, i);
                y[i] += temp1 * crealq(ai[i]);
                for (ptrdiff_t k = i + 1; k < N; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += conjq(ai[k]) * x[k];
                }
                y[i] += alpha * temp2;
            }
        } else {
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[i];
                T temp2 = zero;
                const T *ai = &A_(0, i);
                for (ptrdiff_t k = 0; k < i; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += conjq(ai[k]) * x[k];
                }
                y[i] += temp1 * crealq(ai[i]) + alpha * temp2;
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'L') {
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[kx + (ptrdiff_t)i * incx];
                T temp2 = zero;
                y[ky + (ptrdiff_t)i * incy] += temp1 * crealq(A_(i, i));
                for (ptrdiff_t k = i + 1; k < N; ++k) {
                    y[ky + (ptrdiff_t)k * incy] += temp1 * A_(k, i);
                    temp2 += conjq(A_(k, i)) * x[kx + (ptrdiff_t)k * incx];
                }
                y[ky + (ptrdiff_t)i * incy] += alpha * temp2;
            }
        } else {
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[kx + (ptrdiff_t)i * incx];
                T temp2 = zero;
                for (ptrdiff_t k = 0; k < i; ++k) {
                    y[ky + (ptrdiff_t)k * incy] += temp1 * A_(k, i);
                    temp2 += conjq(A_(k, i)) * x[kx + (ptrdiff_t)k * incx];
                }
                y[ky + (ptrdiff_t)i * incy] += temp1 * crealq(A_(i, i)) + alpha * temp2;
            }
        }
    }
}

#undef A_

EPBLAS_FACADE_SYMV(xhemv, T)
