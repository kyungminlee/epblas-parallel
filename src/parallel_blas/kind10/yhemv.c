/*
 * yhemv — kind10 complex Hermitian matrix-vector multiply.
 *   y := alpha · A · x + beta · y    where A is N×N Hermitian
 *
 * Same two-pass pattern as esymv. Hermitian twist: the cross
 * reflection conjugates A. Diagonal of A is treated as real (matches
 * Netlib ZHEMV — uses DBLE(A(I,I))).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YHEMV_OMP_MIN 128

typedef _Complex long double T;
static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void yhemv_(
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

    if (N == 0) return;

    if (beta != ONE) {
        if (incy == 1) {
            if (beta == ZERO) for (int i = 0; i < N; ++i) y[i] = ZERO;
            else              for (int i = 0; i < N; ++i) y[i] *= beta;
        } else {
            int iy = (incy < 0) ? -(N - 1) * incy : 0;
            for (int i = 0; i < N; ++i) {
                if (beta == ZERO) y[iy] = ZERO;
                else              y[iy] *= beta;
                iy += incy;
            }
        }
    }

    if (alpha == ZERO) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int nt = blas_omp_max_threads();
        const int use_omp = (N >= YHEMV_OMP_MIN && nt > 1 && !omp_in_parallel());
#else
        const int use_omp = 0;
        const int nt = 1;
#endif
        if (use_omp) {
            /* Parallel column-walk with per-thread private y, then reduce.
             * Same pattern as esymv (Addendum 36). Hermitian conjugation
             * stays inside the column loop unchanged. */
            T *y_priv_all = (T *)aligned_alloc(64,
                (((size_t)nt * N * sizeof(T)) + 63) & ~(size_t)63);
            if (y_priv_all) {
#ifdef _OPENMP
                #pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    T *y_priv = &y_priv_all[(size_t)tid * N];
                    for (int k = 0; k < N; ++k) y_priv[k] = ZERO;

                    if (UPLO == 'L') {
                        #pragma omp for schedule(static, 1)
                        for (int j = 0; j < N; ++j) {
                            const T temp1 = alpha * x[j];
                            T temp2 = ZERO;
                            const T *aj = &A_(0, j);
                            y_priv[j] += temp1 * __real__ aj[j];
                            for (int k = j + 1; k < N; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += cconj(aj[k]) * x[k];
                            }
                            y_priv[j] += alpha * temp2;
                        }
                    } else {
                        #pragma omp for schedule(static, 1)
                        for (int j = 0; j < N; ++j) {
                            const T temp1 = alpha * x[j];
                            T temp2 = ZERO;
                            const T *aj = &A_(0, j);
                            for (int k = 0; k < j; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += cconj(aj[k]) * x[k];
                            }
                            y_priv[j] += temp1 * __real__ aj[j] + alpha * temp2;
                        }
                    }

                    #pragma omp for schedule(static)
                    for (int i = 0; i < N; ++i) {
                        T s = ZERO;
                        for (int t = 0; t < nt; ++t)
                            s += y_priv_all[(size_t)t * N + i];
                        y[i] += s;
                    }
                }
#endif
                free(y_priv_all);
                return;
            }
        }
        if (UPLO == 'L') {
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                y[i] += temp1 * __real__ ai[i];
                for (int k = i + 1; k < N; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += cconj(ai[k]) * x[k];
                }
                y[i] += alpha * temp2;
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (int k = 0; k < i; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += cconj(ai[k]) * x[k];
                }
                y[i] += temp1 * __real__ ai[i] + alpha * temp2;
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'L') {
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[kx + i * incx];
                T temp2 = ZERO;
                y[ky + i * incy] += temp1 * __real__ A_(i, i);
                for (int k = i + 1; k < N; ++k) {
                    y[ky + k * incy] += temp1 * A_(k, i);
                    temp2 += cconj(A_(k, i)) * x[kx + k * incx];
                }
                y[ky + i * incy] += alpha * temp2;
            }
        } else {
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[kx + i * incx];
                T temp2 = ZERO;
                for (int k = 0; k < i; ++k) {
                    y[ky + k * incy] += temp1 * A_(k, i);
                    temp2 += cconj(A_(k, i)) * x[kx + k * incx];
                }
                y[ky + i * incy] += temp1 * __real__ A_(i, i) + alpha * temp2;
            }
        }
    }
}

#undef A_
