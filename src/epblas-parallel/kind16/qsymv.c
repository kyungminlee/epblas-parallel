/*
 * qsymv — kind16 symmetric matrix-vector multiply.
 *   y := alpha · A · x + beta · y     A symmetric
 *
 * Netlib two-pass pattern. Threaded (unit-stride) with a per-thread private
 * y accumulator: the two-pass column walk writes y[k] for the reflected
 * triangle, which races if threads share column ranges, so each thread sums
 * its column contributions into a private y_priv[N] and a final reduction
 * folds them into y. Quad is compute-bound under libquadmath so the O(N^2)
 * column work threads ~4x while the O(nt·N) reduction stays negligible.
 * Faithful port of kind10 esymv.
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

#define QSYMV_OMP_MIN 128

typedef __float128 T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void qsymv_core(
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
    const T zero = 0.0Q, one = 1.0Q;

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
        if (N >= QSYMV_OMP_MIN && blas_omp_should_thread()) {
            /* Parallel two-pass with per-thread private y accumulator;
             * schedule(static,1) interleaves columns to balance the
             * triangular per-column work (linear in N-j for L, j for U). */
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
                            y_priv[j] += temp1 * aj[j];
                            for (ptrdiff_t k = j + 1; k < N; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += aj[k] * x[k];
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
                                temp2 += aj[k] * x[k];
                            }
                            y_priv[j] += temp1 * aj[j] + alpha * temp2;
                        }
                    }
                    /* Implicit barrier after `omp for` orders the writes
                     * before the reduction reads. */
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
            /* aligned_alloc failed — fall through to serial. */
        }
#endif
        if (UPLO == 'L') {
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[i];
                T temp2 = zero;
                const T *ai = &A_(0, i);
                y[i] += temp1 * ai[i];
                for (ptrdiff_t k = i + 1; k < N; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += ai[k] * x[k];
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
                    temp2 += ai[k] * x[k];
                }
                y[i] += temp1 * ai[i] + alpha * temp2;
            }
        }
    } else {
        /* General-stride fallback: walks ix/iy by incrementing (matches
         * Netlib reference's IX=IX+INCX, not k*incx recomputation). */
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'L') {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[jx];
                T temp2 = zero;
                y[jy] += temp1 * A_(i, i);
                ptrdiff_t ix = jx, iy = jy;
                for (ptrdiff_t k = i + 1; k < N; ++k) {
                    ix += incx; iy += incy;
                    y[iy] += temp1 * A_(k, i);
                    temp2 += A_(k, i) * x[ix];
                }
                y[jy] += alpha * temp2;
                jx += incx; jy += incy;
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t i = 0; i < N; ++i) {
                const T temp1 = alpha * x[jx];
                T temp2 = zero;
                ptrdiff_t ix = kx, iy = ky;
                for (ptrdiff_t k = 0; k < i; ++k) {
                    y[iy] += temp1 * A_(k, i);
                    temp2 += A_(k, i) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] += temp1 * A_(i, i) + alpha * temp2;
                jx += incx; jy += incy;
            }
        }
    }
}


EPBLAS_FACADE_SYMV(qsymv, T)

#undef A_
