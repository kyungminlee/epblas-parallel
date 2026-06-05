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
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QSYMV_OMP_MIN 128

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void qsymv_(
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
    const T zero = 0.0Q, one = 1.0Q;

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
        if (N >= QSYMV_OMP_MIN && nt > 1 && !omp_in_parallel()) {
            /* Parallel two-pass with per-thread private y accumulator;
             * schedule(static,1) interleaves columns to balance the
             * triangular per-column work (linear in N-j for L, j for U). */
            T *y_priv_all = (T *)aligned_alloc(64,
                (((size_t)nt * N * sizeof(T)) + 63) & ~(size_t)63);
            if (y_priv_all) {
                #pragma omp parallel
                {
                    const int tid = omp_get_thread_num();
                    T *y_priv = &y_priv_all[(size_t)tid * N];
                    for (int k = 0; k < N; ++k) y_priv[k] = zero;

                    if (UPLO == 'L') {
                        #pragma omp for schedule(static, 1)
                        for (int j = 0; j < N; ++j) {
                            const T temp1 = alpha * x[j];
                            T temp2 = zero;
                            const T *aj = &A_(0, j);
                            y_priv[j] += temp1 * aj[j];
                            for (int k = j + 1; k < N; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += aj[k] * x[k];
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
                                temp2 += aj[k] * x[k];
                            }
                            y_priv[j] += temp1 * aj[j] + alpha * temp2;
                        }
                    }
                    /* Implicit barrier after `omp for` orders the writes
                     * before the reduction reads. */
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
            /* aligned_alloc failed — fall through to serial. */
        }
#endif
        if (UPLO == 'L') {
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[i];
                T temp2 = zero;
                const T *ai = &A_(0, i);
                y[i] += temp1 * ai[i];
                for (int k = i + 1; k < N; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += ai[k] * x[k];
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
                    temp2 += ai[k] * x[k];
                }
                y[i] += temp1 * ai[i] + alpha * temp2;
            }
        }
    } else {
        /* General-stride fallback: walks ix/iy by incrementing (matches
         * Netlib reference's IX=IX+INCX, not k*incx recomputation). */
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'L') {
            int jx = kx, jy = ky;
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[jx];
                T temp2 = zero;
                y[jy] += temp1 * A_(i, i);
                int ix = jx, iy = jy;
                for (int k = i + 1; k < N; ++k) {
                    ix += incx; iy += incy;
                    y[iy] += temp1 * A_(k, i);
                    temp2 += A_(k, i) * x[ix];
                }
                y[jy] += alpha * temp2;
                jx += incx; jy += incy;
            }
        } else {
            int jx = kx, jy = ky;
            for (int i = 0; i < N; ++i) {
                const T temp1 = alpha * x[jx];
                T temp2 = zero;
                int ix = kx, iy = ky;
                for (int k = 0; k < i; ++k) {
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

#undef A_
