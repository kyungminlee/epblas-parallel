/*
 * esymv — kind10 (REAL(KIND=10)) symmetric matrix-vector multiply.
 *   y := alpha · A · x + beta · y    where A is N×N symmetric
 *
 * Uses Netlib DSYMV's two-pass pattern: for each i,
 *   temp1 = alpha · x(i)   (contributes to y(k) for k!=i via A column reads)
 *   temp2 = sum_k A(k,i) · x(k)   (dot-product accumulator)
 *   y(i) += temp1 · A(i,i) + alpha · temp2
 * Stride-1 walks of A by columns; same direction-flip trick as the
 * symm/hemm diagonal kernel.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ESYMV_OMP_MIN 128

typedef long double T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void esymv_(
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

    const T zero = 0.0L, one = 1.0L;

    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) for (int i = 0; i < N; ++i) y[i] = zero;
            else              for (int i = 0; i < N; ++i) y[i] *= beta;
        } else {
            int iy = (incy < 0) ? -(N - 1) * incy : 0;
            for (int i = 0; i < N; ++i) {
                if (beta == zero) y[iy] = zero;
                else              y[iy] *= beta;
                iy += incy;
            }
        }
    }

    if (alpha == zero) return;

    /* The unit-stride path: stride-1 column walks of A. */
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int nt = blas_omp_max_threads();
        const int use_omp = (N >= ESYMV_OMP_MIN && nt > 1 && !omp_in_parallel());
#else
        const int use_omp = 0;
        const int nt = 1;
#endif
        if (use_omp) {
            /* Parallel two-pass with per-thread private y accumulator.
             *
             * The Netlib two-pass form walks A column-by-column (stride-1)
             * and on each column j writes y[k] for k > j (L) or k < j (U),
             * which races if multiple threads share column ranges. Fix:
             * each thread gets a private y_priv[N], accumulates its own
             * column contributions, then a final reduction sums all
             * y_priv[t] into y.
             *
             * schedule(static, 1) interleaves columns across threads to
             * balance the triangular work (per-column work is linear in
             * (N - j) for L, j for U). */
            T *y_priv_all = (T *)aligned_alloc(64,
                (((size_t)nt * N * sizeof(T)) + 63) & ~(size_t)63);
            if (y_priv_all) {
#ifdef _OPENMP
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
                    /* Implicit barrier at end of the `omp for` ensures
                     * every thread's y_priv slice is fully written
                     * before the reduction begins reading. */

                    #pragma omp for schedule(static)
                    for (int i = 0; i < N; ++i) {
                        T s = zero;
                        for (int t = 0; t < nt; ++t)
                            s += y_priv_all[(size_t)t * N + i];
                        y[i] += s;
                    }
                }
#endif
                free(y_priv_all);
                return;
            }
            /* aligned_alloc failed — fall through to serial. */
        }
        if (UPLO == 'L') {
            /* Iterate i forward; the inner k loop covers k = i..N-1
             * (stored lower triangle). Uses A_(k, i) (stride-1 in k). */
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
            /* UPLO='U': iterate i forward; inner k = 0..i-1
             * (stored upper triangle). */
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
