/*
 * qsymv — kind16 symmetric matrix-vector multiply.
 *   y := alpha · A · x + beta · y     A symmetric
 *
 * Netlib two-pass pattern. Threaded (unit-stride) with a per-thread private
 * y accumulator: the two-pass column walk writes y[k] for the reflected
 * triangle, which races if threads share column ranges, so each thread sums
 * its column contributions into a private y_priv[N] and a final reduction
 * folds them into y. Quad is compute-bound under libquadmath so the O(N^2)
 * column work threads ~4x while the O(nthreads·N) reduction stays negligible.
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

typedef __float128 TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void qsymv_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict a, ptrdiff_t lda,
    const TR *restrict x, ptrdiff_t incx,
    const TR *beta_,
    TR *restrict y, ptrdiff_t incy)
{
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const TR zero = 0.0Q, one = 1.0Q;

    if (n == 0) return;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(n - 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            if (beta == zero) y[iy] = zero;
            else              y[iy] *= beta;
            iy += incy;
        }
    }
    if (alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const ptrdiff_t nthreads = blas_omp_max_threads();
        if (n >= QSYMV_OMP_MIN && blas_omp_should_thread()) {
            /* Parallel two-pass with per-thread private y accumulator;
             * schedule(static,1) interleaves columns to balance the
             * triangular per-column work (linear in N-j for L, j for U). */
            TR *y_priv_all = (TR *)calloc((size_t)nthreads * (size_t)n, sizeof(TR));
            if (y_priv_all) {
                #pragma omp parallel num_threads(nthreads)
                {
                    const ptrdiff_t tid = omp_get_thread_num();
                    TR *y_priv = &y_priv_all[(size_t)tid * n];  /* calloc-zeroed */

                    if (UPLO == 'L') {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < n; ++j) {
                            const TR temp1 = alpha * x[j];
                            TR temp2 = zero;
                            const TR *aj = &A_(0, j);
                            y_priv[j] += temp1 * aj[j];
                            for (ptrdiff_t k = j + 1; k < n; ++k) {
                                y_priv[k] += temp1 * aj[k];
                                temp2 += aj[k] * x[k];
                            }
                            y_priv[j] += alpha * temp2;
                        }
                    } else {
                        #pragma omp for schedule(static, 1)
                        for (ptrdiff_t j = 0; j < n; ++j) {
                            const TR temp1 = alpha * x[j];
                            TR temp2 = zero;
                            const TR *aj = &A_(0, j);
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
                    for (ptrdiff_t i = 0; i < n; ++i) {
                        TR s = zero;
                        for (ptrdiff_t t = 0; t < nthreads; ++t)
                            s += y_priv_all[(size_t)t * n + i];
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
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TR temp1 = alpha * x[i];
                TR temp2 = zero;
                const TR *ai = &A_(0, i);
                y[i] += temp1 * ai[i];
                for (ptrdiff_t k = i + 1; k < n; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += ai[k] * x[k];
                }
                y[i] += alpha * temp2;
            }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TR temp1 = alpha * x[i];
                TR temp2 = zero;
                const TR *ai = &A_(0, i);
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
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        if (UPLO == 'L') {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TR temp1 = alpha * x[jx];
                TR temp2 = zero;
                y[jy] += temp1 * A_(i, i);
                ptrdiff_t ix = jx, iy = jy;
                for (ptrdiff_t k = i + 1; k < n; ++k) {
                    ix += incx; iy += incy;
                    y[iy] += temp1 * A_(k, i);
                    temp2 += A_(k, i) * x[ix];
                }
                y[jy] += alpha * temp2;
                jx += incx; jy += incy;
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TR temp1 = alpha * x[jx];
                TR temp2 = zero;
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


EPBLAS_FACADE_SYMV(qsymv, TR)

#undef A_
