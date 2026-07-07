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
#include <math.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define QSYMV_OMP_MIN 128
#define QSYMV_MAX_CPUS 256

typedef __float128 TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifdef _OPENMP
/* Sqrt-balanced contiguous column partition (OpenBLAS symv_partition,
 * mask=3, min_width=4): per-column work grows with j for UPPER, shrinks
 * for LOWER. Mirrors the kind10 yhemv fix and the packed twins. */
static ptrdiff_t qsymv_partition(int upper, ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
{
    const ptrdiff_t mask = 3, min_width = 4;
    const double dnum = (double)n * (double)n / (double)nthreads;
    ptrdiff_t num_cpu = 0;
    range[0] = 0;
    ptrdiff_t i = 0;
    while (i < n) {
        ptrdiff_t width;
        if (nthreads - num_cpu > 1) {
            if (upper) {
                double di = (double)i;
                width = (ptrdiff_t)(sqrt(di * di + dnum) - di);
            } else {
                double di = (double)(n - i);
                double rad = di * di - dnum;
                width = (rad > 0.0) ? (ptrdiff_t)(-sqrt(rad) + di) : (n - i);
            }
            width = (width + mask) & ~(ptrdiff_t)mask;
            if (width < min_width) width = min_width;
            if (width > n - i)     width = n - i;
        } else {
            width = n - i;
        }
        range[num_cpu + 1] = range[num_cpu] + width;
        num_cpu++;
        i += width;
        if (num_cpu >= QSYMV_MAX_CPUS) break;
    }
    return num_cpu;
}
#endif

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
        if (n >= QSYMV_OMP_MIN && blas_omp_should_thread() && nthreads > 1) {
            /* Sqrt-balanced contiguous column partition + per-thread
             * range-limited slot + serial fold (this routine's OpenBLAS
             * reference structure and the kind10 yhemv fix, task 15).
             * Replaces the old cyclic schedule(static,1) + rectangular
             * nthreads*n parallel reduction, whose re-read cost the quad
             * variant 7-17% at omp4 and GREW with N. */
            ptrdiff_t nt = (nthreads > QSYMV_MAX_CPUS) ? QSYMV_MAX_CPUS : nthreads;
            ptrdiff_t range[QSYMV_MAX_CPUS + 1];
            ptrdiff_t num_cpu = qsymv_partition(UPLO == 'U', n, nt, range);
            TR *buf = (num_cpu > 1)
                ? (TR *)calloc((size_t)num_cpu * (size_t)n, sizeof(TR)) : NULL;
            if (buf) {
                #pragma omp parallel for schedule(static, 1) num_threads(num_cpu)
                for (ptrdiff_t t = 0; t < num_cpu; ++t)
                {
                    TR *restrict slot = buf + (size_t)t * (size_t)n;
                    if (UPLO == 'U') {
                        /* Reversed thread->column assignment (task 14/15): the
                         * master (thread 0) takes the HIGH columns so its own
                         * slot 0 gains full [0,n) coverage and the serial fold
                         * below RMWs core-0-local memory (mirroring LOWER). */
                        ptrdiff_t r = num_cpu - 1 - t;
                        for (ptrdiff_t j = range[r]; j < range[r + 1]; ++j) {
                            const TR *restrict aj = &A_(0, j);
                            const TR t1 = x[j];
                            TR t2 = zero;
                            for (ptrdiff_t i = 0; i < j; ++i) {
                                slot[i] += t1 * aj[i];
                                t2      += aj[i] * x[i];
                            }
                            slot[j] += t1 * aj[j] + t2;
                        }
                    } else {
                        for (ptrdiff_t j = range[t]; j < range[t + 1]; ++j) {
                            const TR *restrict aj = &A_(0, j);
                            const TR t1 = x[j];
                            TR t2 = zero;
                            slot[j] += t1 * aj[j];
                            for (ptrdiff_t i = j + 1; i < n; ++i) {
                                slot[i] += t1 * aj[i];
                                t2      += aj[i] * x[i];
                            }
                            slot[j] += t2;
                        }
                    }
                }
                /* Range-limited serial fold into the master's full-coverage
                 * slot 0 (local), then one alpha-AXPY into y. */
                TR *restrict target = buf;
                if (UPLO == 'U') {
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const TR *restrict src = buf + (size_t)t * (size_t)n;
                        ptrdiff_t len = range[num_cpu - t];
                        for (ptrdiff_t k = 0; k < len; ++k) target[k] += src[k];
                    }
                } else {
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const TR *restrict src = buf + (size_t)t * (size_t)n;
                        for (ptrdiff_t k = range[t]; k < n; ++k) target[k] += src[k];
                    }
                }
                for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                free(buf);
                return;
            }
            free(buf);
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
