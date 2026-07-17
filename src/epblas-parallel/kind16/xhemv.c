/*
 * xhemv — kind16 Hermitian matrix-vector multiply.
 *   y := alpha · A · x + beta · y     A Hermitian
 * Diagonal of A treated as real (Hermitian convention).
 */

#include <stddef.h>
#include <stdbool.h>
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

#define XHEMV_OMP_MIN 128
#define XHEMV_MAX_CPUS 256

typedef __complex128 TC;



static inline TC cconj(TC z) { return conjq(z); }
#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifdef _OPENMP
/* Sqrt-balanced contiguous column partition (OpenBLAS symv_partition,
 * mask=3, min_width=4): per-column work grows with j for UPPER, shrinks
 * for LOWER. Mirrors the kind10 yhemv fix and the packed twins. */
static ptrdiff_t xhemv_partition(bool upper, ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
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
        if (num_cpu >= XHEMV_MAX_CPUS) break;
    }
    return num_cpu;
}
#endif

void xhemv_core(
    char uplo,
    ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict a, ptrdiff_t lda,
    const TC *restrict x, ptrdiff_t incx,
    const TC *beta_,
    TC *restrict y, ptrdiff_t incy)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const TC zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;

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
        if (n >= XHEMV_OMP_MIN && blas_omp_should_thread() && nthreads > 1) {
            /* Sqrt-balanced contiguous column partition + per-thread
             * range-limited slot + serial fold (the kind10 yhemv fix, task 15).
             * Replaces the old cyclic schedule(static,1) + rectangular
             * nthreads*n parallel reduction, whose re-read cost the complex
             * quad variant 5-11% at omp4 and GREW with N. */
            ptrdiff_t nt = (nthreads > XHEMV_MAX_CPUS) ? XHEMV_MAX_CPUS : nthreads;
            ptrdiff_t range[XHEMV_MAX_CPUS + 1];
            ptrdiff_t num_cpu = xhemv_partition(UPLO == 'U', n, nt, range);
            TC *buf = (num_cpu > 1)
                ? (TC *)calloc((size_t)num_cpu * (size_t)n, sizeof(TC)) : NULL;
            if (buf) {
                #pragma omp parallel for schedule(static, 1) num_threads(num_cpu)
                for (ptrdiff_t t = 0; t < num_cpu; ++t)
                {
                    TC *restrict slot = buf + (size_t)t * (size_t)n;
                    if (UPLO == 'U') {
                        /* Reversed thread->column assignment (task 14/15): the
                         * master (thread 0) takes the HIGH columns so its own
                         * slot 0 gains full [0,n) coverage and the serial fold
                         * below RMWs core-0-local memory (mirroring LOWER). */
                        ptrdiff_t r = num_cpu - 1 - t;
                        for (ptrdiff_t j = range[r]; j < range[r + 1]; ++j) {
                            const TC *restrict aj = &A_(0, j);
                            const TC t1 = x[j];
                            TC t2 = zero;
                            for (ptrdiff_t i = 0; i < j; ++i) {
                                slot[i] += t1 * aj[i];
                                t2      += cconj(aj[i]) * x[i];
                            }
                            slot[j] += t1 * crealq(aj[j]) + t2;   /* real diag */
                        }
                    } else {
                        for (ptrdiff_t j = range[t]; j < range[t + 1]; ++j) {
                            const TC *restrict aj = &A_(0, j);
                            const TC t1 = x[j];
                            TC t2 = zero;
                            slot[j] += t1 * crealq(aj[j]);        /* real diag */
                            for (ptrdiff_t i = j + 1; i < n; ++i) {
                                slot[i] += t1 * aj[i];
                                t2      += cconj(aj[i]) * x[i];
                            }
                            slot[j] += t2;
                        }
                    }
                }
                /* Range-limited serial fold into the master's full-coverage
                 * slot 0 (local), then one alpha-AXPY into y. */
                TC *restrict target = buf;
                if (UPLO == 'U') {
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const TC *restrict src = buf + (size_t)t * (size_t)n;
                        ptrdiff_t len = range[num_cpu - t];
                        for (ptrdiff_t k = 0; k < len; ++k) target[k] += src[k];
                    }
                } else {
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const TC *restrict src = buf + (size_t)t * (size_t)n;
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
                const TC temp1 = alpha * x[i];
                TC temp2 = zero;
                const TC *ai = &A_(0, i);
                y[i] += temp1 * crealq(ai[i]);
                for (ptrdiff_t k = i + 1; k < n; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += cconj(ai[k]) * x[k];
                }
                y[i] += alpha * temp2;
            }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TC temp1 = alpha * x[i];
                TC temp2 = zero;
                const TC *ai = &A_(0, i);
                for (ptrdiff_t k = 0; k < i; ++k) {
                    y[k]  += temp1 * ai[k];
                    temp2 += cconj(ai[k]) * x[k];
                }
                y[i] += temp1 * crealq(ai[i]) + alpha * temp2;
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        if (UPLO == 'L') {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TC temp1 = alpha * x[kx + (ptrdiff_t)i * incx];
                TC temp2 = zero;
                y[ky + (ptrdiff_t)i * incy] += temp1 * crealq(A_(i, i));
                for (ptrdiff_t k = i + 1; k < n; ++k) {
                    y[ky + (ptrdiff_t)k * incy] += temp1 * A_(k, i);
                    temp2 += cconj(A_(k, i)) * x[kx + (ptrdiff_t)k * incx];
                }
                y[ky + (ptrdiff_t)i * incy] += alpha * temp2;
            }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) {
                const TC temp1 = alpha * x[kx + (ptrdiff_t)i * incx];
                TC temp2 = zero;
                for (ptrdiff_t k = 0; k < i; ++k) {
                    y[ky + (ptrdiff_t)k * incy] += temp1 * A_(k, i);
                    temp2 += cconj(A_(k, i)) * x[kx + (ptrdiff_t)k * incx];
                }
                y[ky + (ptrdiff_t)i * incy] += temp1 * crealq(A_(i, i)) + alpha * temp2;
            }
        }
    }
}

#undef A_

EPBLAS_FACADE_SYMV(xhemv, TC)
