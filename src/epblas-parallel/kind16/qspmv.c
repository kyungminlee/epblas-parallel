/*
 * qspmv — kind16 (__float128) symmetric packed matrix-vector multiply.
 *   y := alpha*A*x + beta*y
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef __float128 T;

/* Thread the contiguous path once n*n exceeds this (matches dspmv's
 * MULTI_THREAD_MINIMAL): below it the serial sweep wins outright. Faithful
 * port of kind10 espmv. */
#define QSPMV_OMP_MIN 16384
#define QSPMV_MAX_CPUS 256


#ifdef _OPENMP
/* Sqrt-balanced contiguous column partition (OpenBLAS symv_partition, mask=3,
 * min_width=4). Per-column work grows with j for UPPER (length j+1) and shrinks
 * for LOWER (length n-j), so widths shrink / grow to equalize triangle area. */
static ptrdiff_t qspmv_partition(bool upper, ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
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
        if (num_cpu >= QSPMV_MAX_CPUS) break;
    }
    return num_cpu;
}
#endif

void qspmv_core(
    char uplo,
    ptrdiff_t n,
    const T *alpha_,
    const T *restrict ap,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0Q, one = 1.0Q;
    const char UPLO = blas_up(uplo);

    if (n == 0 || (alpha == zero && beta == one)) return;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(n - 1) * incy : 0;
        if (beta == zero) {
            for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = zero; iy += incy; }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (alpha == zero) return;

    ptrdiff_t kk = 0;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if ((size_t)n * (size_t)n > QSPMV_OMP_MIN
            && blas_omp_should_thread()) {
            ptrdiff_t nthreads = blas_omp_max_threads();
            if (nthreads > QSPMV_MAX_CPUS) nthreads = QSPMV_MAX_CPUS;
            ptrdiff_t range[QSPMV_MAX_CPUS + 1];
            ptrdiff_t num_cpu = qspmv_partition(UPLO == 'U', n, nthreads, range);
            T *buf = (num_cpu > 1)
                ? (T *)calloc((size_t)num_cpu * (size_t)n, sizeof(T)) : NULL;
            if (buf) {
                #pragma omp parallel num_threads(num_cpu)
                {
                    ptrdiff_t t = omp_get_thread_num();
                    ptrdiff_t m_from = range[t], m_to = range[t + 1];
                    T *restrict slot = buf + (size_t)t * (size_t)n;
                    if (UPLO == 'U') {
                        size_t k = (size_t)m_from * (size_t)(m_from + 1) / 2;
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            const T t1 = x[j];
                            T t2 = zero;
                            for (ptrdiff_t i = 0; i < j; ++i) {
                                slot[i] += t1 * ap[k];
                                t2      += ap[k] * x[i];
                                ++k;
                            }
                            slot[j] += t1 * ap[k] + t2;   /* diagonal */
                            ++k;
                        }
                    } else {
                        size_t k = (size_t)m_from * (size_t)(2 * n - m_from + 1) / 2;
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            const T t1 = x[j];
                            T t2 = zero;
                            slot[j] += t1 * ap[k];        /* diagonal */
                            ++k;
                            for (ptrdiff_t i = j + 1; i < n; ++i) {
                                slot[i] += t1 * ap[k];
                                t2      += ap[k] * x[i];
                                ++k;
                            }
                            slot[j] += t2;
                        }
                    }
                }
                /* Range-limited reduction: each UPPER thread touched [0,range[t+1]),
                 * each LOWER thread [range[t],n). Fold into one slot, then alpha-AXPY. */
                if (UPLO == 'U') {
                    T *restrict target = buf + (size_t)(num_cpu - 1) * (size_t)n;
                    for (ptrdiff_t t = 0; t < num_cpu - 1; ++t) {
                        const T *restrict src = buf + (size_t)t * (size_t)n;
                        ptrdiff_t len = range[t + 1];
                        for (ptrdiff_t k = 0; k < len; ++k) target[k] += src[k];
                    }
                    for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                } else {
                    T *restrict target = buf;
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const T *restrict src = buf + (size_t)t * (size_t)n;
                        for (ptrdiff_t k = range[t]; k < n; ++k) target[k] += src[k];
                    }
                    for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                }
                free(buf);
                return;
            }
            free(buf);
        }
#endif
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < n; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                ptrdiff_t k = kk;
                for (ptrdiff_t i = 0; i < j; ++i) {
                    y[i] += t1 * ap[k];
                    t2 += ap[k] * x[i];
                    ++k;
                }
                y[j] += t1 * ap[kk + j] + alpha * t2;
                kk += j + 1;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                y[j] += t1 * ap[kk];
                ptrdiff_t k = kk + 1;
                for (ptrdiff_t i = j + 1; i < n; ++i) {
                    y[i] += t1 * ap[k];
                    t2 += ap[k] * x[i];
                    ++k;
                }
                y[j] += alpha * t2;
                kk += n - j;
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        if (UPLO == 'U') {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < n; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                ptrdiff_t ix = kx, iy = ky;
                for (ptrdiff_t k = kk; k < kk + j; ++k) {
                    y[iy] += t1 * ap[k];
                    t2 += ap[k] * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] += t1 * ap[kk + j] + alpha * t2;
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < n; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                y[jy] += t1 * ap[kk];
                ptrdiff_t ix = jx, iy = jy;
                for (ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                    ix += incx; iy += incy;
                    y[iy] += t1 * ap[k];
                    t2 += ap[k] * x[ix];
                }
                y[jy] += alpha * t2;
                jx += incx; jy += incy;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPMV(qspmv, T)
