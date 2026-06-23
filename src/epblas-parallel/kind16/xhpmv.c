/*
 * xhpmv — kind16 complex Hermitian packed matrix-vector multiply.
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

typedef __complex128 TC;

static inline TC cconj(TC z) { return conjq(z); }
typedef __float128 TR;

/* Thread the contiguous path once n*n exceeds this (matches OpenBLAS zhpmv's
 * MULTI_THREAD_MINIMAL): below it the serial sweep wins outright. Faithful
 * port of kind10 yhpmv. */
#define XHPMV_OMP_MIN 16384
#define XHPMV_MAX_CPUS 256


#ifdef _OPENMP
/* Sqrt-balanced contiguous column partition (OpenBLAS symv_partition, mask=3,
 * min_width=4): per-column work grows with j for UPPER, shrinks for LOWER. */
static ptrdiff_t xhpmv_partition(bool upper, ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
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
        if (num_cpu >= XHPMV_MAX_CPUS) break;
    }
    return num_cpu;
}
#endif

void xhpmv_core(
    char uplo,
    ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict ap,
    const TC *restrict x, ptrdiff_t incx,
    const TC *beta_,
    TC *restrict y, ptrdiff_t incy)
{
    const TC alpha = *alpha_, beta = *beta_;
    const TC zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;
    const char UPLO = blas_up(uplo);

    if (n == 0 || (alpha == zero && beta == one)) return;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(n - 1) * incy : 0;
        if (beta == zero) for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = zero; iy += incy; }
        else              for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = beta * y[iy]; iy += incy; }
    }
    if (alpha == zero) return;

    ptrdiff_t kk = 0;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if ((size_t)n * (size_t)n > XHPMV_OMP_MIN
            && blas_omp_should_thread()) {
            ptrdiff_t nthreads = blas_omp_max_threads();
            if (nthreads > XHPMV_MAX_CPUS) nthreads = XHPMV_MAX_CPUS;
            ptrdiff_t range[XHPMV_MAX_CPUS + 1];
            ptrdiff_t num_cpu = xhpmv_partition(UPLO == 'U', n, nthreads, range);
            TC *buf = (num_cpu > 1)
                ? (TC *)calloc((size_t)num_cpu * (size_t)n, sizeof(TC)) : NULL;
            if (buf) {
                #pragma omp parallel num_threads(num_cpu)
                {
                    ptrdiff_t t = omp_get_thread_num();
                    ptrdiff_t m_from = range[t], m_to = range[t + 1];
                    TC *restrict slot = buf + (size_t)t * (size_t)n;
                    if (UPLO == 'U') {
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            const TC *restrict aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                            const TC t1 = x[j];
                            TC t2 = zero;
                            for (ptrdiff_t i = 0; i < j; ++i) {
                                slot[i] += t1 * aj[i];
                                t2      += cconj(aj[i]) * x[i];
                            }
                            slot[j] += t1 * (TR)crealq(aj[j]) + t2;  /* real diag */
                        }
                    } else {
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            const TC *restrict aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                            const TC t1 = x[j];
                            TC t2 = zero;
                            slot[j] += t1 * (TR)crealq(aj[j]);       /* real diag */
                            for (ptrdiff_t i = j + 1; i < n; ++i) {
                                slot[i] += t1 * aj[i];
                                t2      += cconj(aj[i]) * x[i];
                            }
                            slot[j] += t2;
                        }
                    }
                }
                /* Range-limited reduction: UPPER thread touched [0,range[t+1]),
                 * LOWER thread [range[t],n). Fold into one slot, then alpha-AXPY. */
                if (UPLO == 'U') {
                    TC *restrict target = buf + (size_t)(num_cpu - 1) * (size_t)n;
                    for (ptrdiff_t t = 0; t < num_cpu - 1; ++t) {
                        const TC *restrict src = buf + (size_t)t * (size_t)n;
                        ptrdiff_t len = range[t + 1];
                        for (ptrdiff_t k = 0; k < len; ++k) target[k] += src[k];
                    }
                    for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                } else {
                    TC *restrict target = buf;
                    for (ptrdiff_t t = 1; t < num_cpu; ++t) {
                        const TC *restrict src = buf + (size_t)t * (size_t)n;
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
                const TC t1 = alpha * x[j];
                TC t2 = zero;
                ptrdiff_t k = kk;
                for (ptrdiff_t i = 0; i < j; ++i) {
                    y[i] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[i];
                    ++k;
                }
                y[j] += t1 * (TR)crealq(ap[kk + j]) + alpha * t2;
                kk += j + 1;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                const TC t1 = alpha * x[j];
                TC t2 = zero;
                y[j] += t1 * (TR)crealq(ap[kk]);
                ptrdiff_t k = kk + 1;
                for (ptrdiff_t i = j + 1; i < n; ++i) {
                    y[i] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[i];
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
                const TC t1 = alpha * x[jx];
                TC t2 = zero;
                ptrdiff_t ix = kx, iy = ky;
                for (ptrdiff_t k = kk; k < kk + j; ++k) {
                    y[iy] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] += t1 * (TR)crealq(ap[kk + j]) + alpha * t2;
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < n; ++j) {
                const TC t1 = alpha * x[jx];
                TC t2 = zero;
                y[jy] += t1 * (TR)crealq(ap[kk]);
                ptrdiff_t ix = jx, iy = jy;
                for (ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                    ix += incx; iy += incy;
                    y[iy] += t1 * ap[k];
                    t2 += cconj(ap[k]) * x[ix];
                }
                y[jy] += alpha * t2;
                jx += incx; jy += incy;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPMV(xhpmv, TC)
