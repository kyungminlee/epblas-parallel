/*
 * xhpmv — kind16 port of OpenBLAS zhpmv.  Hermitian packed MV.
 *
 *   y := alpha * A * x + beta * y   (A Hermitian, packed storage)
 *
 * Threaded path mirrors driver/level2/spmv_thread.c (HEMV variant): same
 * sqrt-balanced contiguous partition + AXPY-chain reduce as xhemv. Only
 * the inner kernel uses packed column offsets — UPPER col j at ap[j*(j+1)/2]
 * (length j+1), LOWER col j at ap[j*(2n-j-1)/2] (length n-j). Diagonal
 * A(j,j) treated as real.
 *
 * Fortran ABI:  subroutine xhpmv(uplo, n, alpha, ap, x, incx, beta, y, incy)
 */

#include <stddef.h>
#include <quadmath.h>
#include <stdlib.h>
#include <complex.h>
#include <ctype.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __complex128 C;
typedef __float128 R;

#define MULTI_THREAD_MINIMAL 16384
#define MAX_PARTITION_CPUS   256

static int symv_partition(int upper, ptrdiff_t n, int nthreads,
                          int mask, int min_width, ptrdiff_t *range)
{
    int num_cpu = 0;
    double dnum = (double)n * (double)n / (double)nthreads;
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
                if (rad > 0.0) width = (ptrdiff_t)(-sqrt(rad) + di);
                else           width = n - i;
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
        if (num_cpu >= MAX_PARTITION_CPUS) break;
    }
    return num_cpu;
}

void xhpmv_(const char *UPLO, const int *N, const C *ALPHA,
            const C *ap,
            const C *x, const int *INCX,
            const C *BETA, C *y, const int *INCY,
            size_t uplo_len)
{
    (void)uplo_len;
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    C alpha = *ALPHA;
    C beta  = *BETA;

    if (n == 0 || (alpha == 0.0Q && beta == 1.0Q)) return;

    int upper = (toupper((unsigned char)*UPLO) == 'U');

    if (beta != 1.0Q) {
        ptrdiff_t absy = incy < 0 ? -incy : incy;
        ptrdiff_t iy = 0;
        if (beta == 0.0Q) for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = 0.0Q; iy += absy; }
        else              for (ptrdiff_t i = 0; i < n; ++i) { y[iy] *= beta;  iy += absy; }
    }

    if (alpha == 0.0Q) return;

    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        int nthreads = 1;
        if (n * n > MULTI_THREAD_MINIMAL) nthreads = omp_get_max_threads();
        if (nthreads > MAX_PARTITION_CPUS) nthreads = MAX_PARTITION_CPUS;
        if (nthreads > 1) {
            ptrdiff_t range[MAX_PARTITION_CPUS + 1];
            int num_cpu = symv_partition(upper, n, nthreads, 3, 4, range);
            if (num_cpu > 1) {
                C *buf = (C *)calloc((size_t)num_cpu * (size_t)n, sizeof(C));
                if (buf) {
                    #pragma omp parallel num_threads(num_cpu)
                    {
                        int t = omp_get_thread_num();
                        ptrdiff_t m_from = range[t];
                        ptrdiff_t m_to   = range[t + 1];
                        C *slot = buf + (size_t)t * (size_t)n;
                        if (upper) {
                            for (ptrdiff_t j = m_from; j < m_to; ++j) {
                                C t1 = x[j];
                                C t2 = 0.0Q;
                                const C *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                                for (ptrdiff_t i = 0; i < j; ++i) {
                                    slot[i] += t1 * aj[i];
                                    t2      += conjq(aj[i]) * x[i];
                                }
                                slot[j] += t1 * (R)crealq(aj[j]) + t2;
                            }
                        } else {
                            for (ptrdiff_t j = m_from; j < m_to; ++j) {
                                C t1 = x[j];
                                C t2 = 0.0Q;
                                const C *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                                slot[j] += t1 * (R)crealq(aj[j]);
                                for (ptrdiff_t i = j + 1; i < n; ++i) {
                                    slot[i] += t1 * aj[i];
                                    t2      += conjq(aj[i]) * x[i];
                                }
                                slot[j] += t2;
                            }
                        }
                    }
                    if (upper) {
                        C *target = buf + (size_t)(num_cpu - 1) * (size_t)n;
                        for (int i = 0; i < num_cpu - 1; ++i) {
                            const C *src = buf + (size_t)i * (size_t)n;
                            ptrdiff_t len = range[i + 1];
                            for (ptrdiff_t k = 0; k < len; ++k) target[k] += src[k];
                        }
                        for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                    } else {
                        C *target = buf;
                        for (int i = 1; i < num_cpu; ++i) {
                            const C *src = buf + (size_t)i * (size_t)n;
                            ptrdiff_t m_from = range[i];
                            for (ptrdiff_t k = m_from; k < n; ++k) target[k] += src[k];
                        }
                        for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                    }
                    free(buf);
                    return;
                }
            }
        }
#endif
        if (upper) {
            for (ptrdiff_t j = 0; j < n; ++j) {
                C t1 = alpha * x[j];
                C t2 = 0.0Q;
                const C *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                for (ptrdiff_t i = 0; i < j; ++i) {
                    y[i] += t1 * aj[i];
                    t2   += conjq(aj[i]) * x[i];
                }
                R ajj = (R)crealq(aj[j]);
                y[j] += t1 * ajj + alpha * t2;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                C t1 = alpha * x[j];
                C t2 = 0.0Q;
                const C *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                R ajj = (R)crealq(aj[j]);
                y[j] += t1 * ajj;
                for (ptrdiff_t i = j + 1; i < n; ++i) {
                    y[i] += t1 * aj[i];
                    t2   += conjq(aj[i]) * x[i];
                }
                y[j] += alpha * t2;
            }
        }
        return;
    }

    /* Strided. */
    ptrdiff_t kx = 0, ky = 0;
    ptrdiff_t jx = kx, jy = ky;
    if (upper) {
        for (ptrdiff_t j = 0; j < n; ++j) {
            C t1 = alpha * x[jx];
            C t2 = 0.0Q;
            const C *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t i = 0; i < j; ++i) {
                y[iy] += t1 * aj[i];
                t2    += conjq(aj[i]) * x[ix];
                ix += incx; iy += incy;
            }
            R ajj = (R)crealq(aj[j]);
            y[jy] += t1 * ajj + alpha * t2;
            jx += incx; jy += incy;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            C t1 = alpha * x[jx];
            C t2 = 0.0Q;
            const C *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
            R ajj = (R)crealq(aj[j]);
            y[jy] += t1 * ajj;
            ptrdiff_t ix = jx + incx, iy = jy + incy;
            for (ptrdiff_t i = j + 1; i < n; ++i) {
                y[iy] += t1 * aj[i];
                t2    += conjq(aj[i]) * x[ix];
                ix += incx; iy += incy;
            }
            y[jy] += alpha * t2;
            jx += incx; jy += incy;
        }
    }
}
