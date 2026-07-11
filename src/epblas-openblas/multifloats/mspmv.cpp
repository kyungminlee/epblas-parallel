/*
 * mspmv — kind10 port of OpenBLAS dspmv.  Packed symmetric matrix-vector.
 *
 *   y := alpha * A * x + beta * y    (A is N x N symmetric, packed)
 *
 * Threaded path mirrors driver/level2/spmv_thread.c (structurally identical
 * to symv_thread.c): sqrt-balanced contiguous partition (mask=3, min-width
 * 4), per-thread slot, AXPY-chain reduce, final alpha-AXPY into y. Only
 * the inner kernel changes — column j of A is at the packed offset
 * j*(j+1)/2 (UPPER, length j+1) or j*(2n-j-1)/2 (LOWER, length n-j).
 *
 * Fortran ABI:  subroutine mspmv(uplo, n, alpha, ap, x, incx, beta, y, incy)
 */

#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <stdlib.h>
#include <ctype.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

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

extern "C" void mspmv_(const char *UPLO, const int *N, const T *ALPHA,
            const T *ap,
            const T *x, const int *INCX,
            const T *BETA, T *y, const int *INCY)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    T alpha = *ALPHA;
    T beta  = *BETA;

    if (n == 0) return;

    int upper = (toupper((unsigned char)*UPLO) == 'U');

    if (beta != 1.0) {
        ptrdiff_t absy = incy < 0 ? -incy : incy;
        ptrdiff_t iy = 0;
        if (beta == 0.0) for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = 0.0; iy += absy; }
        else              for (ptrdiff_t i = 0; i < n; ++i) { y[iy] *= beta;  iy += absy; }
    }

    if (alpha == 0.0) return;

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
                T *buf = (T *)calloc((size_t)num_cpu * (size_t)n, sizeof(T));
                if (buf) {
                    #pragma omp parallel num_threads(num_cpu)
                    {
                        int t = omp_get_thread_num();
                        ptrdiff_t m_from = range[t];
                        ptrdiff_t m_to   = range[t + 1];
                        T *slot = buf + (size_t)t * (size_t)n;
                        if (upper) {
                            for (ptrdiff_t j = m_from; j < m_to; ++j) {
                                T temp1 = x[j];
                                T temp2 = 0.0;
                                const T *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                                for (ptrdiff_t i = 0; i < j; ++i) {
                                    slot[i] += temp1 * aj[i];
                                    temp2   += aj[i] * x[i];
                                }
                                slot[j] += temp1 * aj[j] + temp2;
                            }
                        } else {
                            for (ptrdiff_t j = m_from; j < m_to; ++j) {
                                T temp1 = x[j];
                                T temp2 = 0.0;
                                const T *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                                slot[j] += temp1 * aj[j];
                                for (ptrdiff_t i = j + 1; i < n; ++i) {
                                    slot[i] += temp1 * aj[i];
                                    temp2   += aj[i] * x[i];
                                }
                                slot[j] += temp2;
                            }
                        }
                    }
                    if (upper) {
                        T *target = buf + (size_t)(num_cpu - 1) * (size_t)n;
                        for (int i = 0; i < num_cpu - 1; ++i) {
                            const T *src = buf + (size_t)i * (size_t)n;
                            ptrdiff_t len = range[i + 1];
                            for (ptrdiff_t k = 0; k < len; ++k) target[k] += src[k];
                        }
                        for (ptrdiff_t k = 0; k < n; ++k) y[k] += alpha * target[k];
                    } else {
                        T *target = buf;
                        for (int i = 1; i < num_cpu; ++i) {
                            const T *src = buf + (size_t)i * (size_t)n;
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
                T temp1 = alpha * x[j];
                T temp2 = 0.0;
                const T *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                for (ptrdiff_t i = 0; i < j; ++i) {
                    y[i]  += temp1 * aj[i];
                    temp2 += aj[i] * x[i];
                }
                y[j] += temp1 * aj[j] + alpha * temp2;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                T temp1 = alpha * x[j];
                T temp2 = 0.0;
                const T *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                y[j] += temp1 * aj[j];
                for (ptrdiff_t i = j + 1; i < n; ++i) {
                    y[i]  += temp1 * aj[i];
                    temp2 += aj[i] * x[i];
                }
                y[j] += alpha * temp2;
            }
        }
        return;
    }

    /* Strided path. */
    ptrdiff_t kx = 0, ky = 0;
    ptrdiff_t jx = kx, jy = ky;
    if (upper) {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T temp1 = alpha * x[jx];
            T temp2 = 0.0;
            const T *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t i = 0; i < j; ++i) {
                y[iy] += temp1 * aj[i];
                temp2 += aj[i] * x[ix];
                ix += incx; iy += incy;
            }
            y[jy] += temp1 * aj[j] + alpha * temp2;
            jx += incx; jy += incy;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T temp1 = alpha * x[jx];
            T temp2 = 0.0;
            const T *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
            y[jy] += temp1 * aj[j];
            ptrdiff_t ix = jx + incx, iy = jy + incy;
            for (ptrdiff_t i = j + 1; i < n; ++i) {
                y[iy] += temp1 * aj[i];
                temp2 += aj[i] * x[ix];
                ix += incx; iy += incy;
            }
            y[jy] += alpha * temp2;
            jx += incx; jy += incy;
        }
    }
}
