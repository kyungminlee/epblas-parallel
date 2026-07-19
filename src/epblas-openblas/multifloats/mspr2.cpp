/*
 * mspr2 — kind10 port of OpenBLAS dspr2.  Packed symmetric rank-2 update.
 *
 *   A := alpha * x * y^T + alpha * y * x^T + A   (A symmetric, packed)
 *
 * Threaded path mirrors driver/level2/spr2_thread.c: sqrt-balanced
 * contiguous column-stripe partition (mask=7, min-width 16). Disjoint
 * column writes; no reduction. UPPER reverse-mapped, LOWER forward-mapped.
 *
 * Fortran ABI:  subroutine mspr2(uplo, n, alpha, x, incx, y, incy, ap)
 */

#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <ctype.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#include "mblas_tuning.h"
#define MULTI_THREAD_MINIMAL MBLAS_MT_MIN_L2_NN

static int syr_partition(int upper, ptrdiff_t n, int nthreads,
                         int mask, int min_width,
                         ptrdiff_t *m_from, ptrdiff_t *m_to)
{
    ptrdiff_t w[MAX_PARTITION_CPUS];
    int num_cpu = 0;
    double dnum = (double)n * (double)n / (double)nthreads;
    ptrdiff_t i = 0;
    while (i < n) {
        ptrdiff_t width;
        if (nthreads - num_cpu > 1) {
            double di = (double)(n - i);
            double rad = di * di - dnum;
            if (rad > 0.0) width = (ptrdiff_t)(-sqrt(rad) + di);
            else           width = n - i;
            width = (width + mask) & ~(ptrdiff_t)mask;
            if (width < min_width) width = min_width;
            if (width > n - i)     width = n - i;
        } else {
            width = n - i;
        }
        w[num_cpu] = width;
        num_cpu++;
        i += width;
        if (num_cpu >= MAX_PARTITION_CPUS) break;
    }
    if (!upper) {
        ptrdiff_t cum = 0;
        for (int k = 0; k < num_cpu; ++k) {
            m_from[k] = cum;
            cum += w[k];
            m_to[k] = cum;
        }
    } else {
        ptrdiff_t cum = n;
        for (int k = 0; k < num_cpu; ++k) {
            m_to[k] = cum;
            cum -= w[k];
            m_from[k] = cum;
        }
    }
    return num_cpu;
}

extern "C" void mspr2_(const char *UPLO, const int *N, const T *ALPHA,
            const T *x, const int *INCX,
            const T *y, const int *INCY,
            T *ap)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    T alpha = *ALPHA;

    if (n == 0 || alpha == 0.0) return;

    int upper = (toupper((unsigned char)*UPLO) == 'U');

    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        int nthreads = 1;
        if (n * n > MULTI_THREAD_MINIMAL) nthreads = omp_get_max_threads();
        if (nthreads > MAX_PARTITION_CPUS) nthreads = MAX_PARTITION_CPUS;
        if (nthreads > 1) {
            ptrdiff_t mf[MAX_PARTITION_CPUS], mt[MAX_PARTITION_CPUS];
            int num_cpu = syr_partition(upper, n, nthreads, 7, 16, mf, mt);
            if (num_cpu > 1) {
                #pragma omp parallel num_threads(num_cpu)
                {
                    int t = omp_get_thread_num();
                    ptrdiff_t m_from = mf[t];
                    ptrdiff_t m_to   = mt[t];
                    if (upper) {
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            T t1 = alpha * y[j];
                            T t2 = alpha * x[j];
                            if (t1 == 0.0 && t2 == 0.0) continue;
                            T *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                            for (ptrdiff_t i = 0; i <= j; ++i)
                                aj[i] += x[i] * t1 + y[i] * t2;
                        }
                    } else {
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            T t1 = alpha * y[j];
                            T t2 = alpha * x[j];
                            if (t1 == 0.0 && t2 == 0.0) continue;
                            T *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                            for (ptrdiff_t i = j; i < n; ++i)
                                aj[i] += x[i] * t1 + y[i] * t2;
                        }
                    }
                }
                return;
            }
        }
#endif
        if (upper) {
            for (ptrdiff_t j = 0; j < n; ++j) {
                T t1 = alpha * y[j];
                T t2 = alpha * x[j];
                if (t1 == 0.0 && t2 == 0.0) continue;
                T *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                for (ptrdiff_t i = 0; i <= j; ++i)
                    aj[i] += x[i] * t1 + y[i] * t2;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                T t1 = alpha * y[j];
                T t2 = alpha * x[j];
                if (t1 == 0.0 && t2 == 0.0) continue;
                T *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                for (ptrdiff_t i = j; i < n; ++i)
                    aj[i] += x[i] * t1 + y[i] * t2;
            }
        }
    } else {
        ptrdiff_t jx = 0, jy = 0;
        if (upper) {
            for (ptrdiff_t j = 0; j < n; ++j) {
                T t1 = alpha * y[jy];
                T t2 = alpha * x[jx];
                if (t1 != 0.0 || t2 != 0.0) {
                    T *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                    ptrdiff_t ix = 0, iy = 0;
                    for (ptrdiff_t i = 0; i <= j; ++i) {
                        aj[i] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                T t1 = alpha * y[jy];
                T t2 = alpha * x[jx];
                if (t1 != 0.0 || t2 != 0.0) {
                    T *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                    ptrdiff_t ix = jx, iy = jy;
                    for (ptrdiff_t i = j; i < n; ++i) {
                        aj[i] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
            }
        }
    }
}
