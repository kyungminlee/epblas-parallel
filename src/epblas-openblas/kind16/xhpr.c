/*
 * xhpr — kind16 port of OpenBLAS zhpr.  Hermitian packed rank-1 update.
 *
 *   A := alpha * x * x^H + A   (alpha REAL; A Hermitian, packed)
 *
 * Threaded path mirrors driver/level2/spr_thread.c (HER variant): sqrt-
 * balanced contiguous column-stripe partition (mask=7, min-width 16).
 * Disjoint column writes — no reduction. UPPER reverse-mapped, LOWER
 * forward-mapped.
 *
 * Fortran ABI:  subroutine xhpr(uplo, n, alpha, x, incx, ap)
 */

#include <stddef.h>
#include <quadmath.h>
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

void xhpr_(const char *UPLO, const int *N, const R *ALPHA,
           const C *x, const int *INCX, C *ap,
           size_t uplo_len)
{
    (void)uplo_len;
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    R alpha = *ALPHA;

    if (n == 0 || alpha == 0.0Q) return;

    int upper = (toupper((unsigned char)*UPLO) == 'U');

    if (incx < 0) x -= (n - 1) * incx;

    if (incx == 1) {
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
                            C *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                            if (x[j] != 0.0Q) {
                                C temp = (C)alpha * conjq(x[j]);
                                for (ptrdiff_t i = 0; i < j; ++i) aj[i] += x[i] * temp;
                                aj[j] = (R)crealq(aj[j]) + (R)crealq(x[j] * temp);
                            } else {
                                aj[j] = (R)crealq(aj[j]);
                            }
                        }
                    } else {
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            C *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                            if (x[j] != 0.0Q) {
                                C temp = (C)alpha * conjq(x[j]);
                                aj[j] = (R)crealq(aj[j]) + (R)crealq(temp * x[j]);
                                for (ptrdiff_t i = j + 1; i < n; ++i) aj[i] += x[i] * temp;
                            } else {
                                aj[j] = (R)crealq(aj[j]);
                            }
                        }
                    }
                }
                return;
            }
        }
#endif
        if (upper) {
            for (ptrdiff_t j = 0; j < n; ++j) {
                C *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                if (x[j] != 0.0Q) {
                    C temp = (C)alpha * conjq(x[j]);
                    for (ptrdiff_t i = 0; i < j; ++i) aj[i] += x[i] * temp;
                    aj[j] = (R)crealq(aj[j]) + (R)crealq(x[j] * temp);
                } else {
                    aj[j] = (R)crealq(aj[j]);
                }
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                C *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                if (x[j] != 0.0Q) {
                    C temp = (C)alpha * conjq(x[j]);
                    aj[j] = (R)crealq(aj[j]) + (R)crealq(temp * x[j]);
                    for (ptrdiff_t i = j + 1; i < n; ++i) aj[i] += x[i] * temp;
                } else {
                    aj[j] = (R)crealq(aj[j]);
                }
            }
        }
    } else {
        ptrdiff_t jx = 0;
        if (upper) {
            for (ptrdiff_t j = 0; j < n; ++j) {
                C *aj = &ap[(size_t)j * (size_t)(j + 1) / 2];
                if (x[jx] != 0.0Q) {
                    C temp = (C)alpha * conjq(x[jx]);
                    ptrdiff_t ix = 0;
                    for (ptrdiff_t i = 0; i < j; ++i) { aj[i] += x[ix] * temp; ix += incx; }
                    aj[j] = (R)crealq(aj[j]) + (R)crealq(x[jx] * temp);
                } else {
                    aj[j] = (R)crealq(aj[j]);
                }
                jx += incx;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                C *aj = &ap[(size_t)j * (size_t)(2 * n - j - 1) / 2];
                if (x[jx] != 0.0Q) {
                    C temp = (C)alpha * conjq(x[jx]);
                    aj[j] = (R)crealq(aj[j]) + (R)crealq(temp * x[jx]);
                    ptrdiff_t ix = jx;
                    for (ptrdiff_t i = j + 1; i < n; ++i) { ix += incx; aj[i] += x[ix] * temp; }
                } else {
                    aj[j] = (R)crealq(aj[j]);
                }
                jx += incx;
            }
        }
    }
}
