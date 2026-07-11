/*
 * yher2 — kind10 port of OpenBLAS zher2.  Hermitian rank-2 update.
 *
 *   A := alpha * x * y^H + conj(alpha) * y * x^H + A
 *
 * alpha COMPLEX; A Hermitian (diagonal forced real on every column).
 *
 * Threaded path mirrors driver/level2/syr2_thread.c (HER2 variant): sqrt-
 * balanced contiguous partition (mask=7, min-width 16) on columns. Each
 * thread writes a disjoint column stripe; no reduction needed. UPPER
 * reverse-mapped, LOWER forward-mapped.
 *
 * Fortran ABI:  subroutine yher2(uplo, n, alpha, x, incx, y, incy, a, lda)
 */

#include <stddef.h>
#include <complex.h>
#include <ctype.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef _Complex long double C;
typedef long double R;

#define MULTI_THREAD_MINIMAL 16384
#define MAX_PARTITION_CPUS   256

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

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

void yher2_(const char *UPLO, const int *N, const C *ALPHA,
            const C *x, const int *INCX,
            const C *y, const int *INCY,
            C *a, const int *LDA)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    C alpha = *ALPHA;

    if (n == 0 || alpha == 0.0L) return;

    int upper = (toupper((unsigned char)*UPLO) == 'U');

    ptrdiff_t kx = (incx > 0) ? 0 : -(n - 1) * incx;
    ptrdiff_t ky = (incy > 0) ? 0 : -(n - 1) * incy;

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
                            C *aj = &A_(0, j);
                            if (x[j] != 0.0L || y[j] != 0.0L) {
                                C t1 = alpha * conjl(y[j]);
                                C t2 = conjl(alpha * x[j]);
                                for (ptrdiff_t i = 0; i < j; ++i)
                                    aj[i] += x[i] * t1 + y[i] * t2;
                                aj[j] = (R)creall(aj[j]) + (R)creall(x[j] * t1 + y[j] * t2);
                            } else {
                                aj[j] = (R)creall(aj[j]);
                            }
                        }
                    } else {
                        for (ptrdiff_t j = m_from; j < m_to; ++j) {
                            C *aj = &A_(0, j);
                            if (x[j] != 0.0L || y[j] != 0.0L) {
                                C t1 = alpha * conjl(y[j]);
                                C t2 = conjl(alpha * x[j]);
                                aj[j] = (R)creall(aj[j]) + (R)creall(x[j] * t1 + y[j] * t2);
                                for (ptrdiff_t i = j + 1; i < n; ++i)
                                    aj[i] += x[i] * t1 + y[i] * t2;
                            } else {
                                aj[j] = (R)creall(aj[j]);
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
                C *aj = &A_(0, j);
                if (x[j] != 0.0L || y[j] != 0.0L) {
                    C t1 = alpha * conjl(y[j]);
                    C t2 = conjl(alpha * x[j]);
                    for (ptrdiff_t i = 0; i < j; ++i)
                        aj[i] += x[i] * t1 + y[i] * t2;
                    aj[j] = (R)creall(aj[j]) + (R)creall(x[j] * t1 + y[j] * t2);
                } else {
                    aj[j] = (R)creall(aj[j]);
                }
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                C *aj = &A_(0, j);
                if (x[j] != 0.0L || y[j] != 0.0L) {
                    C t1 = alpha * conjl(y[j]);
                    C t2 = conjl(alpha * x[j]);
                    aj[j] = (R)creall(aj[j]) + (R)creall(x[j] * t1 + y[j] * t2);
                    for (ptrdiff_t i = j + 1; i < n; ++i)
                        aj[i] += x[i] * t1 + y[i] * t2;
                } else {
                    aj[j] = (R)creall(aj[j]);
                }
            }
        }
        return;
    }

    /* Strided path. */
    ptrdiff_t jx = kx, jy = ky;
    if (upper) {
        for (ptrdiff_t j = 0; j < n; ++j) {
            C *aj = &A_(0, j);
            if (x[jx] != 0.0L || y[jy] != 0.0L) {
                C t1 = alpha * conjl(y[jy]);
                C t2 = conjl(alpha * x[jx]);
                ptrdiff_t ix = kx, iy = ky;
                for (ptrdiff_t i = 0; i < j; ++i) {
                    aj[i] += x[ix] * t1 + y[iy] * t2;
                    ix += incx; iy += incy;
                }
                aj[j] = (R)creall(aj[j]) + (R)creall(x[jx] * t1 + y[jy] * t2);
            } else {
                aj[j] = (R)creall(aj[j]);
            }
            jx += incx; jy += incy;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            C *aj = &A_(0, j);
            if (x[jx] != 0.0L || y[jy] != 0.0L) {
                C t1 = alpha * conjl(y[jy]);
                C t2 = conjl(alpha * x[jx]);
                aj[j] = (R)creall(aj[j]) + (R)creall(x[jx] * t1 + y[jy] * t2);
                ptrdiff_t ix = jx, iy = jy;
                for (ptrdiff_t i = j + 1; i < n; ++i) {
                    ix += incx; iy += incy;
                    aj[i] += x[ix] * t1 + y[iy] * t2;
                }
            } else {
                aj[j] = (R)creall(aj[j]);
            }
            jx += incx; jy += incy;
        }
    }
}

#undef A_
