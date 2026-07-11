/*
 * yhemv — kind10 port of OpenBLAS zhemv.  Hermitian MV.
 *
 *   y := alpha * A * x + beta * y    (A is N x N Hermitian)
 *
 * Threaded path mirrors driver/level2/symv_thread.c (HEMV preprocess
 * variant): sqrt-balanced contiguous partition (mask=3, min-width 4),
 * per-thread slot, AXPY-chain reduce, final alpha-AXPY into y. Hermitian
 * structure changes only the kernel: AXPY part uses A(i,j) directly, DOT
 * (reflection) part uses conj(A(i,j)); diagonal A(j,j) treated as real.
 *
 * Fortran ABI:  subroutine yhemv(uplo, n, alpha, a, lda, x, incx, beta, y, incy)
 */

#include <stddef.h>
#include <stdlib.h>
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

void yhemv_(const char *UPLO, const int *N, const C *ALPHA,
            const C *a, const int *LDA,
            const C *x, const int *INCX,
            const C *BETA, C *y, const int *INCY)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    C alpha = *ALPHA;
    C beta  = *BETA;

    if (n == 0 || (alpha == 0.0L && beta == 1.0L)) return;

    int upper = (toupper((unsigned char)*UPLO) == 'U');

    if (beta != 1.0L) {
        ptrdiff_t absy = incy < 0 ? -incy : incy;
        ptrdiff_t iy = 0;
        if (beta == 0.0L) for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = 0.0L; iy += absy; }
        else              for (ptrdiff_t i = 0; i < n; ++i) { y[iy] *= beta;  iy += absy; }
    }

    if (alpha == 0.0L) return;

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
                                C t2 = 0.0L;
                                const C *aj = &A_(0, j);
                                for (ptrdiff_t i = 0; i < j; ++i) {
                                    slot[i] += t1 * aj[i];
                                    t2      += conjl(aj[i]) * x[i];
                                }
                                slot[j] += t1 * (R)creall(aj[j]) + t2;
                            }
                        } else {
                            for (ptrdiff_t j = m_from; j < m_to; ++j) {
                                C t1 = x[j];
                                C t2 = 0.0L;
                                const C *aj = &A_(0, j);
                                slot[j] += t1 * (R)creall(aj[j]);
                                for (ptrdiff_t i = j + 1; i < n; ++i) {
                                    slot[i] += t1 * aj[i];
                                    t2      += conjl(aj[i]) * x[i];
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
                C t2 = 0.0L;
                const C *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < j; ++i) {
                    y[i] += t1 * aj[i];
                    t2   += conjl(aj[i]) * x[i];
                }
                y[j] += t1 * (R)creall(aj[j]) + alpha * t2;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                C t1 = alpha * x[j];
                C t2 = 0.0L;
                const C *aj = &A_(0, j);
                y[j] += t1 * (R)creall(aj[j]);
                for (ptrdiff_t i = j + 1; i < n; ++i) {
                    y[i] += t1 * aj[i];
                    t2   += conjl(aj[i]) * x[i];
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
            C t2 = 0.0L;
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t i = 0; i < j; ++i) {
                y[iy] += t1 * A_(i, j);
                t2    += conjl(A_(i, j)) * x[ix];
                ix += incx; iy += incy;
            }
            y[jy] += t1 * (R)creall(A_(j, j)) + alpha * t2;
            jx += incx; jy += incy;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            C t1 = alpha * x[jx];
            C t2 = 0.0L;
            y[jy] += t1 * (R)creall(A_(j, j));
            ptrdiff_t ix = jx + incx, iy = jy + incy;
            for (ptrdiff_t i = j + 1; i < n; ++i) {
                y[iy] += t1 * A_(i, j);
                t2    += conjl(A_(i, j)) * x[ix];
                ix += incx; iy += incy;
            }
            y[jy] += alpha * t2;
            jx += incx; jy += incy;
        }
    }
}

#undef A_
