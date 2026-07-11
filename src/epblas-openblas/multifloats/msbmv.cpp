/*
 * msbmv — kind10 port of OpenBLAS dsbmv (interface/sbmv.c +
 * driver/level2/sbmv_thread.c).
 *
 *   y := alpha * A * x + beta * y  (A is N x N symmetric, K extra diagonals)
 *
 * Threading mirrors OpenBLAS dsbmv_thread:
 *   - nthreads = 1 if n < 200, else min(maxthreads, available)
 *   - if nthreads > 1: partition n columns across threads, each thread
 *     accumulates into a private y_priv buffer (size N each), final
 *     controller-side AXPY sums them all into y with the alpha applied.
 *   - Width partition: load-balanced sqrt formula when n < 2*k (work
 *     per column ~ min(i, k)); even partition (size-n/nthreads, min 4)
 *     when n >= 2*k.
 *
 * Band storage matches Fortran reference (and OpenBLAS):
 *   UPLO='U': ab[(k - j + i) + j*lda] = A(i,j) for max(0, j-k) <= i <= j
 *   UPLO='L': ab[(i - j) + j*lda]     = A(i,j) for j <= i <= min(n-1, j+k)
 *
 * Fortran ABI:
 *   subroutine msbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
 */

#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

/* Per-thread kernel: walks column range [n_from, n_to); writes into yp
 * (size n, zero-initialized by caller).  Computes the bare A*x for the
 * thread's columns (alpha NOT applied here — controller scales at the end). */
static void sbmv_kernel_U(ptrdiff_t n_from, ptrdiff_t n_to, ptrdiff_t n,
                          ptrdiff_t k, const T *a, ptrdiff_t lda,
                          const T *x, T *yp)
{
    const T *acol = &A_(0, n_from);
    for (ptrdiff_t i = n_from; i < n_to; ++i) {
        ptrdiff_t length = (i < k) ? i : k;
        /* AXPY length elements: yp[i-length..i-1] += x[i] * a[(k-length)..k-1, i] */
        T xi = x[i];
        const T *aoff = acol + (k - length);
        for (ptrdiff_t j = 0; j < length; ++j)
            yp[i - length + j] += xi * aoff[j];
        /* DOT length+1 elements: yp[i] += sum(a[(k-length)..k, i] * x[i-length..i]) */
        T s = 0.0;
        const T *acol_dot = acol + (k - length);
        const T *xoff = &x[i - length];
        for (ptrdiff_t j = 0; j < length + 1; ++j)
            s += acol_dot[j] * xoff[j];
        yp[i] += s;
        acol += lda;
    }
}

static void sbmv_kernel_L(ptrdiff_t n_from, ptrdiff_t n_to, ptrdiff_t n,
                          ptrdiff_t k, const T *a, ptrdiff_t lda,
                          const T *x, T *yp)
{
    const T *acol = &A_(0, n_from);
    for (ptrdiff_t i = n_from; i < n_to; ++i) {
        ptrdiff_t length = (n - i - 1 < k) ? n - i - 1 : k;
        /* AXPY length elements: yp[i+1..i+length] += x[i] * a[1..length, i] */
        T xi = x[i];
        for (ptrdiff_t j = 0; j < length; ++j)
            yp[i + 1 + j] += xi * acol[1 + j];
        /* DOT length+1 elements: yp[i] += sum(a[0..length, i] * x[i..i+length]) */
        T s = 0.0;
        const T *xoff = &x[i];
        for (ptrdiff_t j = 0; j < length + 1; ++j)
            s += acol[j] * xoff[j];
        yp[i] += s;
        acol += lda;
    }
}

/* Width-partition mirror of sbmv_thread.c, n < 2*k branch (sqrt formula). */
static void partition_sbmv_diag(int upper, ptrdiff_t n, int nthreads,
                                 ptrdiff_t *range_m)
{
    const int mask = 7;
    double dnum = (double)n * (double)n / (double)nthreads;
    ptrdiff_t i, width;
    int num_cpu = 0;
    if (!upper) {
        range_m[0] = 0;
        i = 0;
        while (i < n) {
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                if (di * di - dnum > 0.0) {
                    width = ((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask;
                } else {
                    width = n - i;
                }
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else {
                width = n - i;
            }
            range_m[num_cpu + 1] = range_m[num_cpu] + width;
            num_cpu++;
            i += width;
        }
    } else {
        range_m[nthreads] = n;
        i = 0;
        while (i < n) {
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                if (di * di - dnum > 0.0) {
                    width = ((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask;
                } else {
                    width = n - i;
                }
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else {
                width = n - i;
            }
            range_m[nthreads - num_cpu - 1] = range_m[nthreads - num_cpu] - width;
            num_cpu++;
            i += width;
        }
    }
    /* Fill remaining (unused) slots harmlessly. */
    for (int t = num_cpu + 1; t <= nthreads; ++t) {
        if (upper) range_m[nthreads - t] = range_m[nthreads - num_cpu];
        else       range_m[t] = range_m[num_cpu];
    }
}

static void partition_sbmv_even(ptrdiff_t n, int nthreads, ptrdiff_t *range_m)
{
    range_m[0] = 0;
    ptrdiff_t i = n;
    int num_cpu = 0;
    while (i > 0) {
        ptrdiff_t width = (i + nthreads - num_cpu - 1) / (nthreads - num_cpu);
        if (width < 4) width = 4;
        if (i < width) width = i;
        range_m[num_cpu + 1] = range_m[num_cpu] + width;
        num_cpu++;
        i -= width;
    }
    for (int t = num_cpu + 1; t <= nthreads; ++t) range_m[t] = range_m[num_cpu];
}

extern "C" void msbmv_(const char *UPLO, const int *N, const int *K, const T *ALPHA,
            const T *a, const int *LDA,
            const T *x, const int *INCX,
            const T *BETA, T *y, const int *INCY)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t k    = (ptrdiff_t)(*K);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    T alpha = *ALPHA;
    T beta  = *BETA;

    if (n == 0) return;
    int upper = (toupper((unsigned char)*UPLO) == 'U');

    /* Beta-scale y first (OpenBLAS does this via SCAL_K before the SMP branch). */
    if (beta != 1.0) {
        ptrdiff_t absy = incy < 0 ? -incy : incy;
        ptrdiff_t iy = 0;
        if (beta == 0.0) for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = 0.0; iy += absy; }
        else              for (ptrdiff_t i = 0; i < n; ++i) { y[iy] *= beta;  iy += absy; }
    }
    if (alpha == 0.0) return;

    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

#ifdef _OPENMP
    int nthreads = (n < 200) ? 1 : omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    if (nthreads > 1) {
        /* Mirror sbmv_thread: per-thread y buffer + final reduction. */
        const T *xptr = x;
        T *xbuf = NULL;
        if (incx != 1) {
            xbuf = (T *)malloc((size_t)n * sizeof(T));
            if (xbuf) {
                for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
                xptr = xbuf;
            } else {
                nthreads = 1;
            }
        }
        T *y_priv_all = (T *)calloc((size_t)nthreads * (size_t)n, sizeof(T));
        ptrdiff_t *range_m = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
        if (y_priv_all && range_m && nthreads > 1) {
            if (n < 2 * k) partition_sbmv_diag(upper, n, nthreads, range_m);
            else           partition_sbmv_even(n, nthreads, range_m);

            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                T *yp = &y_priv_all[(size_t)tid * (size_t)n];
                ptrdiff_t n_from = range_m[tid];
                ptrdiff_t n_to   = range_m[tid + 1];
                if (n_from < n_to) {
                    if (upper) sbmv_kernel_U(n_from, n_to, n, k, a, lda, xptr, yp);
                    else       sbmv_kernel_L(n_from, n_to, n, k, a, lda, xptr, yp);
                }
            }

            /* Controller AXPY: y += alpha * sum(y_priv across threads).
             * OpenBLAS sums into queue[0].sb first, then alpha-AXPY into y. */
            if (incy == 1) {
                for (ptrdiff_t i = 0; i < n; ++i) {
                    T s = 0.0;
                    for (int t = 0; t < nthreads; ++t)
                        s += y_priv_all[(size_t)t * (size_t)n + (size_t)i];
                    y[i] += alpha * s;
                }
            } else {
                ptrdiff_t iy = 0;
                for (ptrdiff_t i = 0; i < n; ++i) {
                    T s = 0.0;
                    for (int t = 0; t < nthreads; ++t)
                        s += y_priv_all[(size_t)t * (size_t)n + (size_t)i];
                    y[iy] += alpha * s;
                    iy += incy;
                }
            }
            free(range_m); free(y_priv_all);
            if (xbuf) free(xbuf);
            return;
        }
        free(range_m); free(y_priv_all);
        if (xbuf) free(xbuf);
        nthreads = 1;
    }

    /* Serial path — Fortran reference algorithm. */
    if (incx == 1 && incy == 1) {
        if (upper) {
            for (ptrdiff_t j = 0; j < n; ++j) {
                T temp1 = alpha * x[j];
                T temp2 = 0.0;
                ptrdiff_t l_lo = (j > k) ? j - k : 0;
                for (ptrdiff_t i = l_lo; i < j; ++i) {
                    T aij = a[(k - j + i) + (size_t)j * lda];
                    y[i]  += temp1 * aij;
                    temp2 += aij * x[i];
                }
                T ajj = a[k + (size_t)j * lda];
                y[j] += temp1 * ajj + alpha * temp2;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                T temp1 = alpha * x[j];
                T temp2 = 0.0;
                T ajj = a[(size_t)j * lda];
                y[j] += temp1 * ajj;
                ptrdiff_t l_hi = (j + k < n - 1) ? j + k : n - 1;
                for (ptrdiff_t i = j + 1; i <= l_hi; ++i) {
                    T aij = a[(i - j) + (size_t)j * lda];
                    y[i]  += temp1 * aij;
                    temp2 += aij * x[i];
                }
                y[j] += alpha * temp2;
            }
        }
        return;
    }

    /* Strided serial — same as before. */
    ptrdiff_t jx = 0, jy = 0;
    if (upper) {
        ptrdiff_t kx = 0, ky = 0;
        for (ptrdiff_t j = 0; j < n; ++j) {
            T temp1 = alpha * x[jx];
            T temp2 = 0.0;
            ptrdiff_t l_lo = (j > k) ? j - k : 0;
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t i = l_lo; i < j; ++i) {
                T aij = a[(k - j + i) + (size_t)j * lda];
                y[iy] += temp1 * aij;
                temp2 += aij * x[ix];
                ix += incx; iy += incy;
            }
            T ajj = a[k + (size_t)j * lda];
            y[jy] += temp1 * ajj + alpha * temp2;
            jx += incx; jy += incy;
            if (j >= k) { kx += incx; ky += incy; }
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T temp1 = alpha * x[jx];
            T temp2 = 0.0;
            T ajj = a[(size_t)j * lda];
            y[jy] += temp1 * ajj;
            ptrdiff_t l_hi = (j + k < n - 1) ? j + k : n - 1;
            ptrdiff_t ix = jx + incx, iy = jy + incy;
            for (ptrdiff_t i = j + 1; i <= l_hi; ++i) {
                T aij = a[(i - j) + (size_t)j * lda];
                y[iy] += temp1 * aij;
                temp2 += aij * x[ix];
                ix += incx; iy += incy;
            }
            y[jy] += alpha * temp2;
            jx += incx; jy += incy;
        }
    }
}

#undef A_
