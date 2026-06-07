/*
 * whbmv — kind10 port of OpenBLAS zhbmv (interface/zhbmv.c +
 * driver/level2/sbmv_thread.c with HEMV define).
 *
 *   y := alpha * A * x + beta * y    (A Hermitian banded, K extra diagonals)
 *
 * Threading mirrors OpenBLAS zhbmv_thread (HEMV variant of sbmv_thread):
 *   - Off-diagonal band: AXPY (unconjugated) for the column AXPY phase
 *   - Off-diagonal band: DOTC (conjugating band entries) for the dot phase
 *   - Diagonal element: REAL(A[j,j]) only — imaginary part discarded
 *
 * Band storage matches esbmv (identical to OpenBLAS).
 *
 * Fortran ABI:  subroutine whbmv(uplo, n, k, alpha, a, lda, x, incx, beta, y, incy)
 */

#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <complex>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using C = std::complex<multifloats::float64x2>;
typedef std::complex<multifloats::float64x2> C;
typedef multifloats::float64x2 R;

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

static void hbmv_kernel_U(ptrdiff_t n_from, ptrdiff_t n_to, ptrdiff_t n,
                          ptrdiff_t k, const C *a, ptrdiff_t lda,
                          const C *x, C *yp)
{
    const C *acol = &A_(0, n_from);
    for (ptrdiff_t i = n_from; i < n_to; ++i) {
        ptrdiff_t length = (i < k) ? i : k;
        C xi = x[i];
        const C *aoff = acol + (k - length);
        /* AXPY (unconjugated): yp[i-length..i-1] += x[i] * a[(k-length)..k-1, i] */
        for (ptrdiff_t j = 0; j < length; ++j)
            yp[i - length + j] += xi * aoff[j];
        /* DOTC over band above diagonal: sum_{j=0..length-1} conj(a[k-length+j, i]) * x[i-length+j] */
        C s = C(0.0, 0.0);
        const C *xoff = &x[i - length];
        for (ptrdiff_t j = 0; j < length; ++j)
            s += std::conj(aoff[j]) * xoff[j];
        /* Diagonal: REAL(a[k, i]) * x[i] */
        R ajj_re = (R)(acol[k]).real();
        yp[i] += s + ajj_re * xi;
        acol += lda;
    }
}

static void hbmv_kernel_L(ptrdiff_t n_from, ptrdiff_t n_to, ptrdiff_t n,
                          ptrdiff_t k, const C *a, ptrdiff_t lda,
                          const C *x, C *yp)
{
    const C *acol = &A_(0, n_from);
    for (ptrdiff_t i = n_from; i < n_to; ++i) {
        ptrdiff_t length = (n - i - 1 < k) ? n - i - 1 : k;
        C xi = x[i];
        /* AXPY (unconjugated): yp[i+1..i+length] += x[i] * a[1..length, i] */
        for (ptrdiff_t j = 0; j < length; ++j)
            yp[i + 1 + j] += xi * acol[1 + j];
        /* DOTC over band below diagonal: sum_{j=0..length-1} conj(a[1+j, i]) * x[i+1+j] */
        C s = C(0.0, 0.0);
        const C *xoff = &x[i + 1];
        for (ptrdiff_t j = 0; j < length; ++j)
            s += std::conj(acol[1 + j]) * xoff[j];
        /* Diagonal: REAL(a[0, i]) * x[i] */
        R ajj_re = (R)(acol[0]).real();
        yp[i] += s + ajj_re * xi;
        acol += lda;
    }
}

static void partition_diag(int upper, ptrdiff_t n, int nthreads, ptrdiff_t *range_m)
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
                } else width = n - i;
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else width = n - i;
            range_m[num_cpu + 1] = range_m[num_cpu] + width;
            num_cpu++; i += width;
        }
    } else {
        range_m[nthreads] = n;
        i = 0;
        while (i < n) {
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                if (di * di - dnum > 0.0) {
                    width = ((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask;
                } else width = n - i;
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else width = n - i;
            range_m[nthreads - num_cpu - 1] = range_m[nthreads - num_cpu] - width;
            num_cpu++; i += width;
        }
    }
    for (int t = num_cpu + 1; t <= nthreads; ++t) {
        if (upper) range_m[nthreads - t] = range_m[nthreads - num_cpu];
        else       range_m[t] = range_m[num_cpu];
    }
}

static void partition_even(ptrdiff_t n, int nthreads, ptrdiff_t *range_m)
{
    range_m[0] = 0;
    ptrdiff_t i = n;
    int num_cpu = 0;
    while (i > 0) {
        ptrdiff_t width = (i + nthreads - num_cpu - 1) / (nthreads - num_cpu);
        if (width < 4) width = 4;
        if (i < width) width = i;
        range_m[num_cpu + 1] = range_m[num_cpu] + width;
        num_cpu++; i -= width;
    }
    for (int t = num_cpu + 1; t <= nthreads; ++t) range_m[t] = range_m[num_cpu];
}

extern "C" void whbmv_(const char *UPLO, const int *N, const int *K, const C *ALPHA,
            const C *a, const int *LDA,
            const C *x, const int *INCX,
            const C *BETA, C *y, const int *INCY,
            size_t uplo_len)
{
    (void)uplo_len;
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t k    = (ptrdiff_t)(*K);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    C alpha = *ALPHA;
    C beta  = *BETA;

    if (n == 0 || (alpha == C(0.0, 0.0) && beta == C(1.0, 0.0))) return;
    int upper = (toupper((unsigned char)*UPLO) == 'U');

    if (beta != C(1.0, 0.0)) {
        ptrdiff_t absy = incy < 0 ? -incy : incy;
        ptrdiff_t iy = 0;
        if (beta == C(0.0, 0.0)) for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = C(0.0, 0.0); iy += absy; }
        else              for (ptrdiff_t i = 0; i < n; ++i) { y[iy] *= beta;  iy += absy; }
    }
    if (alpha == C(0.0, 0.0)) return;

    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

#ifdef _OPENMP
    int nthreads = (n < 200) ? 1 : omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    if (nthreads > 1) {
        const C *xptr = x;
        C *xbuf = NULL;
        if (incx != 1) {
            xbuf = (C *)malloc((size_t)n * sizeof(C));
            if (xbuf) {
                for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
                xptr = xbuf;
            } else nthreads = 1;
        }
        C *y_priv_all = (C *)calloc((size_t)nthreads * (size_t)n, sizeof(C));
        ptrdiff_t *range_m = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
        if (y_priv_all && range_m && nthreads > 1) {
            if (n < 2 * k) partition_diag(upper, n, nthreads, range_m);
            else           partition_even(n, nthreads, range_m);

            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                C *yp = &y_priv_all[(size_t)tid * (size_t)n];
                ptrdiff_t n_from = range_m[tid], n_to = range_m[tid + 1];
                if (n_from < n_to) {
                    if (upper) hbmv_kernel_U(n_from, n_to, n, k, a, lda, xptr, yp);
                    else       hbmv_kernel_L(n_from, n_to, n, k, a, lda, xptr, yp);
                }
            }
            if (incy == 1) {
                for (ptrdiff_t i = 0; i < n; ++i) {
                    C s = C(0.0, 0.0);
                    for (int t = 0; t < nthreads; ++t)
                        s += y_priv_all[(size_t)t * (size_t)n + (size_t)i];
                    y[i] += alpha * s;
                }
            } else {
                ptrdiff_t iy = 0;
                for (ptrdiff_t i = 0; i < n; ++i) {
                    C s = C(0.0, 0.0);
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

    /* Serial fallback (Hermitian: REAL diagonal, conj for the implicit half). */
    if (incx == 1 && incy == 1) {
        if (upper) {
            for (ptrdiff_t j = 0; j < n; ++j) {
                C t1 = alpha * x[j];
                C t2 = C(0.0, 0.0);
                ptrdiff_t i_lo = (j > k) ? j - k : 0;
                for (ptrdiff_t i = i_lo; i < j; ++i) {
                    C aij = a[(k - j + i) + (size_t)j * lda];
                    y[i] += t1 * aij;
                    t2   += std::conj(aij) * x[i];
                }
                R ajj = (R)(a[k + (size_t)j * lda]).real();
                y[j] += t1 * ajj + alpha * t2;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                C t1 = alpha * x[j];
                C t2 = C(0.0, 0.0);
                R ajj = (R)(a[(size_t)j * lda]).real();
                y[j] += t1 * ajj;
                ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                for (ptrdiff_t i = j + 1; i <= i_hi; ++i) {
                    C aij = a[(i - j) + (size_t)j * lda];
                    y[i] += t1 * aij;
                    t2   += std::conj(aij) * x[i];
                }
                y[j] += alpha * t2;
            }
        }
        return;
    }

    /* Strided serial. */
    ptrdiff_t jx = 0, jy = 0;
    if (upper) {
        ptrdiff_t kx = 0, ky = 0;
        for (ptrdiff_t j = 0; j < n; ++j) {
            C t1 = alpha * x[jx];
            C t2 = C(0.0, 0.0);
            ptrdiff_t i_lo = (j > k) ? j - k : 0;
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t i = i_lo; i < j; ++i) {
                C aij = a[(k - j + i) + (size_t)j * lda];
                y[iy] += t1 * aij;
                t2    += std::conj(aij) * x[ix];
                ix += incx; iy += incy;
            }
            R ajj = (R)(a[k + (size_t)j * lda]).real();
            y[jy] += t1 * ajj + alpha * t2;
            jx += incx; jy += incy;
            if (j >= k) { kx += incx; ky += incy; }
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            C t1 = alpha * x[jx];
            C t2 = C(0.0, 0.0);
            R ajj = (R)(a[(size_t)j * lda]).real();
            y[jy] += t1 * ajj;
            ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
            ptrdiff_t ix = jx + incx, iy = jy + incy;
            for (ptrdiff_t i = j + 1; i <= i_hi; ++i) {
                C aij = a[(i - j) + (size_t)j * lda];
                y[iy] += t1 * aij;
                t2    += std::conj(aij) * x[ix];
                ix += incx; iy += incy;
            }
            y[jy] += alpha * t2;
            jx += incx; jy += incy;
        }
    }
}

#undef A_
