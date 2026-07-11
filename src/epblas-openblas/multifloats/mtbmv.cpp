/*
 * mtbmv — kind10 port of OpenBLAS dtbmv (interface/tbmv.c +
 * driver/level2/tbmv_thread.c).  Triangular banded matrix-vector.
 *
 *   x := op(A) * x   where A is N x N triangular with K extra diagonals
 *
 * Band storage (lda >= K+1):
 *   UPLO='U': ab[(k - j + i) + j*lda] = A(i,j) for max(0, j-k) <= i <= j
 *   UPLO='L': ab[(i - j) + j*lda]     = A(i,j) for j <= i <= min(n-1, j+k)
 *
 * Threading mirrors tbmv_thread:
 *   - Sqrt-partition if n < 2k (work per col ~ min(k, j) or min(k, n-j-1));
 *     even split if n >= 2k (each col costs k).
 *   - Per-thread accumulator slot of length n (calloc'd zero).  All
 *     four (UPLO × TRANS) cases use per-thread slots — TRANS does NOT
 *     short-cut to a shared slot here (unlike trmv_thread).
 *   - Final controller AXPY-reduces all slots into slot[0] (full length).
 *   - Copy slot[0][0..n) back to x with stride.
 *
 * Fortran ABI:  subroutine mtbmv(uplo, trans, diag, n, k, a, lda, x, incx)
 */

#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <stdlib.h>
#include <math.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

static void tbmv_partition_diag(int upper, ptrdiff_t n, int nthreads, ptrdiff_t *range)
{
    const int mask = 7;
    double dnum = (double)n * (double)n / (double)nthreads;
    if (!upper) {
        range[0] = 0;
        ptrdiff_t i = 0;
        int num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                if (di * di - dnum > 0.0)
                    width = ((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask;
                else
                    width = n - i;
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else {
                width = n - i;
            }
            range[num_cpu + 1] = range[num_cpu] + width;
            num_cpu++;
            i += width;
        }
        for (int t = num_cpu + 1; t <= nthreads; ++t) range[t] = range[num_cpu];
    } else {
        range[nthreads] = n;
        ptrdiff_t i = 0;
        int num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                if (di * di - dnum > 0.0)
                    width = ((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask;
                else
                    width = n - i;
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else {
                width = n - i;
            }
            range[nthreads - num_cpu - 1] = range[nthreads - num_cpu] - width;
            num_cpu++;
            i += width;
        }
        for (int t = 0; t < nthreads - num_cpu; ++t) range[t] = range[nthreads - num_cpu];
    }
}

static void tbmv_partition_even(ptrdiff_t n, int nthreads, ptrdiff_t *range)
{
    range[0] = 0;
    ptrdiff_t i = n;
    int num_cpu = 0;
    while (i > 0 && num_cpu < nthreads) {
        ptrdiff_t width = (i + nthreads - num_cpu - 1) / (nthreads - num_cpu);
        if (width < 4) width = 4;
        if (i < width) width = i;
        range[num_cpu + 1] = range[num_cpu] + width;
        num_cpu++;
        i -= width;
    }
    for (int t = num_cpu + 1; t <= nthreads; ++t) range[t] = range[num_cpu];
}

static void tbmv_kernel(int upper, int trans, int nounit,
                        ptrdiff_t n, ptrdiff_t k,
                        ptrdiff_t n_from, ptrdiff_t n_to,
                        const T *a, ptrdiff_t lda, const T *x, T *y)
{
    for (ptrdiff_t i = n_from; i < n_to; ++i) {
        ptrdiff_t length;
        if (upper) length = (i < k) ? i : k;
        else       length = (n - i - 1 < k) ? n - i - 1 : k;
        const T *col = &A_(0, i);
        if (upper) {
            if (length > 0) {
                if (!trans) {
                    /* y[(i-length)..i) += x[i] * a[(k-length)..k, i]. */
                    T xi = x[i];
                    const T *coff = col + (k - length);
                    for (ptrdiff_t j = 0; j < length; ++j)
                        y[i - length + j] += xi * coff[j];
                } else {
                    /* y[i] += dot(a[(k-length)..k, i], x[(i-length)..i)). */
                    T s = 0.0;
                    const T *coff = col + (k - length);
                    const T *xoff = &x[i - length];
                    for (ptrdiff_t j = 0; j < length; ++j) s += coff[j] * xoff[j];
                    y[i] += s;
                }
            }
            /* Diagonal. */
            if (nounit) y[i] += col[k] * x[i];
            else        y[i] += x[i];
        } else {
            /* Diagonal first for LOWER. */
            if (nounit) y[i] += col[0] * x[i];
            else        y[i] += x[i];
            if (length > 0) {
                if (!trans) {
                    /* y[(i+1)..(i+length+1)) += x[i] * a[1..length+1, i]. */
                    T xi = x[i];
                    for (ptrdiff_t j = 0; j < length; ++j)
                        y[i + 1 + j] += xi * col[1 + j];
                } else {
                    /* y[i] += dot(a[1..length+1, i], x[(i+1)..(i+length+1))). */
                    T s = 0.0;
                    const T *xoff = &x[i + 1];
                    for (ptrdiff_t j = 0; j < length; ++j) s += col[1 + j] * xoff[j];
                    y[i] += s;
                }
            }
        }
    }
}

extern "C" void mtbmv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const int *K, const T *a, const int *LDA,
            T *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t k    = (ptrdiff_t)(*K);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);

    if (n == 0) return;

    int upper  = (toupper((unsigned char)*UPLO)  == 'U');
    char trc   = (char)toupper((unsigned char)*TRANS);
    int trans  = (trc == 'T' || trc == 'C') ? 1 : 0;
    int nounit = (toupper((unsigned char)*DIAG) == 'N');

    if (incx < 0) x -= (n - 1) * incx;

#ifdef _OPENMP
    int nthreads = (n >= 200) ? omp_get_max_threads() : 1;
    if (nthreads > 1) {
        T *buf_all = (T *)calloc((size_t)nthreads * (size_t)n, sizeof(T));
        ptrdiff_t *range = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
        T *xbuf = NULL;
        const T *xptr = x;
        if (incx != 1) {
            xbuf = (T *)malloc((size_t)n * sizeof(T));
            if (xbuf) {
                for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
                xptr = xbuf;
            }
        }
        if (buf_all && range && (incx == 1 || xbuf)) {
            if (n < 2 * k) tbmv_partition_diag(upper, n, nthreads, range);
            else           tbmv_partition_even(n, nthreads, range);

            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                T *y = &buf_all[(size_t)tid * (size_t)n];
                ptrdiff_t n_from, n_to;
                if (n < 2 * k && upper) {
                    n_from = range[nthreads - tid - 1];
                    n_to   = range[nthreads - tid];
                } else {
                    n_from = range[tid];
                    n_to   = range[tid + 1];
                }
                if (n_from < n_to)
                    tbmv_kernel(upper, trans, nounit, n, k, n_from, n_to, a, lda, xptr, y);
            }

            /* Full-length AXPY-reduce all slots into slot[0]. */
            for (int t = 1; t < nthreads; ++t) {
                T *slot = &buf_all[(size_t)t * (size_t)n];
                for (ptrdiff_t i = 0; i < n; ++i) buf_all[i] += slot[i];
            }

            if (incx == 1) {
                for (ptrdiff_t i = 0; i < n; ++i) x[i] = buf_all[i];
            } else {
                for (ptrdiff_t i = 0; i < n; ++i) x[i * incx] = buf_all[i];
            }

            free(buf_all); free(range);
            if (xbuf) free(xbuf);
            return;
        }
        free(buf_all); free(range);
        if (xbuf) free(xbuf);
    }
#endif

    /* Serial Fortran-reference path. */
    ptrdiff_t kx = 0;

    if (!trans) {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0) {
                        T temp = x[j];
                        ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        const T *col = &A_(0, j);
                        ptrdiff_t off = k - j;
                        for (ptrdiff_t i = i_lo; i < j; ++i) x[i] += temp * col[off + i];
                        if (nounit) x[j] *= col[k];
                    }
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != 0.0) {
                        T temp = x[jx];
                        ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        ptrdiff_t ix = kx;
                        const T *col = &A_(0, j);
                        ptrdiff_t off = k - j;
                        for (ptrdiff_t i = i_lo; i < j; ++i) { x[ix] += temp * col[off + i]; ix += incx; }
                        if (nounit) x[jx] *= col[k];
                    }
                    jx += incx;
                    if (j >= k) kx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0) {
                        T temp = x[j];
                        ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        const T *col = &A_(0, j);
                        ptrdiff_t off = -j;
                        for (ptrdiff_t i = i_hi; i > j; --i) x[i] += temp * col[off + i];
                        if (nounit) x[j] *= col[0];
                    }
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != 0.0) {
                        T temp = x[jx];
                        ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        ptrdiff_t ix = kx;
                        const T *col = &A_(0, j);
                        ptrdiff_t off = -j;
                        for (ptrdiff_t i = i_hi; i > j; --i) { x[ix] += temp * col[off + i]; ix -= incx; }
                        if (nounit) x[jx] *= col[0];
                    }
                    jx -= incx;
                    if (n - 1 - j >= k) kx -= incx;
                }
            }
        }
    } else {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[j];
                    const T *col = &A_(0, j);
                    ptrdiff_t off = k - j;
                    if (nounit) temp *= col[k];
                    ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (ptrdiff_t i = j - 1; i >= i_lo; --i) temp += col[off + i] * x[i];
                    x[j] = temp;
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T temp = x[jx];
                    kx -= incx;
                    ptrdiff_t ix = kx;
                    const T *col = &A_(0, j);
                    ptrdiff_t off = k - j;
                    if (nounit) temp *= col[k];
                    ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (ptrdiff_t i = j - 1; i >= i_lo; --i) { temp += col[off + i] * x[ix]; ix -= incx; }
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[j];
                    const T *col = &A_(0, j);
                    ptrdiff_t off = -j;
                    if (nounit) temp *= col[0];
                    ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (ptrdiff_t i = j + 1; i <= i_hi; ++i) temp += col[off + i] * x[i];
                    x[j] = temp;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T temp = x[jx];
                    kx += incx;
                    ptrdiff_t ix = kx;
                    const T *col = &A_(0, j);
                    ptrdiff_t off = -j;
                    if (nounit) temp *= col[0];
                    ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (ptrdiff_t i = j + 1; i <= i_hi; ++i) { temp += col[off + i] * x[ix]; ix += incx; }
                    x[jx] = temp;
                    jx += incx;
                }
            }
        }
    }
}

#undef A_
