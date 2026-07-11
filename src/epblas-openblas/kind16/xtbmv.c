/*
 * xtbmv — kind16 port of OpenBLAS ztbmv (interface/tbmv.c +
 * driver/level2/tbmv_thread.c, modes N/T/C).  Complex banded triangular MV.
 *
 *   x := op(A) * x   (op = A, A^T, or A^H).  Band storage matches qtbmv.
 *
 * Threading mirrors tbmv_thread: per-thread buffer + sqrt/even partition
 * + full-length AXPY-reduce.  'C' mode applies conj on every A read.
 *
 * Fortran ABI:  subroutine xtbmv(uplo, trans, diag, n, k, a, lda, x, incx)
 */

#include <stddef.h>
#include <quadmath.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __complex128 C;

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

static void tbmv_kernel(int upper, int trans, int conj, int nounit,
                        ptrdiff_t n, ptrdiff_t k,
                        ptrdiff_t n_from, ptrdiff_t n_to,
                        const C *a, ptrdiff_t lda, const C *x, C *y)
{
    for (ptrdiff_t i = n_from; i < n_to; ++i) {
        ptrdiff_t length;
        if (upper) length = (i < k) ? i : k;
        else       length = (n - i - 1 < k) ? n - i - 1 : k;
        const C *col = &A_(0, i);
        if (upper) {
            if (length > 0) {
                if (!trans) {
                    C xi = x[i];
                    const C *coff = col + (k - length);
                    for (ptrdiff_t j = 0; j < length; ++j)
                        y[i - length + j] += xi * coff[j];
                } else {
                    C s = 0.0Q;
                    const C *coff = col + (k - length);
                    const C *xoff = &x[i - length];
                    if (!conj) for (ptrdiff_t j = 0; j < length; ++j) s += coff[j] * xoff[j];
                    else       for (ptrdiff_t j = 0; j < length; ++j) s += conjq(coff[j]) * xoff[j];
                    y[i] += s;
                }
            }
            if (nounit) y[i] += (conj && trans) ? conjq(col[k]) * x[i] : col[k] * x[i];
            else        y[i] += x[i];
        } else {
            if (nounit) y[i] += (conj && trans) ? conjq(col[0]) * x[i] : col[0] * x[i];
            else        y[i] += x[i];
            if (length > 0) {
                if (!trans) {
                    C xi = x[i];
                    for (ptrdiff_t j = 0; j < length; ++j)
                        y[i + 1 + j] += xi * col[1 + j];
                } else {
                    C s = 0.0Q;
                    const C *xoff = &x[i + 1];
                    if (!conj) for (ptrdiff_t j = 0; j < length; ++j) s += col[1 + j] * xoff[j];
                    else       for (ptrdiff_t j = 0; j < length; ++j) s += conjq(col[1 + j]) * xoff[j];
                    y[i] += s;
                }
            }
        }
    }
}

void xtbmv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const int *K, const C *a, const int *LDA,
            C *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t k    = (ptrdiff_t)(*K);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);

    if (n == 0) return;

    int upper  = (toupper((unsigned char)*UPLO)  == 'U');
    char trc   = (char)toupper((unsigned char)*TRANS);
    int trans  = (trc == 'T' || trc == 'C') ? 1 : 0;
    int noconj = (trc == 'T') ? 1 : 0;
    int nounit = (toupper((unsigned char)*DIAG) == 'N');

    if (incx < 0) x -= (n - 1) * incx;

#ifdef _OPENMP
    int nthreads = (n >= 200) ? omp_get_max_threads() : 1;
    if (nthreads > 1) {
        C *buf_all = (C *)calloc((size_t)nthreads * (size_t)n, sizeof(C));
        ptrdiff_t *range = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
        C *xbuf = NULL;
        const C *xptr = x;
        if (incx != 1) {
            xbuf = (C *)malloc((size_t)n * sizeof(C));
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
                C *y = &buf_all[(size_t)tid * (size_t)n];
                ptrdiff_t n_from, n_to;
                if (n < 2 * k && upper) {
                    n_from = range[nthreads - tid - 1];
                    n_to   = range[nthreads - tid];
                } else {
                    n_from = range[tid];
                    n_to   = range[tid + 1];
                }
                if (n_from < n_to)
                    tbmv_kernel(upper, trans, !noconj, nounit, n, k, n_from, n_to, a, lda, xptr, y);
            }

            for (int t = 1; t < nthreads; ++t) {
                C *slot = &buf_all[(size_t)t * (size_t)n];
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

#define CONJIF(z) (noconj ? (z) : conjq(z))

    if (!trans) {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0Q) {
                        C temp = x[j];
                        ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        const C *col = &A_(0, j);
                        ptrdiff_t off = k - j;
                        for (ptrdiff_t i = i_lo; i < j; ++i) x[i] += temp * col[off + i];
                        if (nounit) x[j] *= col[k];
                    }
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != 0.0Q) {
                        C temp = x[jx];
                        ptrdiff_t i_lo = (j > k) ? j - k : 0;
                        ptrdiff_t ix = kx;
                        const C *col = &A_(0, j);
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
                    if (x[j] != 0.0Q) {
                        C temp = x[j];
                        ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        const C *col = &A_(0, j);
                        ptrdiff_t off = -j;
                        for (ptrdiff_t i = i_hi; i > j; --i) x[i] += temp * col[off + i];
                        if (nounit) x[j] *= col[0];
                    }
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != 0.0Q) {
                        C temp = x[jx];
                        ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                        ptrdiff_t ix = kx;
                        const C *col = &A_(0, j);
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
                    C temp = x[j];
                    const C *col = &A_(0, j);
                    ptrdiff_t off = k - j;
                    if (nounit) temp *= CONJIF(col[k]);
                    ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (ptrdiff_t i = j - 1; i >= i_lo; --i) temp += CONJIF(col[off + i]) * x[i];
                    x[j] = temp;
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[jx];
                    kx -= incx;
                    ptrdiff_t ix = kx;
                    const C *col = &A_(0, j);
                    ptrdiff_t off = k - j;
                    if (nounit) temp *= CONJIF(col[k]);
                    ptrdiff_t i_lo = (j > k) ? j - k : 0;
                    for (ptrdiff_t i = j - 1; i >= i_lo; --i) { temp += CONJIF(col[off + i]) * x[ix]; ix -= incx; }
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = x[j];
                    const C *col = &A_(0, j);
                    ptrdiff_t off = -j;
                    if (nounit) temp *= CONJIF(col[0]);
                    ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (ptrdiff_t i = j + 1; i <= i_hi; ++i) temp += CONJIF(col[off + i]) * x[i];
                    x[j] = temp;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = x[jx];
                    kx += incx;
                    ptrdiff_t ix = kx;
                    const C *col = &A_(0, j);
                    ptrdiff_t off = -j;
                    if (nounit) temp *= CONJIF(col[0]);
                    ptrdiff_t i_hi = (j + k < n - 1) ? j + k : n - 1;
                    for (ptrdiff_t i = j + 1; i <= i_hi; ++i) { temp += CONJIF(col[off + i]) * x[ix]; ix += incx; }
                    x[jx] = temp;
                    jx += incx;
                }
            }
        }
    }
#undef CONJIF
}

#undef A_
