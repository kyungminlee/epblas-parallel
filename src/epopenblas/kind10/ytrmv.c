/*
 * ytrmv — kind10 port of OpenBLAS ztrmv (interface/trmv.c +
 * driver/level2/trmv_thread.c, modes N/T/C).  Complex triangular MV.
 *
 *   x := op(A) * x   where op(A) = A ('N'), A^T ('T'), or A^H ('C')
 *
 * Threading mirrors trmv_thread.c: out-of-place per-thread buffer, sqrt
 * partition, DTB_ENTRIES-tile kernel.  'C' adds conjugation on every
 * read of A (mirrors the TRANSA=4 / MYDOT=DOTC variant).
 *
 * Fortran ABI:  subroutine ytrmv(uplo, trans, diag, n, a, lda, x, incx)
 */

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>
#include <math.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef _Complex long double C;

#define DTB_ENTRIES_K10  32

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

static void trmv_partition(int upper, ptrdiff_t n, int nthreads, ptrdiff_t *range)
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

static void trmv_kernel_N(int upper, int nounit, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const C *a, ptrdiff_t lda, const C *x, C *y)
{
    const ptrdiff_t TB = DTB_ENTRIES_K10;
    for (ptrdiff_t is = m_from; is < m_to; is += TB) {
        ptrdiff_t min_i = (m_to - is < TB) ? m_to - is : TB;
        if (upper && is > 0) {
            for (ptrdiff_t j = is; j < is + min_i; ++j) {
                C xj = x[j];
                const C *col = &A_(0, j);
                for (ptrdiff_t i = 0; i < is; ++i) y[i] += col[i] * xj;
            }
        }
        for (ptrdiff_t i = is; i < is + min_i; ++i) {
            if (upper && i > is) {
                C xi = x[i];
                const C *col = &A_(0, i);
                for (ptrdiff_t k = is; k < i; ++k) y[k] += col[k] * xi;
            }
            if (nounit) y[i] += A_(i, i) * x[i];
            else        y[i] += x[i];
            if (!upper && i + 1 < is + min_i) {
                C xi = x[i];
                const C *col = &A_(0, i);
                for (ptrdiff_t k = i + 1; k < is + min_i; ++k) y[k] += col[k] * xi;
            }
        }
        if (!upper && is + min_i < n) {
            for (ptrdiff_t j = is; j < is + min_i; ++j) {
                C xj = x[j];
                const C *col = &A_(0, j);
                for (ptrdiff_t i = is + min_i; i < n; ++i) y[i] += col[i] * xj;
            }
        }
    }
}

static void trmv_kernel_T(int upper, int nounit, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const C *a, ptrdiff_t lda, const C *x, C *y)
{
    const ptrdiff_t TB = DTB_ENTRIES_K10;
    for (ptrdiff_t is = m_from; is < m_to; is += TB) {
        ptrdiff_t min_i = (m_to - is < TB) ? m_to - is : TB;
        if (upper && is > 0) {
            for (ptrdiff_t j = is; j < is + min_i; ++j) {
                C s = 0.0L;
                const C *col = &A_(0, j);
                for (ptrdiff_t i = 0; i < is; ++i) s += col[i] * x[i];
                y[j] += s;
            }
        }
        for (ptrdiff_t i = is; i < is + min_i; ++i) {
            if (upper && i > is) {
                C s = 0.0L;
                const C *col = &A_(0, i);
                for (ptrdiff_t k = is; k < i; ++k) s += col[k] * x[k];
                y[i] += s;
            }
            if (nounit) y[i] += A_(i, i) * x[i];
            else        y[i] += x[i];
            if (!upper && i + 1 < is + min_i) {
                C s = 0.0L;
                const C *col = &A_(0, i);
                for (ptrdiff_t k = i + 1; k < is + min_i; ++k) s += col[k] * x[k];
                y[i] += s;
            }
        }
        if (!upper && is + min_i < n) {
            for (ptrdiff_t j = is; j < is + min_i; ++j) {
                C s = 0.0L;
                const C *col = &A_(0, j);
                for (ptrdiff_t i = is + min_i; i < n; ++i) s += col[i] * x[i];
                y[j] += s;
            }
        }
    }
}

static void trmv_kernel_C(int upper, int nounit, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const C *a, ptrdiff_t lda, const C *x, C *y)
{
    const ptrdiff_t TB = DTB_ENTRIES_K10;
    for (ptrdiff_t is = m_from; is < m_to; is += TB) {
        ptrdiff_t min_i = (m_to - is < TB) ? m_to - is : TB;
        if (upper && is > 0) {
            for (ptrdiff_t j = is; j < is + min_i; ++j) {
                C s = 0.0L;
                const C *col = &A_(0, j);
                for (ptrdiff_t i = 0; i < is; ++i) s += conjl(col[i]) * x[i];
                y[j] += s;
            }
        }
        for (ptrdiff_t i = is; i < is + min_i; ++i) {
            if (upper && i > is) {
                C s = 0.0L;
                const C *col = &A_(0, i);
                for (ptrdiff_t k = is; k < i; ++k) s += conjl(col[k]) * x[k];
                y[i] += s;
            }
            if (nounit) y[i] += conjl(A_(i, i)) * x[i];
            else        y[i] += x[i];
            if (!upper && i + 1 < is + min_i) {
                C s = 0.0L;
                const C *col = &A_(0, i);
                for (ptrdiff_t k = i + 1; k < is + min_i; ++k) s += conjl(col[k]) * x[k];
                y[i] += s;
            }
        }
        if (!upper && is + min_i < n) {
            for (ptrdiff_t j = is; j < is + min_i; ++j) {
                C s = 0.0L;
                const C *col = &A_(0, j);
                for (ptrdiff_t i = is + min_i; i < n; ++i) s += conjl(col[i]) * x[i];
                y[j] += s;
            }
        }
    }
}

void ytrmv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const C *a, const int *LDA,
            C *x, const int *INCX,
            size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    ptrdiff_t n    = (ptrdiff_t)(*N);
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
    int nthreads = 1;
    if (n >= 50) {
        nthreads = omp_get_max_threads();
        if (n < 500 && nthreads > 2) nthreads = 2;
    }
    if (nthreads > 1) {
        C *buf_all = (C *)calloc((size_t)nthreads * (size_t)n, sizeof(C));
        ptrdiff_t *range_m = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
        C *xbuf = NULL;
        const C *xptr = x;
        if (incx != 1) {
            xbuf = (C *)malloc((size_t)n * sizeof(C));
            if (xbuf) {
                for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
                xptr = xbuf;
            }
        }
        if (buf_all && range_m && (incx == 1 || xbuf)) {
            trmv_partition(upper, n, nthreads, range_m);

            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                C *y = trans ? buf_all : &buf_all[(size_t)tid * (size_t)n];
                ptrdiff_t m_from, m_to;
                if (upper) {
                    m_from = range_m[nthreads - tid - 1];
                    m_to   = range_m[nthreads - tid];
                } else {
                    m_from = range_m[tid];
                    m_to   = range_m[tid + 1];
                }
                if (m_from < m_to) {
                    if (!trans)        trmv_kernel_N(upper, nounit, n, m_from, m_to, a, lda, xptr, y);
                    else if (noconj)   trmv_kernel_T(upper, nounit, n, m_from, m_to, a, lda, xptr, y);
                    else               trmv_kernel_C(upper, nounit, n, m_from, m_to, a, lda, xptr, y);
                }
            }

            if (!trans) {
                if (upper) {
                    for (int t = 1; t < nthreads; ++t) {
                        ptrdiff_t m_to_t = range_m[nthreads - t];
                        C *slot = &buf_all[(size_t)t * (size_t)n];
                        for (ptrdiff_t i = 0; i < m_to_t; ++i) buf_all[i] += slot[i];
                    }
                } else {
                    for (int t = 1; t < nthreads; ++t) {
                        ptrdiff_t m_from_t = range_m[t];
                        C *slot = &buf_all[(size_t)t * (size_t)n];
                        for (ptrdiff_t i = m_from_t; i < n; ++i) buf_all[i] += slot[i];
                    }
                }
            }

            if (incx == 1) {
                for (ptrdiff_t i = 0; i < n; ++i) x[i] = buf_all[i];
            } else {
                for (ptrdiff_t i = 0; i < n; ++i) x[i * incx] = buf_all[i];
            }

            free(buf_all); free(range_m);
            if (xbuf) free(xbuf);
            return;
        }
        free(buf_all); free(range_m);
        if (xbuf) free(xbuf);
    }
#endif

    /* Serial Fortran-reference path. */
    ptrdiff_t kx = 0;

    if (!trans) {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0L) {
                        C temp = x[j];
                        const C *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) x[i] += temp * aj[i];
                        if (nounit) x[j] *= aj[j];
                    }
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != 0.0L) {
                        C temp = x[jx];
                        ptrdiff_t ix = kx;
                        const C *aj = &A_(0, j);
                        for (ptrdiff_t i = 0; i < j; ++i) { x[ix] += temp * aj[i]; ix += incx; }
                        if (nounit) x[jx] *= aj[j];
                    }
                    jx += incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0L) {
                        C temp = x[j];
                        const C *aj = &A_(0, j);
                        for (ptrdiff_t i = n - 1; i > j; --i) x[i] += temp * aj[i];
                        if (nounit) x[j] *= aj[j];
                    }
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != 0.0L) {
                        C temp = x[jx];
                        ptrdiff_t ix = kx;
                        const C *aj = &A_(0, j);
                        for (ptrdiff_t i = n - 1; i > j; --i) { x[ix] += temp * aj[i]; ix -= incx; }
                        if (nounit) x[jx] *= aj[j];
                    }
                    jx -= incx;
                }
            }
        }
    } else {
        if (upper) {
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[j];
                    const C *aj = &A_(0, j);
                    if (noconj) {
                        if (nounit) temp *= aj[j];
                        for (ptrdiff_t i = j - 1; i >= 0; --i) temp += aj[i] * x[i];
                    } else {
                        if (nounit) temp *= conjl(aj[j]);
                        for (ptrdiff_t i = j - 1; i >= 0; --i) temp += conjl(aj[i]) * x[i];
                    }
                    x[j] = temp;
                }
            } else {
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[jx];
                    ptrdiff_t ix = jx;
                    const C *aj = &A_(0, j);
                    if (noconj) {
                        if (nounit) temp *= aj[j];
                        for (ptrdiff_t i = j - 1; i >= 0; --i) { ix -= incx; temp += aj[i] * x[ix]; }
                    } else {
                        if (nounit) temp *= conjl(aj[j]);
                        for (ptrdiff_t i = j - 1; i >= 0; --i) { ix -= incx; temp += conjl(aj[i]) * x[ix]; }
                    }
                    x[jx] = temp;
                    jx -= incx;
                }
            }
        } else {
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = x[j];
                    const C *aj = &A_(0, j);
                    if (noconj) {
                        if (nounit) temp *= aj[j];
                        for (ptrdiff_t i = j + 1; i < n; ++i) temp += aj[i] * x[i];
                    } else {
                        if (nounit) temp *= conjl(aj[j]);
                        for (ptrdiff_t i = j + 1; i < n; ++i) temp += conjl(aj[i]) * x[i];
                    }
                    x[j] = temp;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = x[jx];
                    ptrdiff_t ix = jx;
                    const C *aj = &A_(0, j);
                    if (noconj) {
                        if (nounit) temp *= aj[j];
                        for (ptrdiff_t i = j + 1; i < n; ++i) { ix += incx; temp += aj[i] * x[ix]; }
                    } else {
                        if (nounit) temp *= conjl(aj[j]);
                        for (ptrdiff_t i = j + 1; i < n; ++i) { ix += incx; temp += conjl(aj[i]) * x[ix]; }
                    }
                    x[jx] = temp;
                    jx += incx;
                }
            }
        }
    }
}

#undef A_
