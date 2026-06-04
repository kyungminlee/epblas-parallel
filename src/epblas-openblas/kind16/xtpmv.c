/*
 * xtpmv — kind16 port of OpenBLAS ztpmv (interface/tpmv.c +
 * driver/level2/tpmv_thread.c).  Complex packed triangular MV.
 *
 *   x := op(A) * x  (op = A, A^T, A^H).  Packed storage matches qtpmv.
 *
 * Parallel path mirrors tpmv_thread.c (TRANSA=1/2/3/4):
 *   - sqrt-partition (mask=7, min-width 16) with UPPER reverse mapping.
 *   - TRANS path: disjoint y[m_from..m_to) writes into slot[0]; no reduce.
 *   - NoTrans path: per-thread slot of length n + controller AXPY-reduce.
 *   - 'C' (conj-trans) wraps ap reads with conjq().
 *
 * Fortran ABI:  subroutine xtpmv(uplo, trans, diag, n, ap, x, incx)
 */

#include <stddef.h>
#include <quadmath.h>
#include <stdlib.h>
#include <math.h>
#include <complex.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __complex128 C;

static inline size_t col_start_U(ptrdiff_t j) { return (size_t)j * (size_t)(j + 1) / 2; }
static inline size_t col_start_L(ptrdiff_t j, ptrdiff_t n) {
    return (size_t)j * (size_t)(2 * n - j + 1) / 2;
}

static void tpmv_partition(int upper, ptrdiff_t n, int nthreads, ptrdiff_t *range)
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

static void tpmv_kernel_N(int upper, int nounit, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const C *ap, const C *x, C *y)
{
    if (upper) {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_U(j);
            C xj = x[j];
            for (ptrdiff_t i = 0; i < j; ++i) y[i] += ap[cs + (size_t)i] * xj;
            if (nounit) y[j] += ap[cs + (size_t)j] * xj;
            else        y[j] += xj;
        }
    } else {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_L(j, n);
            C xj = x[j];
            if (nounit) y[j] += ap[cs] * xj;
            else        y[j] += xj;
            for (ptrdiff_t i = j + 1; i < n; ++i) y[i] += ap[cs + (size_t)(i - j)] * xj;
        }
    }
}

static void tpmv_kernel_T(int upper, int nounit, int conj, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const C *ap, const C *x, C *y)
{
#define APV(k) (conj ? conjq(ap[k]) : ap[k])
    if (upper) {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_U(j);
            C s = 0.0Q;
            for (ptrdiff_t i = 0; i < j; ++i) s += APV(cs + (size_t)i) * x[i];
            if (nounit) s += APV(cs + (size_t)j) * x[j];
            else        s += x[j];
            y[j] += s;
        }
    } else {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_L(j, n);
            C s = 0.0Q;
            if (nounit) s += APV(cs) * x[j];
            else        s += x[j];
            for (ptrdiff_t i = j + 1; i < n; ++i) s += APV(cs + (size_t)(i - j)) * x[i];
            y[j] += s;
        }
    }
#undef APV
}

void xtpmv_(const char *UPLO, const char *TRANS, const char *DIAG,
            const int *N, const C *ap,
            C *x, const int *INCX,
            size_t uplo_len, size_t trans_len, size_t diag_len)
{
    (void)uplo_len; (void)trans_len; (void)diag_len;
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);

    if (n == 0) return;

    int upper  = (toupper((unsigned char)*UPLO)  == 'U');
    char trc   = (char)toupper((unsigned char)*TRANS);
    int trans  = (trc == 'T' || trc == 'C') ? 1 : 0;
    int noconj = (trc == 'T') ? 1 : 0;
    int conj   = (trc == 'C') ? 1 : 0;
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
            tpmv_partition(upper, n, nthreads, range_m);

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
                    if (!trans) tpmv_kernel_N(upper, nounit, n, m_from, m_to, ap, xptr, y);
                    else        tpmv_kernel_T(upper, nounit, conj, n, m_from, m_to, ap, xptr, y);
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

    /* Serial Fortran-reference path — in-place, no scratch.
     * x has been normalized so x[i*incx] = logical[i] for i in [0,n). */
    ptrdiff_t kx = 0;

#define CONJIF(z) (noconj ? (z) : conjq(z))

    if (!trans) {
        if (upper) {
            ptrdiff_t kk = 0;
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != 0.0Q) {
                        C temp = x[j];
                        ptrdiff_t k = kk;
                        for (ptrdiff_t i = 0; i < j; ++i) { x[i] += temp * ap[k]; ++k; }
                        if (nounit) x[j] *= ap[kk + j];
                    }
                    kk += j + 1;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != 0.0Q) {
                        C temp = x[jx];
                        ptrdiff_t ix = kx;
                        for (ptrdiff_t k = kk; k < kk + j; ++k) { x[ix] += temp * ap[k]; ix += incx; }
                        if (nounit) x[jx] *= ap[kk + j];
                    }
                    jx += incx;
                    kk += j + 1;
                }
            }
        } else {
            ptrdiff_t kk = n * (n + 1) / 2 - 1;
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != 0.0Q) {
                        C temp = x[j];
                        ptrdiff_t k = kk;
                        for (ptrdiff_t i = n - 1; i > j; --i) { x[i] += temp * ap[k]; --k; }
                        if (nounit) x[j] *= ap[kk - (n - 1 - j)];
                    }
                    kk -= (n - j);
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != 0.0Q) {
                        C temp = x[jx];
                        ptrdiff_t ix = kx;
                        ptrdiff_t k = kk;
                        for (ptrdiff_t i = n - 1; i > j; --i) { x[ix] += temp * ap[k]; ix -= incx; --k; }
                        if (nounit) x[jx] *= ap[kk - (n - 1 - j)];
                    }
                    jx -= incx;
                    kk -= (n - j);
                }
            }
        }
    } else {
        if (upper) {
            ptrdiff_t kk = n * (n + 1) / 2 - 1;
            if (incx == 1) {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[j];
                    if (nounit) temp *= CONJIF(ap[kk]);
                    ptrdiff_t k = kk - 1;
                    for (ptrdiff_t i = j - 1; i >= 0; --i) { temp += CONJIF(ap[k]) * x[i]; --k; }
                    x[j] = temp;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    C temp = x[jx];
                    ptrdiff_t ix = jx;
                    if (nounit) temp *= CONJIF(ap[kk]);
                    ptrdiff_t k = kk - 1;
                    for (ptrdiff_t i = j - 1; i >= 0; --i) { ix -= incx; temp += CONJIF(ap[k]) * x[ix]; --k; }
                    x[jx] = temp;
                    jx -= incx;
                    kk -= j + 1;
                }
            }
        } else {
            ptrdiff_t kk = 0;
            if (incx == 1) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = x[j];
                    if (nounit) temp *= CONJIF(ap[kk]);
                    ptrdiff_t k = kk + 1;
                    for (ptrdiff_t i = j + 1; i < n; ++i) { temp += CONJIF(ap[k]) * x[i]; ++k; }
                    x[j] = temp;
                    kk += (n - j);
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = x[jx];
                    ptrdiff_t ix = jx;
                    if (nounit) temp *= CONJIF(ap[kk]);
                    ptrdiff_t k = kk + 1;
                    for (ptrdiff_t i = j + 1; i < n; ++i) { ix += incx; temp += CONJIF(ap[k]) * x[ix]; ++k; }
                    x[jx] = temp;
                    jx += incx;
                    kk += (n - j);
                }
            }
        }
    }
#undef CONJIF
}
