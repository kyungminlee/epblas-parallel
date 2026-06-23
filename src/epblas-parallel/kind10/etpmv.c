/*
 * etpmv — kind10 (long double) triangular packed matrix-vector.
 *   x := A*x or A^T*x
 *
 * The serial reference is the in-place column sweep (each column updates a run
 * of x in dependency order), which is inherently sequential. But the operation
 * is a pure multiply — every *output* element is an independent dot/axpy — so
 * the threaded path (large N) reformulates it out-of-place over a private
 * buffer and partitions columns across threads, mirroring OpenBLAS tpmv_thread:
 *   - NoTrans: column j writes a run of y (cross-column) -> per-thread private
 *     slot, controller AXPY-reduces the touched range into slot 0.
 *   - Trans:   y[j] is a dot of column j with x (disjoint per thread) -> all
 *     threads write disjoint y[m_from..m_to) into the shared slot 0, no reduce.
 * Final copy writes the buffer back to x. The column kernels index a
 * loop-invariant per-column base (ap[cs + i]) to keep the fp80 loop body off
 * a live running packed index.
 */

#include <stddef.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef long double TR;


#ifdef _OPENMP
static inline size_t col_start_U(ptrdiff_t j) { return (size_t)j * (size_t)(j + 1) / 2; }
static inline size_t col_start_L(ptrdiff_t j, ptrdiff_t n) {
    return (size_t)j * (size_t)(2 * n - j + 1) / 2;
}

/* Sqrt-balanced contiguous column partition (OpenBLAS tpmv_partition, mask=7,
 * min-width 16). UPPER reverses the assignment so thread 0 takes the highest
 * (heaviest) columns; LOWER is forward. */
static void tpmv_partition(bool upper, ptrdiff_t n, ptrdiff_t nthreads, ptrdiff_t *range)
{
    const ptrdiff_t mask = 7;
    const double dnum = (double)n * (double)n / (double)nthreads;
    if (!upper) {
        range[0] = 0;
        ptrdiff_t i = 0; ptrdiff_t num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                width = (di * di - dnum > 0.0)
                    ? (((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask)
                    : (n - i);
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else width = n - i;
            range[num_cpu + 1] = range[num_cpu] + width;
            num_cpu++; i += width;
        }
        for (ptrdiff_t t = num_cpu + 1; t <= nthreads; ++t) range[t] = range[num_cpu];
    } else {
        range[nthreads] = n;
        ptrdiff_t i = 0; ptrdiff_t num_cpu = 0;
        while (i < n && num_cpu < nthreads) {
            ptrdiff_t width;
            if (nthreads - num_cpu > 1) {
                double di = (double)(n - i);
                width = (di * di - dnum > 0.0)
                    ? (((ptrdiff_t)(-sqrt(di * di - dnum) + di) + mask) & ~(ptrdiff_t)mask)
                    : (n - i);
                if (width < 16) width = 16;
                if (width > n - i) width = n - i;
            } else width = n - i;
            range[nthreads - num_cpu - 1] = range[nthreads - num_cpu] - width;
            num_cpu++; i += width;
        }
        for (ptrdiff_t t = 0; t < nthreads - num_cpu; ++t) range[t] = range[nthreads - num_cpu];
    }
}

static void tpmv_kernel_N(bool upper, bool nounit, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const TR *ap, const TR *x, TR *y)
{
    if (upper) {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_U(j);
            TR xj = x[j];
            for (ptrdiff_t i = 0; i < j; ++i) y[i] += ap[cs + (size_t)i] * xj;
            y[j] += nounit ? ap[cs + (size_t)j] * xj : xj;
        }
    } else {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_L(j, n);
            TR xj = x[j];
            y[j] += nounit ? ap[cs] * xj : xj;
            for (ptrdiff_t i = j + 1; i < n; ++i) y[i] += ap[cs + (size_t)(i - j)] * xj;
        }
    }
}

static void tpmv_kernel_T(bool upper, bool nounit, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const TR *ap, const TR *x, TR *y)
{
    if (upper) {
        /* Diagonal (ap[cs+j]) sits at the END of the column, so accumulate the
         * cs+0..cs+j-1 run first and fold the diagonal in last — keeping the
         * packed read stream sequential (diag-first jumped back ~5-8%). */
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_U(j);
            TR s = 0.0L;
            for (ptrdiff_t i = 0; i < j; ++i) s += ap[cs + (size_t)i] * x[i];
            s += nounit ? ap[cs + (size_t)j] * x[j] : x[j];
            y[j] += s;
        }
    } else {
        /* Diagonal (ap[cs]) is at the column start, so diag-first stays sequential. */
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_L(j, n);
            TR s = nounit ? ap[cs] * x[j] : x[j];
            for (ptrdiff_t i = j + 1; i < n; ++i) s += ap[cs + (size_t)(i - j)] * x[i];
            y[j] += s;
        }
    }
}

/* Threaded out-of-place path. Returns 1 if it handled the call, 0 to fall back
 * to the serial reference. Kept in its own noinline function so the in-place
 * serial loops below compile in a clean register context (an inline threaded
 * block crowded the x87 allocation and slowed the UPPER NoTrans serial sweep
 * ~14%). */
__attribute__((noinline))
static ptrdiff_t etpmv_omp(bool upper, bool is_t, bool nounit, ptrdiff_t n, ptrdiff_t incx,
                     const TR *restrict ap, TR *restrict x)
{
    ptrdiff_t nthreads = 1;
    if (n >= 50 && !omp_in_parallel()) {
        nthreads = blas_omp_max_threads();
        if (n < 500 && nthreads > 2) nthreads = 2;
    }
    if (nthreads <= 1) return 0;

    const ptrdiff_t kx = (incx < 0) ? -(n - 1) * (ptrdiff_t)incx : 0;
    TR *buf_all = (TR *)calloc((size_t)nthreads * (size_t)n, sizeof(TR));
    ptrdiff_t *range_m = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
    TR *xbuf = NULL;
    const TR *xptr = x;
    if (incx != 1 && buf_all && range_m) {
        xbuf = (TR *)malloc((size_t)n * sizeof(TR));
        if (xbuf) {
            for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[kx + i * incx];
            xptr = xbuf;
        }
    }
    if (!(buf_all && range_m && (incx == 1 || xbuf))) {
        free(buf_all); free(range_m); if (xbuf) free(xbuf);
        return 0;
    }

    tpmv_partition(upper, n, nthreads, range_m);
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        TR *y = is_t ? buf_all : &buf_all[(size_t)tid * (size_t)n];
        ptrdiff_t m_from, m_to;
        if (upper) { m_from = range_m[nthreads - tid - 1]; m_to = range_m[nthreads - tid]; }
        else       { m_from = range_m[tid];               m_to = range_m[tid + 1]; }
        if (m_from < m_to) {
            if (is_t) tpmv_kernel_T(upper, nounit, n, m_from, m_to, ap, xptr, y);
            else      tpmv_kernel_N(upper, nounit, n, m_from, m_to, ap, xptr, y);
        }
    }
    if (!is_t) {  /* reduce private slots into slot 0 over the touched range */
        if (upper) {
            for (ptrdiff_t t = 1; t < nthreads; ++t) {
                ptrdiff_t m_to_t = range_m[nthreads - t];
                const TR *slot = &buf_all[(size_t)t * (size_t)n];
                for (ptrdiff_t i = 0; i < m_to_t; ++i) buf_all[i] += slot[i];
            }
        } else {
            for (ptrdiff_t t = 1; t < nthreads; ++t) {
                ptrdiff_t m_from_t = range_m[t];
                const TR *slot = &buf_all[(size_t)t * (size_t)n];
                for (ptrdiff_t i = m_from_t; i < n; ++i) buf_all[i] += slot[i];
            }
        }
    }
    if (incx == 1) for (ptrdiff_t i = 0; i < n; ++i) x[i] = buf_all[i];
    else           for (ptrdiff_t i = 0; i < n; ++i) x[kx + i * incx] = buf_all[i];
    free(buf_all); free(range_m); if (xbuf) free(xbuf);
    return 1;
}
#endif /* _OPENMP */

static void etpmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n,
    const TR *restrict ap,
    TR *restrict x, ptrdiff_t incx)
{
    const TR zero = 0.0L;
    const char UPLO = blas_up(uplo);
    char TRANS = blas_up(trans);
    if (TRANS == 'C') TRANS = 'T';
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    if (etpmv_omp(UPLO == 'U', TRANS == 'T', nounit, n, incx, ap, x)) return;
#endif

    if (incx == 1) {
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        const TR tmp = x[j];
                        ptrdiff_t k = kk;
                        for (ptrdiff_t i = 0; i < j; ++i) { x[i] += tmp * ap[k]; ++k; }
                        if (nounit) x[j] *= ap[kk + j];
                    }
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const TR tmp = x[j];
                        ptrdiff_t k = kk;
                        for (ptrdiff_t i = n - 1; i > j; --i) { x[i] += tmp * ap[k]; --k; }
                        if (nounit) x[j] *= ap[kk - (n - 1 - j)];
                    }
                    kk -= (n - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR tmp = x[j];
                    if (nounit) tmp *= ap[kk];
                    ptrdiff_t k = kk - 1;
                    for (ptrdiff_t i = j - 1; i >= 0; --i) { tmp += ap[k] * x[i]; --k; }
                    x[j] = tmp;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR tmp = x[j];
                    if (nounit) tmp *= ap[kk];
                    ptrdiff_t k = kk + 1;
                    for (ptrdiff_t i = j + 1; i < n; ++i) { tmp += ap[k] * x[i]; ++k; }
                    x[j] = tmp;
                    kk += n - j;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != zero) {
                        const TR tmp = x[jx];
                        ptrdiff_t ix = kx;
                        for (ptrdiff_t k = kk; k < kk + j; ++k) {
                            x[ix] += tmp * ap[k];
                            ix += incx;
                        }
                        if (nounit) x[jx] *= ap[kk + j];
                    }
                    jx += incx;
                    kk += j + 1;
                }
            } else {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        const TR tmp = x[jx];
                        ptrdiff_t ix = kx;
                        for (ptrdiff_t k = kk; k > kk - (n - 1 - j); --k) {
                            x[ix] += tmp * ap[k];
                            ix -= incx;
                        }
                        if (nounit) x[jx] *= ap[kk - (n - 1 - j)];
                    }
                    jx -= incx;
                    kk -= (n - j);
                }
            }
        } else {
            if (UPLO == 'U') {
                ptrdiff_t kk = (n * (n + 1)) / 2 - 1;
                ptrdiff_t jx = kx + (n - 1) * incx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    TR tmp = x[jx];
                    ptrdiff_t ix = jx;
                    if (nounit) tmp *= ap[kk];
                    for (ptrdiff_t k = kk - 1; k >= kk - j; --k) {
                        ix -= incx;
                        tmp += ap[k] * x[ix];
                    }
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR tmp = x[jx];
                    ptrdiff_t ix = jx;
                    if (nounit) tmp *= ap[kk];
                    for (ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                        ix += incx;
                        tmp += ap[k] * x[ix];
                    }
                    x[jx] = tmp;
                    jx += incx;
                    kk += n - j;
                }
            }
        }
    }
}

EPBLAS_FACADE_TPMV(etpmv, TR)
