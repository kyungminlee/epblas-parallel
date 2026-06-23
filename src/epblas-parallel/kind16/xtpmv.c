/*
 * xtpmv — kind16 complex triangular packed matrix-vector.
 *   x := op(A)*x   (op = none / transpose / conjugate-transpose)
 *
 * Complex twin of qtpmv. The serial reference is the in-place column sweep
 * (sequential); the threaded path (large N) reformulates the multiply out-of-
 * place over a private buffer and partitions columns, mirroring kind10 ytpmv /
 * OpenBLAS tpmv_thread:
 *   - NoTrans: column j writes a run of y (cross-column) -> per-thread private
 *     slot, controller AXPY-reduces the touched range into slot 0.
 *   - Trans/ConjTrans: y[j] = dot(op(column j), x), disjoint per thread -> all
 *     threads write disjoint y[m_from..m_to) into shared slot 0, no reduce.
 * Quad complex arithmetic is heavy under libquadmath, so per-column work
 * amortizes the fork/buffer almost immediately. Serial reference unchanged. */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef __complex128 TC;



static inline TC cconj(TC z) { return conjq(z); }
#ifdef _OPENMP
static inline size_t col_start_U(ptrdiff_t j) { return (size_t)j * (size_t)(j + 1) / 2; }
static inline size_t col_start_L(ptrdiff_t j, ptrdiff_t n) {
    return (size_t)j * (size_t)(2 * n - j + 1) / 2;
}

/* Sqrt-balanced contiguous column partition (OpenBLAS tpmv_partition, mask=7,
 * min-width 16); UPPER reversed so thread 0 takes the heaviest columns. */
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
                          const TC *ap, const TC *x, TC *y)
{
    if (upper) {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_U(j);
            TC xj = x[j];
            for (ptrdiff_t i = 0; i < j; ++i) y[i] += ap[cs + (size_t)i] * xj;
            y[j] += nounit ? ap[cs + (size_t)j] * xj : xj;
        }
    } else {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_L(j, n);
            TC xj = x[j];
            y[j] += nounit ? ap[cs] * xj : xj;
            for (ptrdiff_t i = j + 1; i < n; ++i) y[i] += ap[cs + (size_t)(i - j)] * xj;
        }
    }
}

static void tpmv_kernel_T(bool upper, bool nounit, bool conj, ptrdiff_t n,
                          ptrdiff_t m_from, ptrdiff_t m_to,
                          const TC *ap, const TC *x, TC *y)
{
#define APV(k) (conj ? cconj(ap[k]) : ap[k])
    if (upper) {
        /* Diagonal at column END (cs+j): accumulate the run first, fold diag
         * last to keep the packed read stream sequential. */
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_U(j);
            TC s = 0.0Q;
            for (ptrdiff_t i = 0; i < j; ++i) s += APV(cs + (size_t)i) * x[i];
            s += nounit ? APV(cs + (size_t)j) * x[j] : x[j];
            y[j] += s;
        }
    } else {
        for (ptrdiff_t j = m_from; j < m_to; ++j) {
            size_t cs = col_start_L(j, n);
            TC s = nounit ? APV(cs) * x[j] : x[j];
            for (ptrdiff_t i = j + 1; i < n; ++i) s += APV(cs + (size_t)(i - j)) * x[i];
            y[j] += s;
        }
    }
#undef APV
}

/* Threaded out-of-place path. Returns 1 if it handled the call, 0 to fall back
 * to the serial reference. noinline so the in-place serial loops compile in a
 * clean register context. */
__attribute__((noinline))
static ptrdiff_t xtpmv_omp(bool upper, bool is_t, bool conj, bool nounit, ptrdiff_t n, ptrdiff_t incx,
                     const TC *restrict ap, TC *restrict x)
{
    ptrdiff_t nthreads = 1;
    if (n >= 50 && !omp_in_parallel()) {
        nthreads = blas_omp_max_threads();
        if (n < 500 && nthreads > 2) nthreads = 2;
    }
    if (nthreads <= 1) return 0;

    const ptrdiff_t kx = (incx < 0) ? -(n - 1) * (ptrdiff_t)incx : 0;
    TC *buf_all = (TC *)calloc((size_t)nthreads * (size_t)n, sizeof(TC));
    ptrdiff_t *range_m = (ptrdiff_t *)malloc((size_t)(nthreads + 1) * sizeof(ptrdiff_t));
    TC *xbuf = NULL;
    const TC *xptr = x;
    if (incx != 1 && buf_all && range_m) {
        xbuf = (TC *)malloc((size_t)n * sizeof(TC));
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
        TC *y = is_t ? buf_all : &buf_all[(size_t)tid * (size_t)n];
        ptrdiff_t m_from, m_to;
        if (upper) { m_from = range_m[nthreads - tid - 1]; m_to = range_m[nthreads - tid]; }
        else       { m_from = range_m[tid];               m_to = range_m[tid + 1]; }
        if (m_from < m_to) {
            if (is_t) tpmv_kernel_T(upper, nounit, conj, n, m_from, m_to, ap, xptr, y);
            else      tpmv_kernel_N(upper, nounit, n, m_from, m_to, ap, xptr, y);
        }
    }
    if (!is_t) {  /* reduce private slots into slot 0 over the touched range */
        if (upper) {
            for (ptrdiff_t t = 1; t < nthreads; ++t) {
                ptrdiff_t m_to_t = range_m[nthreads - t];
                const TC *slot = &buf_all[(size_t)t * (size_t)n];
                for (ptrdiff_t i = 0; i < m_to_t; ++i) buf_all[i] += slot[i];
            }
        } else {
            for (ptrdiff_t t = 1; t < nthreads; ++t) {
                ptrdiff_t m_from_t = range_m[t];
                const TC *slot = &buf_all[(size_t)t * (size_t)n];
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

void xtpmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n,
    const TC *restrict ap,
    TC *restrict x, ptrdiff_t incx)
{
    const TC zero = 0.0Q + 0.0Qi;
    const char UPLO = blas_up(uplo);
    const char TRANS = blas_up(trans);
    const bool noconj = (TRANS == 'T');
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    if (xtpmv_omp(UPLO == 'U', TRANS != 'N', (TRANS == 'C'), nounit, n, incx, ap, x)) return;
#endif

    if (incx == 1) {
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        const TC tmp = x[j];
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
                        const TC tmp = x[j];
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
                    TC tmp = x[j];
                    if (nounit) tmp *= (noconj ? ap[kk] : cconj(ap[kk]));
                    ptrdiff_t k = kk - 1;
                    if (noconj) for (ptrdiff_t i = j - 1; i >= 0; --i) { tmp += ap[k] * x[i]; --k; }
                    else        for (ptrdiff_t i = j - 1; i >= 0; --i) { tmp += cconj(ap[k]) * x[i]; --k; }
                    x[j] = tmp;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC tmp = x[j];
                    if (nounit) tmp *= (noconj ? ap[kk] : cconj(ap[kk]));
                    ptrdiff_t k = kk + 1;
                    if (noconj) for (ptrdiff_t i = j + 1; i < n; ++i) { tmp += ap[k] * x[i]; ++k; }
                    else        for (ptrdiff_t i = j + 1; i < n; ++i) { tmp += cconj(ap[k]) * x[i]; ++k; }
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
                        const TC tmp = x[jx];
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
                        const TC tmp = x[jx];
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
                    TC tmp = x[jx];
                    ptrdiff_t ix = jx;
                    if (nounit) tmp *= (noconj ? ap[kk] : cconj(ap[kk]));
                    for (ptrdiff_t k = kk - 1; k >= kk - j; --k) {
                        ix -= incx;
                        tmp += (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                    }
                    x[jx] = tmp;
                    jx -= incx;
                    kk -= j + 1;
                }
            } else {
                ptrdiff_t kk = 0;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TC tmp = x[jx];
                    ptrdiff_t ix = jx;
                    if (nounit) tmp *= (noconj ? ap[kk] : cconj(ap[kk]));
                    for (ptrdiff_t k = kk + 1; k < kk + n - j; ++k) {
                        ix += incx;
                        tmp += (noconj ? ap[k] : cconj(ap[k])) * x[ix];
                    }
                    x[jx] = tmp;
                    jx += incx;
                    kk += n - j;
                }
            }
        }
    }
}

EPBLAS_FACADE_TPMV(xtpmv, TC)
