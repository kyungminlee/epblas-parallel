/*
 * qtbmv — kind16 (__float128) triangular band matrix-vector.
 *   x := A*x or A^T*x, A triangular band with K+1 diagonals.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

typedef __float128 T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Thread-entry threshold. x:=op(A)*x is a pure matvec (no solve dependency):
 * each OUTPUT row is an independent band dot, so the threaded path gives thread
 * t a disjoint row range, gathers into a shared scratch y (one barrier), then
 * copies its range back to x. Quad is compute-bound under libquadmath so the
 * heavy band dot amortizes the fork/malloc/barrier almost immediately -> break-
 * even far below the fp80 etbmv's 768/1024. Faithful port of kind10 etbmv's
 * threaded path; the serial Netlib sweep stays unchanged. */
#ifndef QTBMV_OMP_MIN
#define QTBMV_OMP_MIN 128
#endif
#define QTBMV_MAX_CPUS 256

#ifdef _OPENMP
static ptrdiff_t qtbmv_omp(bool upper, ptrdiff_t trans, bool nounit, ptrdiff_t n, ptrdiff_t k,
                     const T *restrict a, ptrdiff_t lda, T *restrict x, ptrdiff_t incx);
#endif

void qtbmv_core(
    char uplo, char trans, char diag,
    ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx)
{
    const T zero = 0.0Q;
    const char UPLO = blas_up(uplo);
    char TRANS = blas_up(trans);
    if (TRANS == 'C') TRANS = 'T';
    const bool nounit = (blas_up(diag) != 'U');

    if (n == 0) return;

#ifdef _OPENMP
    if (n >= QTBMV_OMP_MIN && blas_omp_max_threads() > 1
        && qtbmv_omp(UPLO == 'U', TRANS == 'T', nounit, n, k, a, lda, x, incx))
        return;
#endif

    if (incx == 1) {
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero) {
                        const T tmp = x[j];
                        const ptrdiff_t L = k - j;
                        const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                        for (ptrdiff_t i = i_lo; i < j; ++i) x[i] += tmp * A_(L + i, j);
                        if (nounit) x[j] *= A_(k, j);
                    }
                }
            } else {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[j] != zero) {
                        const T tmp = x[j];
                        const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                        for (ptrdiff_t i = i_hi - 1; i > j; --i) x[i] += tmp * A_(i - j, j);
                        if (nounit) x[j] *= A_(0, j);
                    }
                }
            }
        } else {
            if (UPLO == 'U') {
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T tmp = x[j];
                    const ptrdiff_t L = k - j;
                    if (nounit) tmp *= A_(k, j);
                    const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                    for (ptrdiff_t i = j - 1; i >= i_lo; --i) tmp += A_(L + i, j) * x[i];
                    x[j] = tmp;
                }
            } else {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T tmp = x[j];
                    if (nounit) tmp *= A_(0, j);
                    const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                    for (ptrdiff_t i = j + 1; i < i_hi; ++i) tmp += A_(i - j, j) * x[i];
                    x[j] = tmp;
                }
            }
        }
    } else {
        /* Non-unit stride — transcribe Netlib. */
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        if (TRANS == 'N') {
            if (UPLO == 'U') {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[jx] != zero) {
                        const T tmp = x[jx];
                        ptrdiff_t ix = kx;
                        const ptrdiff_t L = k - j;
                        const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                        for (ptrdiff_t i = i_lo; i < j; ++i) {
                            x[ix] += tmp * A_(L + i, j);
                            ix += incx;
                        }
                        if (nounit) x[jx] *= A_(k, j);
                    }
                    jx += incx;
                    if (j >= k) kx += incx;
                }
            } else {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    if (x[jx] != zero) {
                        const T tmp = x[jx];
                        ptrdiff_t ix = kx;
                        const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                        for (ptrdiff_t i = i_hi - 1; i > j; --i) {
                            x[ix] += tmp * A_(i - j, j);
                            ix -= incx;
                        }
                        if (nounit) x[jx] *= A_(0, j);
                    }
                    jx -= incx;
                    if ((n - 1 - j) >= k) kx -= incx;
                }
            }
        } else {
            if (UPLO == 'U') {
                kx += (n - 1) * incx;
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = n - 1; j >= 0; --j) {
                    T tmp = x[jx];
                    kx -= incx;
                    ptrdiff_t ix = kx;
                    const ptrdiff_t L = k - j;
                    if (nounit) tmp *= A_(k, j);
                    const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                    for (ptrdiff_t i = j - 1; i >= i_lo; --i) {
                        tmp += A_(L + i, j) * x[ix];
                        ix -= incx;
                    }
                    x[jx] = tmp;
                    jx -= incx;
                }
            } else {
                ptrdiff_t jx = kx;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    T tmp = x[jx];
                    kx += incx;
                    ptrdiff_t ix = kx;
                    if (nounit) tmp *= A_(0, j);
                    const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                    for (ptrdiff_t i = j + 1; i < i_hi; ++i) {
                        tmp += A_(i - j, j) * x[ix];
                        ix += incx;
                    }
                    x[jx] = tmp;
                    jx += incx;
                }
            }
        }
    }
}

#ifdef _OPENMP
/* Row-partitioned gather kernel: y[i] for i in [lo,hi) is a band dot reading x
 * globally (never written here), accumulated in a register-resident scalar,
 * stored once into the disjoint scratch y. Branch hoisted out of the i-loop;
 * lda-1 anti-diagonal stride (NoTrans) vs contiguous (Trans). */
static void qtbmv_rowgather(bool upper, ptrdiff_t trans, bool nounit,
                            ptrdiff_t n, ptrdiff_t k, ptrdiff_t lo, ptrdiff_t hi,
                            const T *restrict a, ptrdiff_t lda,
                            const T *restrict x, T *restrict y)
{
    const ptrdiff_t s1 = lda - 1;
    if (!trans) {
        if (upper) {
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const T *base = &A_(0, i);
                ptrdiff_t len = (n - 1 - i < k) ? n - 1 - i : k;
                T s = nounit ? base[k] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[k + d * s1] * x[i + d];
                y[i] = s;
            }
        } else {
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const T *base = &A_(0, i);
                ptrdiff_t len = (i < k) ? i : k;
                T s = nounit ? base[0] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[-d * s1] * x[i - d];
                y[i] = s;
            }
        }
    } else {
        if (upper) {
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const T *base = &A_(0, i);
                ptrdiff_t len = (i < k) ? i : k;
                T s = nounit ? base[k] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[k - d] * x[i - d];
                y[i] = s;
            }
        } else {
            for (ptrdiff_t i = lo; i < hi; ++i) {
                const T *base = &A_(0, i);
                ptrdiff_t len = (n - 1 - i < k) ? n - 1 - i : k;
                T s = nounit ? base[0] * x[i] : x[i];
                for (ptrdiff_t d = 1; d <= len; ++d) s += base[d] * x[i + d];
                y[i] = s;
            }
        }
    }
}

/* Threaded triangular band matvec. Each thread owns a disjoint output-row range
 * [lo,hi): gathers y[lo,hi) reading x, a barrier, then copies its range back to
 * x. No cross-thread dependence beyond the single read/write barrier. Returns 1
 * if handled, 0 to fall back. noinline so its bookkeeping does not pressure the
 * serial path. */
__attribute__((noinline)) static ptrdiff_t qtbmv_omp(
    bool upper, ptrdiff_t trans, bool nounit, ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda, T *restrict x, ptrdiff_t incx)
{
    if (n < QTBMV_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > QTBMV_MAX_CPUS) nthreads = QTBMV_MAX_CPUS;

    if (incx < 0) x -= (n - 1) * incx;

    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)n * sizeof(T));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
        xptr = xbuf;
    }
    T *y = (T *)malloc((size_t)n * sizeof(T));
    if (!y) { free(xbuf); return 0; }

    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t lo = blas_part_bound(n, tid, nthreads);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nthreads);
        qtbmv_rowgather(upper, trans, nounit, n, k, lo, hi, a, lda, xptr, y);
        #pragma omp barrier
        if (incx == 1) for (ptrdiff_t i = lo; i < hi; ++i) x[i] = y[i];
        else           for (ptrdiff_t i = lo; i < hi; ++i) x[i * incx] = y[i];
    }

    free(y); free(xbuf);
    return 1;
}
#endif /* _OPENMP */

EPBLAS_FACADE_TBMV(qtbmv, T)

#undef A_
