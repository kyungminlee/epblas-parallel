/*
 * xhbmv — kind16 complex Hermitian band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A Hermitian with K super-(sub-)diagonals.
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

typedef __complex128 T;
typedef __float128 TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Thread-entry threshold = the measured break-even where par4 beats par1. The
 * heavy complex band dot amortizes the barrier-free fork almost immediately, so
 * break-even is low. Faithful port of kind10 yhbmv (row-partitioned gather:
 * disjoint output rows, no scratch/barrier since x,y distinct; real diagonal,
 * DIRECT same-triangle neighbour, CONJUGATED reflected neighbour). Only the
 * threaded path uses the gather; serial stays unchanged. */
#ifndef XHBMV_OMP_MIN
#define XHBMV_OMP_MIN 96
#endif
#define XHBMV_MAX_CPUS 256

#ifdef _OPENMP
static ptrdiff_t xhbmv_omp(bool upper, ptrdiff_t n, ptrdiff_t k,
                     const T *restrict a, ptrdiff_t lda,
                     const T *restrict x, ptrdiff_t incx,
                     T alpha, T *restrict y, ptrdiff_t incy);
#endif

void xhbmv_core(
    char uplo,
    ptrdiff_t N, ptrdiff_t K,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;
    const char UPLO = blas_up(uplo);

    if (N == 0 || (alpha == zero && beta == one)) return;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (beta == zero) for (ptrdiff_t i = 0; i < N; ++i) { y[iy] = zero; iy += incy; }
        else              for (ptrdiff_t i = 0; i < N; ++i) { y[iy] = beta * y[iy]; iy += incy; }
    }
    if (alpha == zero) return;

#ifdef _OPENMP
    if (N >= XHBMV_OMP_MIN && blas_omp_max_threads() > 1
        && xhbmv_omp(UPLO == 'U', N, K, a, lda, x, incx, alpha, y, incy))
        return;
#endif

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                const ptrdiff_t L = K - j;
                const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                for (ptrdiff_t i = i_lo; i < j; ++i) {
                    y[i] += t1 * A_(L + i, j);
                    t2 += conjq(A_(L + i, j)) * x[i];
                }
                y[j] += t1 * (TR)crealq(A_(K, j)) + alpha * t2;
            }
        } else {
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                y[j] += t1 * (TR)crealq(A_(0, j));
                const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (ptrdiff_t i = j + 1; i < i_hi; ++i) {
                    y[i] += t1 * A_(i - j, j);
                    t2 += conjq(A_(i - j, j)) * x[i];
                }
                y[j] += alpha * t2;
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'U') {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                ptrdiff_t ix = kx, iy = ky;
                const ptrdiff_t L = K - j;
                const ptrdiff_t i_lo = (j - K > 0) ? (j - K) : 0;
                for (ptrdiff_t i = i_lo; i < j; ++i) {
                    y[iy] += t1 * A_(L + i, j);
                    t2 += conjq(A_(L + i, j)) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] += t1 * (TR)crealq(A_(K, j)) + alpha * t2;
                jx += incx; jy += incy;
                if (j >= K) { kx += incx; ky += incy; }
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                y[jy] += t1 * (TR)crealq(A_(0, j));
                ptrdiff_t ix = jx, iy = jy;
                const ptrdiff_t i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (ptrdiff_t i = j + 1; i < i_hi; ++i) {
                    ix += incx; iy += incy;
                    y[iy] += t1 * A_(i - j, j);
                    t2 += conjq(A_(i - j, j)) * x[ix];
                }
                y[jy] += alpha * t2;
                jx += incx; jy += incy;
            }
        }
    }
}

#ifdef _OPENMP
/* Row-partitioned gather kernel: y[i] for i in [lo,hi) is a register-resident
 * complex band dot reading x (logical order), adding alpha*s into the already-
 * beta-scaled y[i*incy]. Real diagonal (Re*x); DIRECT same-triangle neighbour
 * (lda-1 anti-diagonal); CONJUGATED reflected neighbour (contiguous). No cross-
 * thread write dependence (x,y distinct) -> no scratch/barrier. */
static void xhbmv_rowgather(bool upper, ptrdiff_t n, ptrdiff_t k,
                            ptrdiff_t lo, ptrdiff_t hi,
                            const T *restrict a, ptrdiff_t lda,
                            const T *restrict x, T alpha,
                            T *restrict y, ptrdiff_t incy)
{
    const ptrdiff_t s1 = lda - 1;
    if (upper) {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            const T *base = &A_(0, i);
            T s = (TR)crealq(base[k]) * x[i];
            ptrdiff_t rlen = (n - 1 - i < k) ? n - 1 - i : k;
            for (ptrdiff_t d = 1; d <= rlen; ++d) s += base[k + d * s1] * x[i + d];
            ptrdiff_t llen = (i < k) ? i : k;
            for (ptrdiff_t d = 1; d <= llen; ++d) s += conjq(base[k - d]) * x[i - d];
            y[i * incy] += alpha * s;
        }
    } else {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            const T *base = &A_(0, i);
            T s = (TR)crealq(base[0]) * x[i];
            ptrdiff_t llen = (i < k) ? i : k;
            for (ptrdiff_t d = 1; d <= llen; ++d) s += base[-d * s1] * x[i - d];
            ptrdiff_t rlen = (n - 1 - i < k) ? n - 1 - i : k;
            for (ptrdiff_t d = 1; d <= rlen; ++d) s += conjq(base[d]) * x[i + d];
            y[i * incy] += alpha * s;
        }
    }
}

/* Threaded Hermitian band matvec. Each thread owns a disjoint output-row range
 * [lo,hi): gathers y[lo,hi) reading x. No cross-thread dependence -> no barrier,
 * no scratch beyond the strided-x gather buffer. Returns 1 if handled, 0 to
 * fall back. noinline so its bookkeeping does not pressure the serial path. */
__attribute__((noinline)) static ptrdiff_t xhbmv_omp(
    bool upper, ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    T alpha, T *restrict y, ptrdiff_t incy)
{
    if (n < XHBMV_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > XHBMV_MAX_CPUS) nthreads = XHBMV_MAX_CPUS;

    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)n * sizeof(T));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t lo = blas_part_bound(n, tid, nthreads);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nthreads);
        xhbmv_rowgather(upper, n, k, lo, hi, a, lda, xptr, alpha, y, incy);
    }

    free(xbuf);
    return 1;
}
#endif /* _OPENMP */


EPBLAS_FACADE_SBMV(xhbmv, T)

#undef A_
