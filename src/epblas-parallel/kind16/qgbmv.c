/*
 * qgbmv — kind16 (__float128) general band matrix-vector multiply.
 *   y := alpha*A*x + beta*y  or  y := alpha*A^T*x + beta*y
 * Band storage: A(i,j) at AB[(ku + i - j) + j*lda], for max(0,j-ku) <= i <= min(M,j+kl).
 *
 * NoTrans (y := A*x): every OUTPUT row is an independent dot over the row's band,
 * so we GATHER -- each y[i] is accumulated in a register-resident scalar and
 * stored once (y[i] += alpha*s), never the column SCATTER that reference gbmv
 * uses. For row i, A(i,j) = a[(KU+i) + j*(lda-1)], so with base = a + (KU+i) and
 * s1 = lda-1 the row is a unit-j dot base[j*s1]*x[j] over j in row i's band.
 * Because x and y are DISTINCT arrays (BLAS forbids gbmv aliasing), the threaded
 * path partitions output rows [lo,hi) with NO scratch, NO reduction, NO barrier.
 *
 * Per-output summation order is preserved (ascending j == the scatter's column
 * order), so the serial gather is bit-identical to the old scatter.
 *
 * Faithful port of kind10 egbmv. Partition bounds via blas_part_bound (forms the
 * product in __int128 so total*idx cannot overflow under ILP64).
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#ifndef QGBMV_OMP_MIN
#define QGBMV_OMP_MIN 256
#endif
#define QGBMV_MAX_CPUS 256

typedef __float128 T;

static inline char up(char c) {
    return (char)toupper((unsigned char)c);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifdef _OPENMP
static ptrdiff_t qgbmv_n_omp(ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
                       const T *restrict a, ptrdiff_t lda,
                       const T *restrict x, ptrdiff_t incx,
                       T alpha, T *restrict y, ptrdiff_t incy);
static ptrdiff_t qgbmv_t_omp(ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
                       const T *restrict a, ptrdiff_t lda,
                       const T *restrict x, ptrdiff_t incx,
                       T alpha, T *restrict y, ptrdiff_t incy);
#endif

void qgbmv_core(
    char trans,
    ptrdiff_t M, ptrdiff_t N,
    ptrdiff_t KL, ptrdiff_t KU,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0Q, one = 1.0Q;
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (M == 0 || N == 0 || (alpha == zero && beta == one)) return;

    const ptrdiff_t leny = (TR == 'N') ? M : N;
    const ptrdiff_t lenx = (TR == 'N') ? N : M;

    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(leny - 1) * incy : 0;
        if (beta == zero) {
            for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] = zero; iy += incy; }
        } else {
            for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (alpha == zero) return;

    const ptrdiff_t s1 = (ptrdiff_t)lda - 1;

    if (TR == 'N') {
#ifdef _OPENMP
        if (M >= QGBMV_OMP_MIN && blas_omp_max_threads() > 1
            && qgbmv_n_omp(M, N, KL, KU, a, lda, x, incx, alpha, y, incy))
            return;
#endif
        /* Serial NoTrans row-gather: y[i] = alpha * Σ_j A(i,j)*x[j]. */
        if (incx == 1 && incy == 1) {
            for (ptrdiff_t i = 0; i < M; ++i) {
                const ptrdiff_t j_lo = (i - KL > 0) ? (i - KL) : 0;
                const ptrdiff_t j_hi = (i + KU + 1 < N) ? (i + KU + 1) : N;
                const T *base = a + (KU + i);
                T s = zero;
                for (ptrdiff_t j = j_lo; j < j_hi; ++j) s += base[(ptrdiff_t)j * s1] * x[j];
                y[i] += alpha * s;
            }
        } else {
            const ptrdiff_t ix0 = (incx < 0) ? -(ptrdiff_t)(lenx - 1) * incx : 0;
            const ptrdiff_t iy0 = (incy < 0) ? -(ptrdiff_t)(leny - 1) * incy : 0;
            for (ptrdiff_t i = 0; i < M; ++i) {
                const ptrdiff_t j_lo = (i - KL > 0) ? (i - KL) : 0;
                const ptrdiff_t j_hi = (i + KU + 1 < N) ? (i + KU + 1) : N;
                const T *base = a + (KU + i);
                T s = zero;
                ptrdiff_t xx = ix0 + (ptrdiff_t)j_lo * incx;
                for (ptrdiff_t j = j_lo; j < j_hi; ++j) { s += base[(ptrdiff_t)j * s1] * x[xx]; xx += incx; }
                y[iy0 + (ptrdiff_t)i * incy] += alpha * s;
            }
        }
    } else if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= QGBMV_OMP_MIN && blas_omp_max_threads() > 1);
#else
        const ptrdiff_t use_omp = 0;
#endif
#define QGBMV_T_BODY                                                         \
        for (ptrdiff_t j = 0; j < N; ++j) {                                        \
            T s = zero;                                                      \
            const ptrdiff_t i_lo = (j - KU > 0) ? (j - KU) : 0;                    \
            const ptrdiff_t i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;            \
            const ptrdiff_t k = KU - j;                                            \
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) s += A_(k + i, j) * x[i];      \
            y[j] += alpha * s;                                               \
        }
        if (use_omp) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            QGBMV_T_BODY
        } else {
            QGBMV_T_BODY
        }
#undef QGBMV_T_BODY
    } else {
#ifdef _OPENMP
        if (N >= QGBMV_OMP_MIN && blas_omp_max_threads() > 1
            && qgbmv_t_omp(M, N, KL, KU, a, lda, x, incx, alpha, y, incy))
            return;
#endif
        /* Strided Trans gather (serial). */
        ptrdiff_t kx = (incx < 0) ? -(lenx - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(leny - 1) * incy : 0;
        ptrdiff_t jy = ky;
        for (ptrdiff_t j = 0; j < N; ++j) {
            T s = zero;
            ptrdiff_t ix = kx;
            const ptrdiff_t i_lo = (j - KU > 0) ? (j - KU) : 0;
            const ptrdiff_t i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const ptrdiff_t k = KU - j;
            const T *col = &A_(k + i_lo, j);
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                s += *col++ * x[ix];
                ix += incx;
            }
            y[jy] += alpha * s;
            jy += incy;
            if (j >= KU) kx += incx;
        }
    }
}

#ifdef _OPENMP
/* Row-partitioned NoTrans gather kernel: y[i] for i in [lo,hi). */
static void qgbmv_n_rowgather(ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
                             ptrdiff_t lo, ptrdiff_t hi,
                             const T *restrict a, ptrdiff_t lda,
                             const T *restrict x, T alpha,
                             T *restrict y, ptrdiff_t incy)
{
    const ptrdiff_t s1 = lda - 1;
    for (ptrdiff_t i = lo; i < hi; ++i) {
        ptrdiff_t j_lo = (i - kl > 0) ? (i - kl) : 0;
        ptrdiff_t j_hi = (i + ku + 1 < n) ? (i + ku + 1) : n;
        const T *base = a + (ku + i);
        T s = 0.0Q;
        for (ptrdiff_t j = j_lo; j < j_hi; ++j) s += base[j * s1] * x[j];
        y[i * incy] += alpha * s;
    }
}

__attribute__((noinline)) static ptrdiff_t qgbmv_n_omp(
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    T alpha, T *restrict y, ptrdiff_t incy)
{
    if (m < QGBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > QGBMV_MAX_CPUS) nthreads = QGBMV_MAX_CPUS;

    if (incx < 0) x -= (ptrdiff_t)(n - 1) * incx;
    if (incy < 0) y -= (ptrdiff_t)(m - 1) * incy;

    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)n * sizeof(T));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[(ptrdiff_t)i * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t lo = blas_part_bound(m, tid, nthreads);
        ptrdiff_t hi = blas_part_bound(m, tid + 1, nthreads);
        qgbmv_n_rowgather(m, n, kl, ku, lo, hi, a, lda, xptr, alpha, y, incy);
    }

    free(xbuf);
    return 1;
}

/* Column-partitioned Trans gather kernel: y[j] = alpha * Σ_i A(i,j)*x[i]. */
static void qgbmv_t_colgather(ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
                             ptrdiff_t lo, ptrdiff_t hi,
                             const T *restrict a, ptrdiff_t lda,
                             const T *restrict x, T alpha,
                             T *restrict y, ptrdiff_t incy)
{
    for (ptrdiff_t j = lo; j < hi; ++j) {
        const ptrdiff_t i_lo = (j - ku > 0) ? (j - ku) : 0;
        const ptrdiff_t i_hi = (j + kl + 1 < m) ? (j + kl + 1) : m;
        const ptrdiff_t k = ku - j;
        const T *col = &A_(k + i_lo, j);
        T s = 0.0Q;
        for (ptrdiff_t i = i_lo; i < i_hi; ++i) s += *col++ * x[i];
        y[j * incy] += alpha * s;
    }
}

__attribute__((noinline)) static ptrdiff_t qgbmv_t_omp(
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t kl, ptrdiff_t ku,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    T alpha, T *restrict y, ptrdiff_t incy)
{
    if (n < QGBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > QGBMV_MAX_CPUS) nthreads = QGBMV_MAX_CPUS;

    if (incx < 0) x -= (ptrdiff_t)(m - 1) * incx;
    if (incy < 0) y -= (ptrdiff_t)(n - 1) * incy;

    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)m * sizeof(T));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < m; ++i) xbuf[i] = x[(ptrdiff_t)i * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t lo = blas_part_bound(n, tid, nthreads);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nthreads);
        qgbmv_t_colgather(m, n, kl, ku, lo, hi, a, lda, xptr, alpha, y, incy);
    }

    free(xbuf);
    return 1;
}
#endif /* _OPENMP */

EPBLAS_FACADE_GBMV(qgbmv, T)

#undef A_
