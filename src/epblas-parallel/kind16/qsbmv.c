/*
 * qsbmv — kind16 (__float128) symmetric band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A symmetric with K super-(or sub-)diagonals.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Thread-entry threshold = the measured break-even where par4 beats par1.
 * Quad is compute-bound under libquadmath so the per-row band dot threads
 * profitably; below this the serial sweep wins. Faithful port of kind10 esbmv
 * (row-partitioned gather: disjoint output rows, no scratch/barrier since x,y
 * distinct). Only the threaded path uses the gather; serial stays unchanged. */
#ifndef QSBMV_OMP_MIN
#define QSBMV_OMP_MIN 128
#endif
#define QSBMV_MAX_CPUS 256

#ifdef _OPENMP
static ptrdiff_t qsbmv_omp(ptrdiff_t upper, ptrdiff_t n, ptrdiff_t k,
                     const T *restrict a, ptrdiff_t lda,
                     const T *restrict x, ptrdiff_t incx,
                     T alpha, T *restrict y, ptrdiff_t incy);
#endif

void qsbmv_(
    const char *uplo,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0Q, one = 1.0Q;
    const char UPLO = up(uplo);

    if (N == 0 || (alpha == zero && beta == one)) return;

    /* y := beta*y */
    if (beta != one) {
        int iy = (incy < 0) ? -(N - 1) * incy : 0;
        if (beta == zero) {
            for (int i = 0; i < N; ++i) { y[iy] = zero; iy += incy; }
        } else {
            for (int i = 0; i < N; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (alpha == zero) return;

#ifdef _OPENMP
    if (N >= QSBMV_OMP_MIN && blas_omp_max_threads() > 1
        && qsbmv_omp(UPLO == 'U', N, K, a, lda, x, incx, alpha, y, incy))
        return;
#endif

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            const int KP1 = K + 1;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                const int L = KP1 - 1 - j; /* row index of (i=0, j) is L+i; here L = K - j (0-based) */
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[i] += t1 * A_(L + i, j);
                    t2 += A_(L + i, j) * x[i];
                }
                y[j] += t1 * A_(K, j) + alpha * t2;
            }
        } else {
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[j];
                T t2 = zero;
                y[j] += t1 * A_(0, j);
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    y[i] += t1 * A_(i - j, j);
                    t2 += A_(i - j, j) * x[i];
                }
                y[j] += alpha * t2;
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        if (UPLO == 'U') {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                int ix = kx, iy = ky;
                const int L = K - j;
                const int i_lo = (j - K > 0) ? (j - K) : 0;
                for (int i = i_lo; i < j; ++i) {
                    y[iy] += t1 * A_(L + i, j);
                    t2 += A_(L + i, j) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] += t1 * A_(K, j) + alpha * t2;
                jx += incx; jy += incy;
                if (j >= K) { kx += incx; ky += incy; }
            }
        } else {
            int jx = kx, jy = ky;
            for (int j = 0; j < N; ++j) {
                const T t1 = alpha * x[jx];
                T t2 = zero;
                y[jy] += t1 * A_(0, j);
                int ix = jx, iy = jy;
                const int i_hi = (j + K + 1 < N) ? (j + K + 1) : N;
                for (int i = j + 1; i < i_hi; ++i) {
                    ix += incx; iy += incy;
                    y[iy] += t1 * A_(i - j, j);
                    t2 += A_(i - j, j) * x[ix];
                }
                y[jy] += alpha * t2;
                jx += incx; jy += incy;
            }
        }
    }
}

#ifdef _OPENMP
/* Row-partitioned gather kernel: y[i] for i in [lo,hi) is a register-resident
 * band dot reading x (logical order), adding alpha*s into the already-beta-
 * scaled y[i*incy]. base[K+-d*s1] walks the lda-1 anti-diagonal for same-
 * triangle neighbours; base[K-/+d] reads the reflected ones contiguously. No
 * cross-thread write dependence (x,y distinct) -> no scratch/barrier. */
static void qsbmv_rowgather(ptrdiff_t upper, ptrdiff_t n, ptrdiff_t k,
                            ptrdiff_t lo, ptrdiff_t hi,
                            const T *restrict a, ptrdiff_t lda,
                            const T *restrict x, T alpha,
                            T *restrict y, ptrdiff_t incy)
{
    const ptrdiff_t s1 = lda - 1;
    if (upper) {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            const T *base = &A_(0, i);
            T s = base[k] * x[i];
            ptrdiff_t rlen = (n - 1 - i < k) ? n - 1 - i : k;
            for (ptrdiff_t d = 1; d <= rlen; ++d) s += base[k + d * s1] * x[i + d];
            ptrdiff_t llen = (i < k) ? i : k;
            for (ptrdiff_t d = 1; d <= llen; ++d) s += base[k - d] * x[i - d];
            y[i * incy] += alpha * s;
        }
    } else {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            const T *base = &A_(0, i);
            T s = base[0] * x[i];
            ptrdiff_t llen = (i < k) ? i : k;
            for (ptrdiff_t d = 1; d <= llen; ++d) s += base[-d * s1] * x[i - d];
            ptrdiff_t rlen = (n - 1 - i < k) ? n - 1 - i : k;
            for (ptrdiff_t d = 1; d <= rlen; ++d) s += base[d] * x[i + d];
            y[i * incy] += alpha * s;
        }
    }
}

/* Threaded symmetric band matvec. Each thread owns a disjoint output-row range
 * [lo,hi): gathers y[lo,hi) reading x. No cross-thread dependence -> no barrier,
 * no scratch beyond the strided-x gather buffer. Returns 1 if handled, 0 to
 * fall back. noinline so its bookkeeping does not pressure the serial path. */
__attribute__((noinline)) static ptrdiff_t qsbmv_omp(
    ptrdiff_t upper, ptrdiff_t n, ptrdiff_t k,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    T alpha, T *restrict y, ptrdiff_t incy)
{
    if (n < QSBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > QSBMV_MAX_CPUS) nthreads = QSBMV_MAX_CPUS;

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
        ptrdiff_t lo = (n * (ptrdiff_t)tid) / nthreads;
        ptrdiff_t hi = (n * (ptrdiff_t)(tid + 1)) / nthreads;
        qsbmv_rowgather(upper, n, k, lo, hi, a, lda, xptr, alpha, y, incy);
    }

    free(xbuf);
    return 1;
}
#endif /* _OPENMP */

#undef A_
