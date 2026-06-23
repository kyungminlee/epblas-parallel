/*
 * qsbmv — kind16 (__float128) symmetric band matrix-vector multiply.
 *   y := alpha*A*x + beta*y, A symmetric with K super-(or sub-)diagonals.
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

typedef __float128 TR;


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
static ptrdiff_t qsbmv_omp(bool upper, ptrdiff_t n, ptrdiff_t k,
                     const TR *restrict a, ptrdiff_t lda,
                     const TR *restrict x, ptrdiff_t incx,
                     TR alpha, TR *restrict y, ptrdiff_t incy);
#endif

void qsbmv_core(
    char uplo,
    ptrdiff_t n, ptrdiff_t k,
    const TR *alpha_,
    const TR *restrict a, ptrdiff_t lda,
    const TR *restrict x, ptrdiff_t incx,
    const TR *beta_,
    TR *restrict y, ptrdiff_t incy)
{
    const TR alpha = *alpha_, beta = *beta_;
    const TR zero = 0.0Q, one = 1.0Q;
    const char UPLO = blas_up(uplo);

    if (n == 0 || (alpha == zero && beta == one)) return;

    /* y := beta*y */
    if (beta != one) {
        ptrdiff_t iy = (incy < 0) ? -(n - 1) * incy : 0;
        if (beta == zero) {
            for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = zero; iy += incy; }
        } else {
            for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (alpha == zero) return;

#ifdef _OPENMP
    if (n >= QSBMV_OMP_MIN && blas_omp_max_threads() > 1
        && qsbmv_omp(UPLO == 'U', n, k, a, lda, x, incx, alpha, y, incy))
        return;
#endif

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
            const ptrdiff_t KP1 = k + 1;
            for (ptrdiff_t j = 0; j < n; ++j) {
                const TR t1 = alpha * x[j];
                TR t2 = zero;
                const ptrdiff_t L = KP1 - 1 - j; /* row index of (i=0, j) is L+i; here L = K - j (0-based) */
                const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                for (ptrdiff_t i = i_lo; i < j; ++i) {
                    y[i] += t1 * A_(L + i, j);
                    t2 += A_(L + i, j) * x[i];
                }
                y[j] += t1 * A_(k, j) + alpha * t2;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                const TR t1 = alpha * x[j];
                TR t2 = zero;
                y[j] += t1 * A_(0, j);
                const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                for (ptrdiff_t i = j + 1; i < i_hi; ++i) {
                    y[i] += t1 * A_(i - j, j);
                    t2 += A_(i - j, j) * x[i];
                }
                y[j] += alpha * t2;
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        if (UPLO == 'U') {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < n; ++j) {
                const TR t1 = alpha * x[jx];
                TR t2 = zero;
                ptrdiff_t ix = kx, iy = ky;
                const ptrdiff_t L = k - j;
                const ptrdiff_t i_lo = (j - k > 0) ? (j - k) : 0;
                for (ptrdiff_t i = i_lo; i < j; ++i) {
                    y[iy] += t1 * A_(L + i, j);
                    t2 += A_(L + i, j) * x[ix];
                    ix += incx; iy += incy;
                }
                y[jy] += t1 * A_(k, j) + alpha * t2;
                jx += incx; jy += incy;
                if (j >= k) { kx += incx; ky += incy; }
            }
        } else {
            ptrdiff_t jx = kx, jy = ky;
            for (ptrdiff_t j = 0; j < n; ++j) {
                const TR t1 = alpha * x[jx];
                TR t2 = zero;
                y[jy] += t1 * A_(0, j);
                ptrdiff_t ix = jx, iy = jy;
                const ptrdiff_t i_hi = (j + k + 1 < n) ? (j + k + 1) : n;
                for (ptrdiff_t i = j + 1; i < i_hi; ++i) {
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
static void qsbmv_rowgather(bool upper, ptrdiff_t n, ptrdiff_t k,
                            ptrdiff_t lo, ptrdiff_t hi,
                            const TR *restrict a, ptrdiff_t lda,
                            const TR *restrict x, TR alpha,
                            TR *restrict y, ptrdiff_t incy)
{
    const ptrdiff_t s1 = lda - 1;
    if (upper) {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            const TR *base = &A_(0, i);
            TR s = base[k] * x[i];
            ptrdiff_t rlen = (n - 1 - i < k) ? n - 1 - i : k;
            for (ptrdiff_t d = 1; d <= rlen; ++d) s += base[k + d * s1] * x[i + d];
            ptrdiff_t llen = (i < k) ? i : k;
            for (ptrdiff_t d = 1; d <= llen; ++d) s += base[k - d] * x[i - d];
            y[i * incy] += alpha * s;
        }
    } else {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            const TR *base = &A_(0, i);
            TR s = base[0] * x[i];
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
    bool upper, ptrdiff_t n, ptrdiff_t k,
    const TR *restrict a, ptrdiff_t lda,
    const TR *restrict x, ptrdiff_t incx,
    TR alpha, TR *restrict y, ptrdiff_t incy)
{
    if (n < QSBMV_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > QSBMV_MAX_CPUS) nthreads = QSBMV_MAX_CPUS;

    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

    const TR *xptr = x;
    TR *xbuf = NULL;
    if (incx != 1) {
        xbuf = (TR *)malloc((size_t)n * sizeof(TR));
        if (!xbuf) return 0;
        for (ptrdiff_t i = 0; i < n; ++i) xbuf[i] = x[i * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t lo = blas_part_bound(n, tid, nthreads);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nthreads);
        qsbmv_rowgather(upper, n, k, lo, hi, a, lda, xptr, alpha, y, incy);
    }

    free(xbuf);
    return 1;
}
#endif /* _OPENMP */


EPBLAS_FACADE_SBMV(qsbmv, TR)

#undef A_
