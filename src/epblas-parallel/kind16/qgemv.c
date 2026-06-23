/*
 * qgemv — kind16 (__float128) general matrix-vector multiply.
 *
 *   y := alpha · A · x + beta · y          (TRANS='N')   A is M×N
 *   y := alpha · Aᵀ · x + beta · y         (TRANS='T'/'C') A is M×N, y is N
 *
 * One external-linkage by-value core (`qgemv_core`) drives both Fortran-ABI
 * facades (qgemv_ / qgemv_64_, via EPBLAS_FACADE_GEMV) AND the trsv/tbsv/tpsv
 * cross-calls (the trailing GEMV bypasses the by-ref facade). The core opens
 * its own `#pragma omp parallel` region over the M-axis (TRANS='N') or N-axis
 * (TRANS='T'); it falls back to serial when invoked inside an existing parallel
 * region (omp_in_parallel()), so it is safe to call from etrsv_blocked.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define QGEMV_OMP_MIN 64

typedef __float128 TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Pure serial body for TRANS='N', stride-1: y[i_lo:i_hi] += alpha * A[i_lo:i_hi, :] * x.
 * Each thread (or the lone serial caller) writes a disjoint slice of y. */
static void qgemv_n_stride1_slice(
    ptrdiff_t n, ptrdiff_t i_lo, ptrdiff_t i_hi,
    TR alpha,
    const TR *restrict a, ptrdiff_t lda,
    const TR *restrict x, TR *restrict y)
{
    const TR zero = 0.0Q;
    for (ptrdiff_t j = 0; j < n; ++j) {
        const TR xj = x[j];
        if (xj != zero) {
            const TR t = alpha * xj;
            const TR *aj = &A_(0, j);
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) y[i] += t * aj[i];
        }
    }
}

/* Pure serial body for TRANS ∈ {'T','C'}, stride-1: y[j_lo:j_hi] += alpha * (A^T * x)[j_lo:j_hi].
 * Each thread (or the lone serial caller) writes a disjoint slice of y. */
static void qgemv_t_stride1_slice(
    ptrdiff_t m, ptrdiff_t j_lo, ptrdiff_t j_hi,
    TR alpha,
    const TR *restrict a, ptrdiff_t lda,
    const TR *restrict x, TR *restrict y)
{
    const TR zero = 0.0Q;
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        const TR *aj = &A_(0, j);
        TR s = zero;
        for (ptrdiff_t i = 0; i < m; ++i) s += aj[i] * x[i];
        y[j] += alpha * s;
    }
}

/* General-stride slice (incx≠1 or incy≠1). For TRANS='N' [lo,hi) is the disjoint
 * output-row slice; for TRANS≠'N' it is the disjoint output-index (j) slice. Each
 * output element is written by one thread in the same per-element order as the
 * full serial loop → race-free and bit-exact (iy0/jy0/ix recomputed). */
static void qgemv_general_stride_slice(
    ptrdiff_t m, ptrdiff_t n, char TRANS,
    TR alpha, const TR *a, ptrdiff_t lda,
    const TR *x, ptrdiff_t incx, TR *y, ptrdiff_t incy, ptrdiff_t lo, ptrdiff_t hi)
{
    const TR zero = 0.0Q;
    if (TRANS == 'N') {
        const ptrdiff_t iy0 = (incy < 0) ? -(m - 1) * incy : 0;
        ptrdiff_t jx = (incx < 0) ? -(n - 1) * incx : 0;
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR xj = x[jx];
            if (xj != zero) {
                const TR t = alpha * xj;
                ptrdiff_t iy = iy0 + lo * incy;
                for (ptrdiff_t i = lo; i < hi; ++i) {
                    y[iy] += t * A_(i, j);
                    iy += incy;
                }
            }
            jx += incx;
        }
    } else {
        const ptrdiff_t jy0 = (incy < 0) ? -(n - 1) * incy : 0;
        for (ptrdiff_t j = lo; j < hi; ++j) {
            TR s = zero;
            ptrdiff_t ix = (incx < 0) ? -(m - 1) * incx : 0;
            for (ptrdiff_t i = 0; i < m; ++i) {
                s += A_(i, j) * x[ix];
                ix += incx;
            }
            y[jy0 + j * incy] += alpha * s;
        }
    }
}

/* Apply beta scaling to y[0:leny] (with stride incy). */
static void qgemv_apply_beta(ptrdiff_t leny, ptrdiff_t incy, TR beta, TR *y)
{
    const TR zero = 0.0Q, one = 1.0Q;
    if (beta == one) return;
    ptrdiff_t iy = (incy < 0) ? -(leny - 1) * incy : 0;
    for (ptrdiff_t i = 0; i < leny; ++i) {
        if (beta == zero) y[iy] = zero;
        else              y[iy] *= beta;
        iy += incy;
    }
}

/* External linkage: qtrsv/qtbsv/qtpsv call qgemv_core directly (the §1.3
 * cross-call retarget) so the trailing GEMV bypasses the by-ref facade.
 * Opens its own parallel region; falls back to serial if invoked from inside
 * another parallel region. */
void qgemv_core(
    char trans,
    ptrdiff_t m, ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict a, ptrdiff_t lda,
    const TR *restrict x, ptrdiff_t incx,
    const TR *beta_,
    TR *restrict y, ptrdiff_t incy)
{
    const TR alpha = *alpha_, beta = *beta_;
    char TRANS = blas_up(trans);
    if (TRANS == 'C') TRANS = 'T';

    if (m == 0 || n == 0) return;

    const TR zero = 0.0Q;
    const ptrdiff_t leny = (TRANS == 'N') ? m : n;

    qgemv_apply_beta(leny, incy, beta, y);

    if (alpha == zero) return;

#ifdef _OPENMP
    const bool in_parallel = omp_in_parallel();
#else
    const bool in_parallel = 0;
#endif

    if (TRANS == 'N' && incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (m >= QGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            ptrdiff_t tid = 0, nth = 1;
            if (use_omp) { tid = omp_get_thread_num(); nth = omp_get_num_threads(); }
            const ptrdiff_t i_lo = blas_part_bound(m, tid, nth);
            const ptrdiff_t i_hi = blas_part_bound(m, tid + 1, nth);
            qgemv_n_stride1_slice(n, i_lo, i_hi, alpha, a, lda, x, y);
        }
#else
        qgemv_n_stride1_slice(n, 0, m, alpha, a, lda, x, y);
#endif
    } else if (TRANS != 'N' && incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (n >= QGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            ptrdiff_t tid = 0, nth = 1;
            if (use_omp) { tid = omp_get_thread_num(); nth = omp_get_num_threads(); }
            const ptrdiff_t j_lo = blas_part_bound(n, tid, nth);
            const ptrdiff_t j_hi = blas_part_bound(n, tid + 1, nth);
            qgemv_t_stride1_slice(m, j_lo, j_hi, alpha, a, lda, x, y);
        }
#else
        qgemv_t_stride1_slice(m, 0, n, alpha, a, lda, x, y);
#endif
    } else {
#ifdef _OPENMP
        const ptrdiff_t span = (TRANS == 'N') ? m : n;
        const bool use_omp = (span >= QGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            ptrdiff_t tid = 0, nth = 1;
            if (use_omp) { tid = omp_get_thread_num(); nth = omp_get_num_threads(); }
            const ptrdiff_t lo = blas_part_bound(span, tid, nth);
            const ptrdiff_t hi = blas_part_bound(span, tid + 1, nth);
            qgemv_general_stride_slice(m, n, TRANS, alpha, a, lda, x, incx, y, incy, lo, hi);
        }
#else
        qgemv_general_stride_slice(m, n, TRANS, alpha, a, lda, x, incx, y, incy,
                                   0, (TRANS == 'N') ? m : n);
#endif
    }
}

EPBLAS_FACADE_GEMV(qgemv, TR)

#undef A_
