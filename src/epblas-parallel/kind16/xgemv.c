/*
 * xgemv — kind16 complex (__complex128) general matrix-vector multiply.
 *
 *   y := alpha · A · x  + beta · y          (TRANS='N')   A is M×N
 *   y := alpha · Aᵀ · x + beta · y          (TRANS='T')   A is M×N, y is N
 *   y := alpha · Aᴴ · x + beta · y          (TRANS='C')   conjugated
 *
 * One external-linkage by-value core (`xgemv_core`) drives both Fortran-ABI
 * facades (xgemv_ / xgemv_64_, via EPBLAS_FACADE_GEMV) AND the trsv/tbsv/tpsv
 * cross-calls (the trailing GEMV bypasses the by-ref facade). The core opens
 * its own `#pragma omp parallel` region over the M-axis (TRANS='N') or N-axis
 * (TRANS='T'/'C'); it falls back to serial when invoked inside an existing
 * parallel region (omp_in_parallel()), so it is safe to call from xtrsv_blocked.
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

#define XGEMV_OMP_MIN 64

typedef __complex128 TC;


static inline TC cconj(TC z) { return conjq(z); }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Pure serial body for TRANS='N', stride-1: y[i_lo:i_hi] += alpha * A[i_lo:i_hi, :] * x.
 * Each thread (or the lone serial caller) writes a disjoint slice of y. */
static void xgemv_n_stride1_slice(
    ptrdiff_t n, ptrdiff_t i_lo, ptrdiff_t i_hi,
    TC alpha,
    const TC *restrict a, ptrdiff_t lda,
    const TC *restrict x, TC *restrict y)
{
    const TC zero = 0.0Q + 0.0Qi;
    for (ptrdiff_t j = 0; j < n; ++j) {
        const TC xj = x[j];
        if (xj != zero) {
            const TC t = alpha * xj;
            const TC *aj = &A_(0, j);
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) y[i] += t * aj[i];
        }
    }
}

/* Pure serial body for TRANS ∈ {'T','C'}, stride-1: y[j_lo:j_hi] += alpha * (op(A) * x)[j_lo:j_hi].
 * Each thread (or the lone serial caller) writes a disjoint slice of y. */
static void xgemv_tc_stride1_slice(
    ptrdiff_t m, ptrdiff_t j_lo, ptrdiff_t j_hi, bool conj_a,
    TC alpha,
    const TC *restrict a, ptrdiff_t lda,
    const TC *restrict x, TC *restrict y)
{
    const TC zero = 0.0Q + 0.0Qi;
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        const TC *aj = &A_(0, j);
        TC s = zero;
        if (conj_a) {
            for (ptrdiff_t i = 0; i < m; ++i) s += cconj(aj[i]) * x[i];
        } else {
            for (ptrdiff_t i = 0; i < m; ++i) s += aj[i] * x[i];
        }
        y[j] += alpha * s;
    }
}

/* General-stride slice (incx≠1 or incy≠1). For TRANS='N' [lo,hi) is the disjoint
 * output-row slice; for TRANS≠'N' it is the disjoint output-index (j) slice. Each
 * output element is written by one thread in the same per-element order as the
 * full serial loop → race-free and bit-exact (iy0/jy0/ix recomputed). */
static void xgemv_general_stride_slice(
    ptrdiff_t m, ptrdiff_t n, char TRANS, bool conj_a,
    TC alpha, const TC *a, ptrdiff_t lda,
    const TC *x, ptrdiff_t incx, TC *y, ptrdiff_t incy, ptrdiff_t lo, ptrdiff_t hi)
{
    const TC zero = 0.0Q + 0.0Qi;
    if (TRANS == 'N') {
        const ptrdiff_t iy0 = (incy < 0) ? -(m - 1) * incy : 0;
        ptrdiff_t jx = (incx < 0) ? -(n - 1) * incx : 0;
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TC xj = x[jx];
            if (xj != zero) {
                const TC t = alpha * xj;
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
            TC s = zero;
            ptrdiff_t ix = (incx < 0) ? -(m - 1) * incx : 0;
            for (ptrdiff_t i = 0; i < m; ++i) {
                s += (conj_a ? cconj(A_(i, j)) : A_(i, j)) * x[ix];
                ix += incx;
            }
            y[jy0 + j * incy] += alpha * s;
        }
    }
}

/* Apply beta scaling to y[0:leny] (with stride incy). */
static void xgemv_apply_beta(ptrdiff_t leny, ptrdiff_t incy, TC beta, TC *y)
{
    const TC zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;
    if (beta == one) return;
    ptrdiff_t iy = (incy < 0) ? -(leny - 1) * incy : 0;
    for (ptrdiff_t i = 0; i < leny; ++i) {
        if (beta == zero) y[iy] = zero;
        else              y[iy] *= beta;
        iy += incy;
    }
}

/* External linkage: xtrsv/xtbsv/xtpsv call xgemv_core directly (the §1.3
 * cross-call retarget) so the trailing GEMV bypasses the by-ref facade.
 * Opens its own parallel region; falls back to serial if invoked from inside
 * another parallel region. */
void xgemv_core(
    char trans,
    ptrdiff_t m, ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict a, ptrdiff_t lda,
    const TC *restrict x, ptrdiff_t incx,
    const TC *beta_,
    TC *restrict y, ptrdiff_t incy)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char TRANS = blas_up(trans);

    if (m == 0 || n == 0) return;

    const TC zero = 0.0Q + 0.0Qi;
    const ptrdiff_t leny = (TRANS == 'N') ? m : n;

    xgemv_apply_beta(leny, incy, beta, y);

    if (alpha == zero) return;

#ifdef _OPENMP
    const bool in_parallel = omp_in_parallel();
#else
    const bool in_parallel = 0;
#endif

    if (TRANS == 'N' && incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (m >= XGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            ptrdiff_t tid = 0, nth = 1;
            if (use_omp) { tid = omp_get_thread_num(); nth = omp_get_num_threads(); }
            const ptrdiff_t i_lo = blas_part_bound(m, tid, nth);
            const ptrdiff_t i_hi = blas_part_bound(m, tid + 1, nth);
            xgemv_n_stride1_slice(n, i_lo, i_hi, alpha, a, lda, x, y);
        }
#else
        xgemv_n_stride1_slice(n, 0, m, alpha, a, lda, x, y);
#endif
    } else if ((TRANS == 'T' || TRANS == 'C') && incx == 1 && incy == 1) {
        const bool conj_a = (TRANS == 'C');
#ifdef _OPENMP
        const bool use_omp = (n >= XGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            ptrdiff_t tid = 0, nth = 1;
            if (use_omp) { tid = omp_get_thread_num(); nth = omp_get_num_threads(); }
            const ptrdiff_t j_lo = blas_part_bound(n, tid, nth);
            const ptrdiff_t j_hi = blas_part_bound(n, tid + 1, nth);
            xgemv_tc_stride1_slice(m, j_lo, j_hi, conj_a, alpha, a, lda, x, y);
        }
#else
        xgemv_tc_stride1_slice(m, 0, n, conj_a, alpha, a, lda, x, y);
#endif
    } else {
        const bool conj_a = (TRANS == 'C');
#ifdef _OPENMP
        const ptrdiff_t span = (TRANS == 'N') ? m : n;
        const bool use_omp = (span >= XGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            ptrdiff_t tid = 0, nth = 1;
            if (use_omp) { tid = omp_get_thread_num(); nth = omp_get_num_threads(); }
            const ptrdiff_t lo = blas_part_bound(span, tid, nth);
            const ptrdiff_t hi = blas_part_bound(span, tid + 1, nth);
            xgemv_general_stride_slice(m, n, TRANS, conj_a, alpha, a, lda, x, incx, y, incy, lo, hi);
        }
#else
        xgemv_general_stride_slice(m, n, TRANS, conj_a, alpha, a, lda, x, incx, y, incy,
                                   0, (TRANS == 'N') ? m : n);
#endif
    }
}

EPBLAS_FACADE_GEMV(xgemv, TC)

#undef A_
