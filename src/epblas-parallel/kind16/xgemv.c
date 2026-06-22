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
 * its own `#pragma omp parallel` region over the M-axis (TR='N') or N-axis
 * (TR='T'/'C'); it falls back to serial when invoked inside an existing
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

typedef __complex128 T;


static inline T cconj(T z) { return conjq(z); }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Pure serial body for TR='N', stride-1: y[i_lo:i_hi] += alpha * A[i_lo:i_hi, :] * x.
 * Each thread (or the lone serial caller) writes a disjoint slice of y. */
static void xgemv_n_stride1_slice(
    ptrdiff_t N, ptrdiff_t i_lo, ptrdiff_t i_hi,
    T alpha,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, T *restrict y)
{
    const T zero = 0.0Q + 0.0Qi;
    for (ptrdiff_t j = 0; j < N; ++j) {
        const T xj = x[j];
        if (xj != zero) {
            const T t = alpha * xj;
            const T *aj = &A_(0, j);
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) y[i] += t * aj[i];
        }
    }
}

/* Pure serial body for TR ∈ {'T','C'}, stride-1: y[j_lo:j_hi] += alpha * (op(A) * x)[j_lo:j_hi].
 * Each thread (or the lone serial caller) writes a disjoint slice of y. */
static void xgemv_tc_stride1_slice(
    ptrdiff_t M, ptrdiff_t j_lo, ptrdiff_t j_hi, bool conj_a,
    T alpha,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, T *restrict y)
{
    const T zero = 0.0Q + 0.0Qi;
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        const T *aj = &A_(0, j);
        T s = zero;
        if (conj_a) {
            for (ptrdiff_t i = 0; i < M; ++i) s += cconj(aj[i]) * x[i];
        } else {
            for (ptrdiff_t i = 0; i < M; ++i) s += aj[i] * x[i];
        }
        y[j] += alpha * s;
    }
}

/* General-stride slice (incx≠1 or incy≠1). For TR='N' [lo,hi) is the disjoint
 * output-row slice; for TR≠'N' it is the disjoint output-index (j) slice. Each
 * output element is written by one thread in the same per-element order as the
 * full serial loop → race-free and bit-exact (iy0/jy0/ix recomputed). */
static void xgemv_general_stride_slice(
    ptrdiff_t M, ptrdiff_t N, char TR, bool conj_a,
    T alpha, const T *a, ptrdiff_t lda,
    const T *x, ptrdiff_t incx, T *y, ptrdiff_t incy, ptrdiff_t lo, ptrdiff_t hi)
{
    const T zero = 0.0Q + 0.0Qi;
    if (TR == 'N') {
        const ptrdiff_t iy0 = (incy < 0) ? -(M - 1) * incy : 0;
        ptrdiff_t jx = (incx < 0) ? -(N - 1) * incx : 0;
        for (ptrdiff_t j = 0; j < N; ++j) {
            const T xj = x[jx];
            if (xj != zero) {
                const T t = alpha * xj;
                ptrdiff_t iy = iy0 + lo * incy;
                for (ptrdiff_t i = lo; i < hi; ++i) {
                    y[iy] += t * A_(i, j);
                    iy += incy;
                }
            }
            jx += incx;
        }
    } else {
        const ptrdiff_t jy0 = (incy < 0) ? -(N - 1) * incy : 0;
        for (ptrdiff_t j = lo; j < hi; ++j) {
            T s = zero;
            ptrdiff_t ix = (incx < 0) ? -(M - 1) * incx : 0;
            for (ptrdiff_t i = 0; i < M; ++i) {
                s += (conj_a ? cconj(A_(i, j)) : A_(i, j)) * x[ix];
                ix += incx;
            }
            y[jy0 + j * incy] += alpha * s;
        }
    }
}

/* Apply beta scaling to y[0:leny] (with stride incy). */
static void xgemv_apply_beta(ptrdiff_t leny, ptrdiff_t incy, T beta, T *y)
{
    const T zero = 0.0Q + 0.0Qi, one = 1.0Q + 0.0Qi;
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
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const char TR = blas_up(trans);

    if (M == 0 || N == 0) return;

    const T zero = 0.0Q + 0.0Qi;
    const ptrdiff_t leny = (TR == 'N') ? M : N;

    xgemv_apply_beta(leny, incy, beta, y);

    if (alpha == zero) return;

#ifdef _OPENMP
    const bool in_parallel = omp_in_parallel();
#else
    const bool in_parallel = 0;
#endif

    if (TR == 'N' && incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (M >= XGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            ptrdiff_t tid = 0, nt = 1;
            if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
            const ptrdiff_t i_lo = blas_part_bound(M, tid, nt);
            const ptrdiff_t i_hi = blas_part_bound(M, tid + 1, nt);
            xgemv_n_stride1_slice(N, i_lo, i_hi, alpha, a, lda, x, y);
        }
#else
        xgemv_n_stride1_slice(N, 0, M, alpha, a, lda, x, y);
#endif
    } else if ((TR == 'T' || TR == 'C') && incx == 1 && incy == 1) {
        const bool conj_a = (TR == 'C');
#ifdef _OPENMP
        const bool use_omp = (N >= XGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            ptrdiff_t tid = 0, nt = 1;
            if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
            const ptrdiff_t j_lo = blas_part_bound(N, tid, nt);
            const ptrdiff_t j_hi = blas_part_bound(N, tid + 1, nt);
            xgemv_tc_stride1_slice(M, j_lo, j_hi, conj_a, alpha, a, lda, x, y);
        }
#else
        xgemv_tc_stride1_slice(M, 0, N, conj_a, alpha, a, lda, x, y);
#endif
    } else {
        const bool conj_a = (TR == 'C');
#ifdef _OPENMP
        const ptrdiff_t span = (TR == 'N') ? M : N;
        const bool use_omp = (span >= XGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            ptrdiff_t tid = 0, nt = 1;
            if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
            const ptrdiff_t lo = blas_part_bound(span, tid, nt);
            const ptrdiff_t hi = blas_part_bound(span, tid + 1, nt);
            xgemv_general_stride_slice(M, N, TR, conj_a, alpha, a, lda, x, incx, y, incy, lo, hi);
        }
#else
        xgemv_general_stride_slice(M, N, TR, conj_a, alpha, a, lda, x, incx, y, incy,
                                   0, (TR == 'N') ? M : N);
#endif
    }
}

EPBLAS_FACADE_GEMV(xgemv, T)

#undef A_
