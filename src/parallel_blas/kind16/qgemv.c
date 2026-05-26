/*
 * qgemv — kind16 (__float128) general matrix-vector multiply.
 *
 * Two public entry points share the same per-slice compute kernels:
 *
 *   qgemv_         — top-level entry. Opens its own `#pragma omp parallel`
 *                    region (over the M-axis for TR='N', over the N-axis
 *                    for TR='T').
 *
 *   qgemv_serial_  — bare serial entry. No OpenMP pragma anywhere on the
 *                    call path. Safe to invoke from inside another
 *                    function's `#pragma omp parallel` region.
 *
 * qgemv_ also checks `omp_in_parallel()` and skips its own fork if
 * invoked inside an existing parallel region, falling back to serial.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QGEMV_OMP_MIN 64

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Pure serial body for TR='N', stride-1: y[i_lo:i_hi] += alpha * A[i_lo:i_hi, :] * x.
 * Each thread (or the lone serial caller) writes a disjoint slice of y. */
static void qgemv_n_stride1_slice(
    int N, int i_lo, int i_hi,
    T alpha,
    const T *restrict a, int lda,
    const T *restrict x, T *restrict y)
{
    const T zero = 0.0Q;
    for (int j = 0; j < N; ++j) {
        const T xj = x[j];
        if (xj != zero) {
            const T t = alpha * xj;
            const T *aj = &A_(0, j);
            for (int i = i_lo; i < i_hi; ++i) y[i] += t * aj[i];
        }
    }
}

/* Pure serial body for TR ∈ {'T','C'}, stride-1: y[j_lo:j_hi] += alpha * (A^T * x)[j_lo:j_hi].
 * Each thread (or the lone serial caller) writes a disjoint slice of y. */
static void qgemv_t_stride1_slice(
    int M, int j_lo, int j_hi,
    T alpha,
    const T *restrict a, int lda,
    const T *restrict x, T *restrict y)
{
    const T zero = 0.0Q;
    for (int j = j_lo; j < j_hi; ++j) {
        const T *aj = &A_(0, j);
        T s = zero;
        for (int i = 0; i < M; ++i) s += aj[i] * x[i];
        y[j] += alpha * s;
    }
}

/* General-stride serial fallback (incx≠1 or incy≠1). */
static void qgemv_general_stride(
    int M, int N, int TR,
    T alpha, const T *a, int lda,
    const T *x, int incx, T *y, int incy)
{
    const T zero = 0.0Q;
    if (TR == 'N') {
        int jx = (incx < 0) ? -(N - 1) * incx : 0;
        for (int j = 0; j < N; ++j) {
            const T xj = x[jx];
            if (xj != zero) {
                const T t = alpha * xj;
                int iy = (incy < 0) ? -(M - 1) * incy : 0;
                for (int i = 0; i < M; ++i) {
                    y[iy] += t * A_(i, j);
                    iy += incy;
                }
            }
            jx += incx;
        }
    } else {
        int jy = (incy < 0) ? -(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            T s = zero;
            int ix = (incx < 0) ? -(M - 1) * incx : 0;
            for (int i = 0; i < M; ++i) {
                s += A_(i, j) * x[ix];
                ix += incx;
            }
            y[jy] += alpha * s;
            jy += incy;
        }
    }
}

/* Apply beta scaling to y[0:leny] (with stride incy). */
static void qgemv_apply_beta(int leny, int incy, T beta, T *y)
{
    const T zero = 0.0Q, one = 1.0Q;
    if (beta == one) return;
    int iy = (incy < 0) ? -(leny - 1) * incy : 0;
    for (int i = 0; i < leny; ++i) {
        if (beta == zero) y[iy] = zero;
        else              y[iy] *= beta;
        iy += incy;
    }
}

/* Pure-serial entry. No OpenMP anywhere on this call path. */
void qgemv_serial_(
    const char *trans,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t trans_len)
{
    (void)trans_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (M == 0 || N == 0) return;

    const T zero = 0.0Q;
    const int leny = (TR == 'N') ? M : N;

    qgemv_apply_beta(leny, incy, beta, y);

    if (alpha == zero) return;

    if (TR == 'N' && incx == 1 && incy == 1) {
        qgemv_n_stride1_slice(N, 0, M, alpha, a, lda, x, y);
    } else if (TR != 'N' && incx == 1 && incy == 1) {
        qgemv_t_stride1_slice(M, 0, N, alpha, a, lda, x, y);
    } else {
        qgemv_general_stride(M, N, TR, alpha, a, lda, x, incx, y, incy);
    }
}

/* Parallel entry. Opens its own parallel region; falls back to serial if
 * invoked from inside another parallel region. */
void qgemv_(
    const char *trans,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t trans_len)
{
    (void)trans_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (M == 0 || N == 0) return;

    const T zero = 0.0Q;
    const int leny = (TR == 'N') ? M : N;

    qgemv_apply_beta(leny, incy, beta, y);

    if (alpha == zero) return;

#ifdef _OPENMP
    const int in_parallel = omp_in_parallel();
#else
    const int in_parallel = 0;
#endif

    if (TR == 'N' && incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (M >= QGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            int tid = 0, nt = 1;
            if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
            const int i_lo = ((long long)M * tid) / nt;
            const int i_hi = ((long long)M * (tid + 1)) / nt;
            qgemv_n_stride1_slice(N, i_lo, i_hi, alpha, a, lda, x, y);
        }
#else
        qgemv_n_stride1_slice(N, 0, M, alpha, a, lda, x, y);
#endif
    } else if (TR != 'N' && incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= QGEMV_OMP_MIN && blas_omp_max_threads() > 1 && !in_parallel);
        #pragma omp parallel if(use_omp)
        {
            int tid = 0, nt = 1;
            if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
            const int j_lo = ((long long)N * tid) / nt;
            const int j_hi = ((long long)N * (tid + 1)) / nt;
            qgemv_t_stride1_slice(M, j_lo, j_hi, alpha, a, lda, x, y);
        }
#else
        qgemv_t_stride1_slice(M, 0, N, alpha, a, lda, x, y);
#endif
    } else {
        qgemv_general_stride(M, N, TR, alpha, a, lda, x, incx, y, incy);
    }
}

#undef A_
