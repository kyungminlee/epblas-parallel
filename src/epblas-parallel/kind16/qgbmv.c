/*
 * qgbmv — kind16 (__float128) general band matrix-vector multiply.
 *   y := alpha*A*x + beta*y  or  y := alpha*A^T*x + beta*y
 * Band storage: A(i,j) at AB[(ku + i - j) + j*lda], for max(1,j-ku) <= i <= min(M,j+kl).
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <stdlib.h>
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QGBMV_OMP_MIN 64
#define QGBMV_MAX_CPUS 256

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

#ifdef _OPENMP
/* Threaded NoTrans band matvec (incx==incy==1). Every output row i is an
 * independent band dot, so threads own disjoint y[lo,hi). Bit-identical to the
 * serial column-scatter: ax[j]=alpha*x[j] matches the reference's temp, and
 * seeding the accumulator with y[i] then adding ax[j]*A(i,j) in ascending-j
 * order reproduces the scatter's exact left-to-right association. Returns 1 if
 * handled, 0 to fall back. A(i,j)=base[j*(lda-1)] with base=a+(KU+i). */
static int qgbmv_n_omp(int M, int N, int KL, int KU, T alpha,
                       const T *restrict a, int lda,
                       const T *restrict x, int incx, T *restrict y, int incy);
static int qgbmv_t_omp(int M, int N, int KL, int KU, T alpha,
                       const T *restrict a, int lda,
                       const T *restrict x, int incx, T *restrict y, int incy);
#endif

void qgbmv_(
    const char *trans,
    const int *m_, const int *n_,
    const int *kl_, const int *ku_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict x, const int *incx_,
    const T *beta_,
    T *restrict y, const int *incy_,
    size_t trans_len)
{
    (void)trans_len;
    const int M = *m_, N = *n_;
    const int KL = *kl_, KU = *ku_;
    const int lda = *lda_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_, beta = *beta_;
    const T zero = 0.0Q, one = 1.0Q;
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (M == 0 || N == 0 || (alpha == zero && beta == one)) return;

    const int leny = (TR == 'N') ? M : N;
    const int lenx = (TR == 'N') ? N : M;

    /* y := beta*y */
    if (beta != one) {
        int iy = (incy < 0) ? -(leny - 1) * incy : 0;
        if (beta == zero) {
            for (int i = 0; i < leny; ++i) { y[iy] = zero; iy += incy; }
        } else {
            for (int i = 0; i < leny; ++i) { y[iy] = beta * y[iy]; iy += incy; }
        }
    }
    if (alpha == zero) return;

    if (TR == 'N') {
#ifdef _OPENMP
        /* NoTrans threads for contiguous AND strided x/y (the helper gathers
         * strided x and writes strided y); bit-identical to the serial scatter
         * via ascending-j association. */
        if (M >= QGBMV_OMP_MIN && blas_omp_max_threads() > 1
            && qgbmv_n_omp(M, N, KL, KU, alpha, a, lda, x, incx, y, incy))
            return;
#endif
        if (incx == 1 && incy == 1) {
            /* sequential along j; each j writes y[max(0,j-KU)..min(M,j+KL+1)) so cols overlap */
            for (int j = 0; j < N; ++j) {
                const T tmp = alpha * x[j];
                const int i_lo = (j - KU > 0) ? (j - KU) : 0;
                const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
                const int k = KU - j;
                for (int i = i_lo; i < i_hi; ++i) y[i] += tmp * A_(k + i, j);
            }
        } else {
            /* General stride — transcribe Netlib carefully. */
            int kx = (incx < 0) ? -(lenx - 1) * incx : 0;
            int ky = (incy < 0) ? -(leny - 1) * incy : 0;
            int jx = kx;
            for (int j = 0; j < N; ++j) {
                const T tmp = alpha * x[jx];
                int iy = ky;
                const int i_lo = (j - KU > 0) ? (j - KU) : 0;
                const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
                const int k = KU - j;
                for (int i = i_lo; i < i_hi; ++i) {
                    y[iy] += tmp * A_(k + i, j);
                    iy += incy;
                }
                jx += incx;
                if (j >= KU) ky += incy;
            }
        }
    } else if (incx == 1 && incy == 1) {
        /* T-path: each j computes independent y[j] */
#ifdef _OPENMP
        const int use_omp = (N >= QGBMV_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T s = zero;
            const int i_lo = (j - KU > 0) ? (j - KU) : 0;
            const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const int k = KU - j;
            for (int i = i_lo; i < i_hi; ++i) s += A_(k + i, j) * x[i];
            y[j] += alpha * s;
        }
    } else {
        /* Strided Trans gather (C already mapped to T above). */
#ifdef _OPENMP
        if (N >= QGBMV_OMP_MIN && blas_omp_max_threads() > 1
            && qgbmv_t_omp(M, N, KL, KU, alpha, a, lda, x, incx, y, incy))
            return;
#endif
        int kx = (incx < 0) ? -(lenx - 1) * incx : 0;
        int ky = (incy < 0) ? -(leny - 1) * incy : 0;
        int jy = ky;
        for (int j = 0; j < N; ++j) {
            T s = zero;
            int ix = kx;
            const int i_lo = (j - KU > 0) ? (j - KU) : 0;
            const int i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            const int k = KU - j;
            for (int i = i_lo; i < i_hi; ++i) {
                s += A_(k + i, j) * x[ix];
                ix += incx;
            }
            y[jy] += alpha * s;
            jy += incy;
            if (j >= KU) kx += incx;
        }
    }
}

#ifdef _OPENMP
/* Threaded NoTrans band matvec. Output rows partition across threads (each y[i] an
 * independent band dot, x and y distinct -> no race). ax[j]=alpha*x[j] is
 * precomputed once (matching the serial scatter's temp); seeding the accumulator
 * with y[i] then adding ax[j]*A(i,j) in ascending-j order reproduces the scatter's
 * exact left-to-right association -> bit-identical for both contiguous and strided
 * x/y. NoTrans reads N of x, writes M of y. */
__attribute__((noinline)) static int qgbmv_n_omp(
    int M, int N, int KL, int KU, T alpha,
    const T *restrict a, int lda,
    const T *restrict x, int incx, T *restrict y, int incy)
{
    if (M < QGBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > QGBMV_MAX_CPUS) nthreads = QGBMV_MAX_CPUS;

    /* ax[j] = alpha*x[j] precomputed once (matches the scatter's temp exactly). */
    T *ax = (T *)malloc((size_t)N * sizeof(T));
    if (!ax) return 0;
    const int ix0 = (incx < 0) ? -(N - 1) * incx : 0;
    for (int j = 0; j < N; ++j) ax[j] = alpha * x[ix0 + j * incx];
    const ptrdiff_t iy0 = (incy < 0) ? -(ptrdiff_t)(M - 1) * incy : 0;

    const ptrdiff_t s1 = (ptrdiff_t)lda - 1;
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        ptrdiff_t lo = ((ptrdiff_t)M * tid) / nthreads;
        ptrdiff_t hi = ((ptrdiff_t)M * (tid + 1)) / nthreads;
        for (ptrdiff_t i = lo; i < hi; ++i) {
            ptrdiff_t j_lo = (i - KL > 0) ? (i - KL) : 0;
            ptrdiff_t j_hi = (i + KU + 1 < N) ? (i + KU + 1) : N;
            const T *base = a + (KU + i);
            T *yi = &y[iy0 + i * incy];
            T s = *yi;                        /* seed: post-beta y[i] */
            for (ptrdiff_t j = j_lo; j < j_hi; ++j) s += ax[j] * base[j * s1];
            *yi = s;
        }
    }
    free(ax);
    return 1;
}

/* Threaded Trans band matvec (strided x and/or y; C is mapped to T by the caller).
 * Output columns partition across threads (each y[j]=alpha*Σ_i A(i,j)*x[i] disjoint,
 * x and y distinct -> no race). Strided x is gathered to contiguous so the inner dot
 * reads x[i] directly. Trans reads M of x, writes N of y. Bit-identical to the serial
 * strided gather (same ascending-i association). */
__attribute__((noinline)) static int qgbmv_t_omp(
    int M, int N, int KL, int KU, T alpha,
    const T *restrict a, int lda,
    const T *restrict x, int incx, T *restrict y, int incy)
{
    if (N < QGBMV_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > QGBMV_MAX_CPUS) nthreads = QGBMV_MAX_CPUS;

    if (incx < 0) x -= (ptrdiff_t)(M - 1) * incx;
    if (incy < 0) y -= (ptrdiff_t)(N - 1) * incy;

    const T *xptr = x;
    T *xbuf = NULL;
    if (incx != 1) {
        xbuf = (T *)malloc((size_t)M * sizeof(T));
        if (!xbuf) return 0;
        for (int i = 0; i < M; ++i) xbuf[i] = x[(ptrdiff_t)i * incx];
        xptr = xbuf;
    }

    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        ptrdiff_t lo = ((ptrdiff_t)N * tid) / nthreads;
        ptrdiff_t hi = ((ptrdiff_t)N * (tid + 1)) / nthreads;
        for (ptrdiff_t j = lo; j < hi; ++j) {
            ptrdiff_t i_lo = (j - KU > 0) ? (j - KU) : 0;
            ptrdiff_t i_hi = (j + KL + 1 < M) ? (j + KL + 1) : M;
            ptrdiff_t k = KU - j;
            const T *col = &A_(k + i_lo, j);
            T s = 0.0Q;
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) s += *col++ * xptr[i];
            y[j * incy] += alpha * s;
        }
    }
    free(xbuf);
    return 1;
}
#endif /* _OPENMP */

#undef A_
