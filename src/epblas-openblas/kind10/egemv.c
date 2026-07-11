/*
 * egemv — kind10 (REAL(KIND=10) / 80-bit long double) port of OpenBLAS dgemv.
 *
 *   y := alpha * A   * x + beta * y    (TRANS = 'N')   A is M x N
 *   y := alpha * A^T * x + beta * y    (TRANS = 'T'/'C')
 *
 * Faithful structure mirror of OpenBLAS:
 *   - Outer wrapper mirrors interface/gemv.c — TOUPPER(trans), early-return
 *     on (m==0 || n==0), beta-scale of y, alpha==0 short-circuit, negative-
 *     stride pre-shift, dispatch to GEMV_N / GEMV_T kernel.
 *   - Kernels gemv_n / gemv_t mirror kernel/generic/gemv_{n,t}.c — the row-
 *     outer AXPY form for N and the column-outer dot form for T. No SIMD
 *     (x86_64 has no AVX path for 80-bit long double).
 *
 * OpenMP: GEMV is bandwidth-bound, so block-partition the OUTPUT vector y
 * across threads (rows of A for 'N', columns of A for 'T'). All A reads
 * inside a thread's slice are column-major stride-1 — no false sharing
 * because each thread owns a disjoint range of y indices.
 *
 * split_x fallback (NoTrans only, mirrors gemv_thread.c): when m is too
 * small to feed all threads at OpenBLAS's width=4 minimum, but m*n is still
 * large, switch to N-split — each thread accumulates alpha*A[:,j_lo:j_hi]*x
 * into a private y_dummy slot, then the controller reduces all slots into y.
 * Triggered when m < 4*nthreads, m*n big enough, and m*nthreads <= 1024
 * (matches Y_DUMMY_NUM cap).
 *
 * Fortran ABI:
 *   subroutine egemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
 */

#include <stddef.h>
#include <string.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#define MULTI_THREAD_MINIMAL 4096   /* m*n threshold for main path */
#define SPLIT_X_MN_MIN       9216   /* matches OpenBLAS 24*24*GEMM_THRESH^2 */
#define Y_DUMMY_NUM          1024   /* element cap on m*nthreads in split_x */

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

static void gemv_n_unit(ptrdiff_t i_lo, ptrdiff_t i_hi, ptrdiff_t n,
                        T alpha, const T *a, ptrdiff_t lda,
                        const T *x, T *y)
{
    /* y[i] += alpha * sum_j A(i,j) * x[j], for i in [i_lo, i_hi).
     * J-axis unroll by 2 — halves y memory traffic on this AXPY shape. */
    ptrdiff_t j = 0;
    for (; j + 1 < n; j += 2) {
        T t0 = alpha * x[j];
        T t1 = alpha * x[j + 1];
        const T *a0 = &A_(0, j);
        const T *a1 = &A_(0, j + 1);
        for (ptrdiff_t i = i_lo; i < i_hi; ++i)
            y[i] = (y[i] + t0 * a0[i]) + t1 * a1[i];
    }
    for (; j < n; ++j) {
        T t = alpha * x[j];
        const T *aj = &A_(0, j);
        for (ptrdiff_t i = i_lo; i < i_hi; ++i)
            y[i] += t * aj[i];
    }
}

static void gemv_n_strided(ptrdiff_t i_lo, ptrdiff_t i_hi, ptrdiff_t n,
                           T alpha, const T *a, ptrdiff_t lda,
                           const T *x, ptrdiff_t incx,
                           T *y, ptrdiff_t incy)
{
    ptrdiff_t jx = 0;
    for (ptrdiff_t j = 0; j < n; ++j) {
        T t = alpha * x[jx];
        const T *aj = &A_(0, j);
        ptrdiff_t iy = i_lo * incy;
        for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
            y[iy] += t * aj[i];
            iy += incy;
        }
        jx += incx;
    }
}

static void gemv_t_unit(ptrdiff_t j_lo, ptrdiff_t j_hi, ptrdiff_t m,
                        T alpha, const T *a, ptrdiff_t lda,
                        const T *x, T *y)
{
    /* y[j] += alpha * sum_i A(i,j) * x[i], for j in [j_lo, j_hi). */
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        const T *aj = &A_(0, j);
        T s0 = 0.0L, s1 = 0.0L;
        ptrdiff_t i = 0;
        for (; i + 1 < m; i += 2) {
            s0 += aj[i]     * x[i];
            s1 += aj[i + 1] * x[i + 1];
        }
        T s = s0 + s1;
        for (; i < m; ++i) s += aj[i] * x[i];
        y[j] += alpha * s;
    }
}

static void gemv_t_strided(ptrdiff_t j_lo, ptrdiff_t j_hi, ptrdiff_t m,
                           T alpha, const T *a, ptrdiff_t lda,
                           const T *x, ptrdiff_t incx,
                           T *y, ptrdiff_t incy)
{
    ptrdiff_t jy = j_lo * incy;
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        T s = 0.0L;
        ptrdiff_t ix = 0;
        const T *aj = &A_(0, j);
        for (ptrdiff_t i = 0; i < m; ++i) {
            s += aj[i] * x[ix];
            ix += incx;
        }
        y[jy] += alpha * s;
        jy += incy;
    }
}

/* split_x partial: y[0..m] += alpha * A[:, j_lo:j_hi] * x[j_lo:j_hi]
 * Unit-stride x, unit-stride local y slot. Same j-axis-unroll-2 shape as
 * gemv_n_unit. */
static void gemv_n_split_unit(ptrdiff_t m, ptrdiff_t j_lo, ptrdiff_t j_hi,
                              T alpha, const T *a, ptrdiff_t lda,
                              const T *x, T *y)
{
    ptrdiff_t j = j_lo;
    for (; j + 1 < j_hi; j += 2) {
        T t0 = alpha * x[j];
        T t1 = alpha * x[j + 1];
        const T *a0 = &A_(0, j);
        const T *a1 = &A_(0, j + 1);
        for (ptrdiff_t i = 0; i < m; ++i)
            y[i] = (y[i] + t0 * a0[i]) + t1 * a1[i];
    }
    for (; j < j_hi; ++j) {
        T t = alpha * x[j];
        const T *aj = &A_(0, j);
        for (ptrdiff_t i = 0; i < m; ++i)
            y[i] += t * aj[i];
    }
}

static void gemv_n_split_strided_x(ptrdiff_t m, ptrdiff_t j_lo, ptrdiff_t j_hi,
                                   T alpha, const T *a, ptrdiff_t lda,
                                   const T *x, ptrdiff_t incx, T *y)
{
    for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
        T t = alpha * x[j * incx];
        const T *aj = &A_(0, j);
        for (ptrdiff_t i = 0; i < m; ++i)
            y[i] += t * aj[i];
    }
}

void egemv_(const char *TRANS, const int *M, const int *N,
            const T *ALPHA, const T *a, const int *LDA,
            const T *x, const int *INCX,
            const T *BETA, T *y, const int *INCY)
{
    ptrdiff_t m    = (ptrdiff_t)(*M);
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    T alpha = *ALPHA;
    T beta  = *BETA;

    if (m == 0 || n == 0) return;

    char tr = (char)toupper((unsigned char)*TRANS);
    int trans = (tr == 'T' || tr == 'C') ? 1 : 0;

    ptrdiff_t lenx = trans ? m : n;
    ptrdiff_t leny = trans ? n : m;

    /* Scale y by beta first — independent of the A*x update. */
    if (beta != 1.0L) {
        ptrdiff_t absy = incy < 0 ? -incy : incy;
        ptrdiff_t iy = 0;
        if (beta == 0.0L) {
            for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] = 0.0L; iy += absy; }
        } else {
            for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] *= beta;  iy += absy; }
        }
    }

    if (alpha == 0.0L) return;

    /* OpenBLAS negative-stride pre-shift — both kernel forms then index
     * from y[0] / x[0] walking by positive stride. */
    if (incx < 0) x -= (lenx - 1) * incx;
    if (incy < 0) y -= (leny - 1) * incy;

    int unit = (incx == 1 && incy == 1);

#ifdef _OPENMP
    int nthreads = 1;
    if (m * n > MULTI_THREAD_MINIMAL) nthreads = omp_get_max_threads();
    if (nthreads > 1) {
        if (!trans) {
            /* split_x fallback: m too small for width>=4 across all threads,
             * but m*n still large enough to warrant parallel — N-split into
             * per-thread y_dummy slots, controller reduces. */
            if (m < 4 * nthreads
                && (ptrdiff_t)m * (ptrdiff_t)nthreads <= Y_DUMMY_NUM
                && (ptrdiff_t)m * (ptrdiff_t)n > SPLIT_X_MN_MIN) {
                T y_dummy[Y_DUMMY_NUM];
                size_t used = (size_t)m * (size_t)nthreads;
                memset(y_dummy, 0, sizeof(T) * used);
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int nth = omp_get_num_threads();
                    ptrdiff_t j_lo = ((ptrdiff_t)tid       * n) / nth;
                    ptrdiff_t j_hi = ((ptrdiff_t)(tid + 1) * n) / nth;
                    if (j_lo < j_hi) {
                        T *y_slot = y_dummy + (size_t)tid * (size_t)m;
                        if (incx == 1)
                            gemv_n_split_unit(m, j_lo, j_hi, alpha, a, lda,
                                              x, y_slot);
                        else
                            gemv_n_split_strided_x(m, j_lo, j_hi, alpha,
                                                   a, lda, x, incx, y_slot);
                    }
                }
                ptrdiff_t iy = 0;
                for (ptrdiff_t i = 0; i < m; ++i) {
                    T s = 0.0L;
                    for (int t = 0; t < nthreads; ++t)
                        s += y_dummy[(size_t)t * (size_t)m + i];
                    y[iy] += s;
                    iy += incy;
                }
                return;
            }
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                ptrdiff_t i_lo = ((ptrdiff_t)tid       * m) / nth;
                ptrdiff_t i_hi = ((ptrdiff_t)(tid + 1) * m) / nth;
                if (i_lo < i_hi) {
                    if (unit) gemv_n_unit(i_lo, i_hi, n, alpha, a, lda, x, y);
                    else      gemv_n_strided(i_lo, i_hi, n, alpha, a, lda,
                                              x, incx, y, incy);
                }
            }
        } else {
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                ptrdiff_t j_lo = ((ptrdiff_t)tid       * n) / nth;
                ptrdiff_t j_hi = ((ptrdiff_t)(tid + 1) * n) / nth;
                if (j_lo < j_hi) {
                    if (unit) gemv_t_unit(j_lo, j_hi, m, alpha, a, lda, x, y);
                    else      gemv_t_strided(j_lo, j_hi, m, alpha, a, lda,
                                              x, incx, y, incy);
                }
            }
        }
        return;
    }
#endif

    if (!trans) {
        if (unit) gemv_n_unit(0, m, n, alpha, a, lda, x, y);
        else      gemv_n_strided(0, m, n, alpha, a, lda, x, incx, y, incy);
    } else {
        if (unit) gemv_t_unit(0, n, m, alpha, a, lda, x, y);
        else      gemv_t_strided(0, n, m, alpha, a, lda, x, incx, y, incy);
    }
}

#undef A_
