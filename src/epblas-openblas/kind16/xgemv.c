/*
 * xgemv — kind16 port of OpenBLAS zgemv.  Complex general MV.
 *
 *   y := alpha * op(A) * x + beta * y
 *
 * op(A) = A (TRANS='N'), A^T (TRANS='T'), A^H (TRANS='C', conjugate).
 *
 * Algorithm matches Fortran reference (BLAS): unit-stride column-AXPY for
 * 'N', column-DOT (conjugated or not) for 'T'/'C'.  -fcx-fortran-rules at
 * the CMake level removes the NaN-handling branch around complex multiplies.
 *
 * Fortran ABI:
 *   subroutine xgemv(trans, m, n, alpha, a, lda, x, incx, beta, y, incy)
 */

#include <stddef.h>
#include <quadmath.h>
#include <string.h>
#include <complex.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __complex128 C;

#define MULTI_THREAD_MINIMAL 4096
#define SPLIT_X_MN_MIN       9216   /* matches OpenBLAS 24*24*GEMM_THRESH^2 */
#define Y_DUMMY_NUM          512    /* element cap on m*nthreads (complex = 2x storage) */

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

void xgemv_(const char *TRANS, const int *M, const int *N,
            const C *ALPHA, const C *a, const int *LDA,
            const C *x, const int *INCX,
            const C *BETA, C *y, const int *INCY,
            size_t trans_len)
{
    (void)trans_len;
    ptrdiff_t m    = (ptrdiff_t)(*M);
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    C alpha = *ALPHA;
    C beta  = *BETA;

    char tr = (char)toupper((unsigned char)*TRANS);
    int trans  = (tr == 'T' || tr == 'C') ? 1 : 0;
    int conj   = (tr == 'C') ? 1 : 0;

    if (m == 0 || n == 0 || (alpha == 0.0Q && beta == 1.0Q)) return;

    ptrdiff_t lenx = trans ? m : n;
    ptrdiff_t leny = trans ? n : m;

    if (beta != 1.0Q) {
        ptrdiff_t absy = incy < 0 ? -incy : incy;
        ptrdiff_t iy = 0;
        if (beta == 0.0Q) for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] = 0.0Q; iy += absy; }
        else              for (ptrdiff_t i = 0; i < leny; ++i) { y[iy] *= beta;  iy += absy; }
    }

    if (alpha == 0.0Q) return;

    if (incx < 0) x -= (lenx - 1) * incx;
    if (incy < 0) y -= (leny - 1) * incy;

#ifdef _OPENMP
    int nthreads = 1;
    if (m * n > MULTI_THREAD_MINIMAL) nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    if (!trans) {
        if (incx == 1 && incy == 1) {
#ifdef _OPENMP
            if (nthreads > 1) {
                /* split_x fallback: m too small for width>=4 partition, but
                 * m*n large — N-split into per-thread y_dummy slots,
                 * controller reduces. Mirrors gemv_thread.c. */
                if (m < 4 * nthreads
                    && (ptrdiff_t)m * (ptrdiff_t)nthreads <= Y_DUMMY_NUM
                    && (ptrdiff_t)m * (ptrdiff_t)n > SPLIT_X_MN_MIN) {
                    C y_dummy[Y_DUMMY_NUM];
                    size_t used = (size_t)m * (size_t)nthreads;
                    memset(y_dummy, 0, sizeof(C) * used);
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int nth = omp_get_num_threads();
                        ptrdiff_t j_lo = ((ptrdiff_t)tid       * n) / nth;
                        ptrdiff_t j_hi = ((ptrdiff_t)(tid + 1) * n) / nth;
                        if (j_lo < j_hi) {
                            C *y_slot = y_dummy + (size_t)tid * (size_t)m;
                            for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
                                C temp = alpha * x[j];
                                const C *aj = &A_(0, j);
                                for (ptrdiff_t i = 0; i < m; ++i)
                                    y_slot[i] += temp * aj[i];
                            }
                        }
                    }
                    for (ptrdiff_t i = 0; i < m; ++i) {
                        C s = 0.0Q;
                        for (int t = 0; t < nthreads; ++t)
                            s += y_dummy[(size_t)t * (size_t)m + i];
                        y[i] += s;
                    }
                    return;
                }
                #pragma omp parallel num_threads(nthreads)
                {
                    int tid = omp_get_thread_num();
                    int nth = omp_get_num_threads();
                    ptrdiff_t i_lo = ((ptrdiff_t)tid       * m) / nth;
                    ptrdiff_t i_hi = ((ptrdiff_t)(tid + 1) * m) / nth;
                    for (ptrdiff_t j = 0; j < n; ++j) {
                        C temp = alpha * x[j];
                        const C *aj = &A_(0, j);
                        for (ptrdiff_t i = i_lo; i < i_hi; ++i)
                            y[i] += temp * aj[i];
                    }
                }
                return;
            }
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                C temp = alpha * x[j];
                const C *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < m; ++i) y[i] += temp * aj[i];
            }
        } else {
            ptrdiff_t jx = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                C temp = alpha * x[jx];
                ptrdiff_t iy = 0;
                const C *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < m; ++i) {
                    y[iy] += temp * aj[i];
                    iy += incy;
                }
                jx += incx;
            }
        }
    } else {
        if (incx == 1 && incy == 1) {
#ifdef _OPENMP
            if (nthreads > 1) {
                #pragma omp parallel for num_threads(nthreads) schedule(static)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    C temp = 0.0Q;
                    const C *aj = &A_(0, j);
                    if (!conj) {
                        for (ptrdiff_t i = 0; i < m; ++i) temp += aj[i] * x[i];
                    } else {
                        for (ptrdiff_t i = 0; i < m; ++i) temp += conjq(aj[i]) * x[i];
                    }
                    y[j] += alpha * temp;
                }
                return;
            }
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                C temp = 0.0Q;
                const C *aj = &A_(0, j);
                if (!conj) {
                    for (ptrdiff_t i = 0; i < m; ++i) temp += aj[i] * x[i];
                } else {
                    for (ptrdiff_t i = 0; i < m; ++i) temp += conjq(aj[i]) * x[i];
                }
                y[j] += alpha * temp;
            }
        } else {
            ptrdiff_t jy = 0;
            for (ptrdiff_t j = 0; j < n; ++j) {
                C temp = 0.0Q;
                ptrdiff_t ix = 0;
                const C *aj = &A_(0, j);
                if (!conj) {
                    for (ptrdiff_t i = 0; i < m; ++i) { temp += aj[i] * x[ix]; ix += incx; }
                } else {
                    for (ptrdiff_t i = 0; i < m; ++i) { temp += conjq(aj[i]) * x[ix]; ix += incx; }
                }
                y[jy] += alpha * temp;
                jy += incy;
            }
        }
    }
}

#undef A_
