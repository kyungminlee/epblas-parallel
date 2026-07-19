/*
 * xgeru — kind16 port of OpenBLAS zgeru.  Complex unconjugated outer product.
 *
 *   A := alpha * x * y^T + A     (no conjugate on y)
 *
 * Fortran ABI:  subroutine xgeru(m, n, alpha, x, incx, y, incy, a, lda)
 */

#include <stddef.h>
#include <quadmath.h>
#include <stdlib.h>
#include <complex.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __complex128 C;

#include "qblas_tuning.h"
#define MULTI_THREAD_MINIMAL QBLAS_MT_MIN_L2_MN

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

void xgeru_(const int *M, const int *N, const C *ALPHA,
            const C *x, const int *INCX,
            const C *y, const int *INCY,
            C *a, const int *LDA)
{
    ptrdiff_t m    = (ptrdiff_t)(*M);
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    C alpha = *ALPHA;

    if (m == 0 || n == 0 || alpha == 0.0Q) return;

    if (incx < 0) x -= (m - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

#ifdef _OPENMP
    int nthreads = 1;
    if (m * n > MULTI_THREAD_MINIMAL) nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    const C *xp = x;
    C *x_buf = NULL;
    if (incx != 1 && nthreads > 1) {
        x_buf = (C *)malloc((size_t)m * sizeof(C));
        if (x_buf) {
            ptrdiff_t ix = 0;
            for (ptrdiff_t i = 0; i < m; ++i) { x_buf[i] = x[ix]; ix += incx; }
            xp = x_buf;
        }
    }

    if (xp == x && incx != 1) {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) schedule(static) if(nthreads > 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            C t = alpha * y[j * incy];
            C *aj = &A_(0, j);
            ptrdiff_t ix = 0;
            for (ptrdiff_t i = 0; i < m; ++i) { aj[i] += t * x[ix]; ix += incx; }
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) schedule(static) if(nthreads > 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            C t = alpha * y[j * incy];
            C *aj = &A_(0, j);
            for (ptrdiff_t i = 0; i < m; ++i) aj[i] += t * xp[i];
        }
    }

    if (x_buf) free(x_buf);
}

#undef A_
