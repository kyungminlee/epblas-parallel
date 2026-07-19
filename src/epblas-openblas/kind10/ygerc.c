/*
 * ygerc — kind10 port of OpenBLAS zgerc.  Complex conjugated outer product.
 *
 *   A := alpha * x * y^H + A    (conjugate y)
 *
 * Fortran ABI:  subroutine ygerc(m, n, alpha, x, incx, y, incy, a, lda)
 */

#include <stddef.h>
#include <stdlib.h>
#include <complex.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef _Complex long double C;

/* Family-tuned threading gate (gemv/ger) — deviates from the shared
 * default in eblas_tuning.h; kept local (family tuning). */
#define MULTI_THREAD_MINIMAL 4096

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

void ygerc_(const int *M, const int *N, const C *ALPHA,
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

    if (m == 0 || n == 0 || alpha == 0.0L) return;

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
            C t = alpha * conjl(y[j * incy]);
            C *aj = &A_(0, j);
            ptrdiff_t ix = 0;
            for (ptrdiff_t i = 0; i < m; ++i) { aj[i] += t * x[ix]; ix += incx; }
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) schedule(static) if(nthreads > 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            C t = alpha * conjl(y[j * incy]);
            C *aj = &A_(0, j);
            for (ptrdiff_t i = 0; i < m; ++i) aj[i] += t * xp[i];
        }
    }

    if (x_buf) free(x_buf);
}

#undef A_
