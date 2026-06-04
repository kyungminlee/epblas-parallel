/*
 * xaxpy — kind16 port of OpenBLAS zaxpy.  Y := alpha*X + Y, complex.
 */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __complex128 C;

#define MULTI_THREAD_MINIMAL 10000

static void axpy_kernel(ptrdiff_t n, C alpha, const C *x, ptrdiff_t incx,
                                              C *y,       ptrdiff_t incy)
{
    if (incx == 1 && incy == 1) {
        for (ptrdiff_t i = 0; i < n; ++i) y[i] += alpha * x[i];
        return;
    }
    for (ptrdiff_t i = 0; i < n; ++i) y[i*incy] += alpha * x[i*incx];
}

void xaxpy_(const int *N, const C *ALPHA,
            const C *x, const int *INCX,
            C       *y, const int *INCY)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    C alpha = *ALPHA;
    __float128 ar = __real__ alpha, ai = __imag__ alpha;

    if (n <= 0) return;
    if (ar == 0.0Q && ai == 0.0Q) return;
    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

#ifdef _OPENMP
    if (incx != 0 && incy != 0 && n > MULTI_THREAD_MINIMAL) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                ptrdiff_t chunk = (n + nth - 1) / nth;
                ptrdiff_t start = (ptrdiff_t)tid * chunk;
                ptrdiff_t end   = start + chunk;
                if (end > n) end = n;
                if (start < end)
                    axpy_kernel(end - start, alpha,
                                x + start * incx, incx,
                                y + start * incy, incy);
            }
            return;
        }
    }
#endif
    axpy_kernel(n, alpha, x, incx, y, incy);
}
