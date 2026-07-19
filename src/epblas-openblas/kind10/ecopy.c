/*
 * ecopy — kind10 port of OpenBLAS dcopy.  y := x.
 *
 * Fortran ABI: subroutine ecopy(n, x, incx, y, incy)
 */
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#include "eblas_tuning.h"

static void copy_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx,
                                     T       *y, ptrdiff_t incy)
{
    if (incx == 1 && incy == 1) {
        ptrdiff_t i, n1 = n & -8;
        for (i = 0; i < n1; i += 8) {
            y[i+0] = x[i+0]; y[i+1] = x[i+1];
            y[i+2] = x[i+2]; y[i+3] = x[i+3];
            y[i+4] = x[i+4]; y[i+5] = x[i+5];
            y[i+6] = x[i+6]; y[i+7] = x[i+7];
        }
        for (; i < n; ++i) y[i] = x[i];
        return;
    }
    for (ptrdiff_t i = 0; i < n; ++i)
        y[i*incy] = x[i*incx];
}

void ecopy_(const int *N, const T *x, const int *INCX,
            T *y, const int *INCY)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);

    if (n <= 0) return;
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
                    copy_kernel(end - start,
                                x + start * incx, incx,
                                y + start * incy, incy);
            }
            return;
        }
    }
#endif
    copy_kernel(n, x, incx, y, incy);
}
