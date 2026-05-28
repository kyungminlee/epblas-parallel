/*
 * eswap — kind10 port of OpenBLAS dswap.  x ↔ y.
 */
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#define MULTI_THREAD_MINIMAL 10000

static void swap_kernel(ptrdiff_t n, T *x, ptrdiff_t incx,
                                     T *y, ptrdiff_t incy)
{
    if (incx == 1 && incy == 1) {
        ptrdiff_t i, n1 = n & -4;
        T t0, t1, t2, t3;
        for (i = 0; i < n1; i += 4) {
            t0 = x[i+0]; t1 = x[i+1]; t2 = x[i+2]; t3 = x[i+3];
            x[i+0] = y[i+0]; x[i+1] = y[i+1];
            x[i+2] = y[i+2]; x[i+3] = y[i+3];
            y[i+0] = t0; y[i+1] = t1; y[i+2] = t2; y[i+3] = t3;
        }
        for (; i < n; ++i) {
            T t = x[i]; x[i] = y[i]; y[i] = t;
        }
        return;
    }
    for (ptrdiff_t i = 0; i < n; ++i) {
        T t = x[i*incx]; x[i*incx] = y[i*incy]; y[i*incy] = t;
    }
}

void eswap_(const int *N, T *x, const int *INCX,
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
                    swap_kernel(end - start,
                                x + start * incx, incx,
                                y + start * incy, incy);
            }
            return;
        }
    }
#endif
    swap_kernel(n, x, incx, y, incy);
}
