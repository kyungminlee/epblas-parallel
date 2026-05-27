/*
 * erot — kind10 port of OpenBLAS drot.  Apply plane rotation.
 *   x' :=  c*x + s*y
 *   y' := -s*x + c*y
 */
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#define MULTI_THREAD_MINIMAL 10000

static void rot_kernel(ptrdiff_t n, T *x, ptrdiff_t incx,
                                    T *y, ptrdiff_t incy,
                       T c, T s)
{
    if (incx == 1 && incy == 1) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            T xi = x[i], yi = y[i];
            x[i] = c*xi + s*yi;
            y[i] = c*yi - s*xi;
        }
        return;
    }
    for (ptrdiff_t i = 0; i < n; ++i) {
        T xi = x[i*incx], yi = y[i*incy];
        x[i*incx] = c*xi + s*yi;
        y[i*incy] = c*yi - s*xi;
    }
}

void erot_(const int *N, T *x, const int *INCX, T *y, const int *INCY,
           const T *C, const T *S)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    T c = *C, s = *S;

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
                    rot_kernel(end - start,
                               x + start * incx, incx,
                               y + start * incy, incy, c, s);
            }
            return;
        }
    }
#endif
    rot_kernel(n, x, incx, y, incy, c, s);
}
