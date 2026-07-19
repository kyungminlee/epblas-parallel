/*
 * yerot — kind10 port of OpenBLAS zdrot.
 *
 * Apply real-coefficient plane rotation to complex vectors:
 *   x' :=  c*x + s*y
 *   y' := -s*x + c*y
 * where c, s are REAL(KIND=10) and x, y are COMPLEX(KIND=10).
 *
 * Fortran ABI: yerot(N, ZX, INCX, ZY, INCY, C, S).
 */
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef _Complex long double C;
typedef long double T;

#include "eblas_tuning.h"

static void rot_kernel(ptrdiff_t n, C *x, ptrdiff_t incx,
                                    C *y, ptrdiff_t incy,
                       T c, T s)
{
    /* Apply componentwise: real scalars on complex vectors. */
    if (incx == 1 && incy == 1) {
        T *px = (T *)x, *py = (T *)y;
        ptrdiff_t two_n = 2 * n;
        for (ptrdiff_t i = 0; i < two_n; ++i) {
            T xi = px[i], yi = py[i];
            px[i] = c*xi + s*yi;
            py[i] = c*yi - s*xi;
        }
        return;
    }
    for (ptrdiff_t i = 0; i < n; ++i) {
        T *px = (T *)(x + i*incx);
        T *py = (T *)(y + i*incy);
        T xr = px[0], xi = px[1], yr = py[0], yi = py[1];
        px[0] = c*xr + s*yr; px[1] = c*xi + s*yi;
        py[0] = c*yr - s*xr; py[1] = c*yi - s*xi;
    }
}

void yerot_(const int *N, C *x, const int *INCX, C *y, const int *INCY,
            const T *Creal, const T *Sreal)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    T c = *Creal, s = *Sreal;
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
