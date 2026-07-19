/*
 * wcopy — kind10 port of OpenBLAS zcopy.  Y := X, complex.
 */
#include <stddef.h>
#include <complex>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#endif

using C = std::complex<multifloats::float64x2>;
typedef std::complex<multifloats::float64x2> C;

#include "mblas_tuning.h"
#define MULTI_THREAD_MINIMAL MBLAS_MT_MIN_L1

static void copy_kernel(ptrdiff_t n, const C *x, ptrdiff_t incx,
                                     C *y,       ptrdiff_t incy)
{
    if (incx == 1 && incy == 1) {
        for (ptrdiff_t i = 0; i < n; ++i) y[i] = x[i];
        return;
    }
    for (ptrdiff_t i = 0; i < n; ++i) y[i*incy] = x[i*incx];
}

extern "C" void wcopy_(const int *N, const C *x, const int *INCX,
            C *y, const int *INCY)
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
