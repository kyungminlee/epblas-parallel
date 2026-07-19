/*
 * wdotu — kind10 port of OpenBLAS zdotu.  Returns sum(x*y), unconjugated.
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

static C dotu_kernel(ptrdiff_t n, const C *x, ptrdiff_t incx,
                                  const C *y, ptrdiff_t incy)
{
    C s = C(0.0, 0.0);
    if (incx == 1 && incy == 1) {
        for (ptrdiff_t i = 0; i < n; ++i) s += x[i] * y[i];
        return s;
    }
    for (ptrdiff_t i = 0; i < n; ++i) s += x[i*incx] * y[i*incy];
    return s;
}

extern "C" C wdotu_(const int *N, const C *x, const int *INCX,
         const C *y, const int *INCY)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    if (n <= 0) return C(0.0, 0.0);
    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

#ifdef _OPENMP
    if (incx != 0 && incy != 0 && n > MULTI_THREAD_MINIMAL) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            if (nthreads > MBLAS_L1_MAX_THREADS) nthreads = MBLAS_L1_MAX_THREADS;
            C partial[MBLAS_L1_MAX_THREADS];
            for (int i = 0; i < MBLAS_L1_MAX_THREADS; ++i) partial[i] = C(0.0, 0.0);
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                ptrdiff_t chunk = (n + nth - 1) / nth;
                ptrdiff_t start = (ptrdiff_t)tid * chunk;
                ptrdiff_t end   = start + chunk;
                if (end > n) end = n;
                if (start < end)
                    partial[tid] = dotu_kernel(end - start,
                                               x + start * incx, incx,
                                               y + start * incy, incy);
            }
            C s = C(0.0, 0.0);
            for (int i = 0; i < nthreads; ++i) s += partial[i];
            return s;
        }
    }
#endif
    return dotu_kernel(n, x, incx, y, incy);
}
