/*
 * mdot — kind10 port of OpenBLAS ddot.  Returns x^T y.
 *
 * gfortran ABI: REAL(KIND=10) function returns multifloats::float64x2 by value.
 */
#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#define MULTI_THREAD_MINIMAL 10000

static T dot_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx,
                                 const T *y, ptrdiff_t incy)
{
    /* Match reference DDOT's 5-way unroll into a SINGLE accumulator —
     * different summation order would drift past the fuzz tolerance
     * for small-N seeds where catastrophic cancellation dominates. */
    T s = 0.0;
    if (incx == 1 && incy == 1) {
        ptrdiff_t i = 0, m = n % 5;
        for (; i < m; ++i) s += x[i] * y[i];
        for (; i < n; i += 5)
            s += x[i+0]*y[i+0] + x[i+1]*y[i+1] + x[i+2]*y[i+2]
               + x[i+3]*y[i+3] + x[i+4]*y[i+4];
        return s;
    }
    ptrdiff_t ix = 0, iy = 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        s += x[ix] * y[iy];
        ix += incx; iy += incy;
    }
    return s;
}

extern "C" T mdot_(const int *N, const T *x, const int *INCX,
        const T *y, const int *INCY)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);

    if (n <= 0) return 0.0;
    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

#ifdef _OPENMP
    if (incx != 0 && incy != 0 && n > MULTI_THREAD_MINIMAL) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            T partial[64] = {0};
            if (nthreads > 64) nthreads = 64;
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                ptrdiff_t chunk = (n + nth - 1) / nth;
                ptrdiff_t start = (ptrdiff_t)tid * chunk;
                ptrdiff_t end   = start + chunk;
                if (end > n) end = n;
                if (start < end)
                    partial[tid] = dot_kernel(end - start,
                                              x + start * incx, incx,
                                              y + start * incy, incy);
            }
            T s = 0.0;
            for (int i = 0; i < nthreads; ++i) s += partial[i];
            return s;
        }
    }
#endif
    return dot_kernel(n, x, incx, y, incy);
}
