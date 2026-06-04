/*
 * qscal — kind16 port of OpenBLAS dscal.  x := alpha * x.
 *
 * Reference bails on incx <= 0 (matches blas/src/qscal.f).
 */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __float128 T;

#define MULTI_THREAD_MINIMAL 10000

static void scal_kernel(ptrdiff_t n, T alpha, T *x, ptrdiff_t incx)
{
    if (incx == 1) {
        ptrdiff_t i, n1 = n & -8;
        for (i = 0; i < n1; i += 8) {
            x[i+0] *= alpha; x[i+1] *= alpha;
            x[i+2] *= alpha; x[i+3] *= alpha;
            x[i+4] *= alpha; x[i+5] *= alpha;
            x[i+6] *= alpha; x[i+7] *= alpha;
        }
        for (; i < n; ++i) x[i] *= alpha;
        return;
    }
    for (ptrdiff_t i = 0; i < n; ++i) x[i*incx] *= alpha;
}

void qscal_(const int *N, const T *ALPHA, T *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    T alpha = *ALPHA;

    if (n <= 0 || incx <= 0 || alpha == 1.0Q) return;

#ifdef _OPENMP
    if (n > MULTI_THREAD_MINIMAL) {
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
                    scal_kernel(end - start, alpha,
                                x + start * incx, incx);
            }
            return;
        }
    }
#endif
    scal_kernel(n, alpha, x, incx);
}
