/*
 * qxasum — kind16 port of OpenBLAS dzasum.
 *
 * sum( |Re(z[i])| + |Im(z[i])| ).  Real return, complex input.
 * Bails on n<=0 or incx<=0.
 */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __complex128 C;
typedef __float128 T;

#define MULTI_THREAD_MINIMAL 10000

static inline T ldabs(T x) { return x < 0 ? -x : x; }

static T asum_kernel(ptrdiff_t n, const C *x, ptrdiff_t incx)
{
    T s = 0.0Q;
    const T *p;
    if (incx == 1) {
        p = (const T *)x;
        for (ptrdiff_t i = 0; i < 2*n; i += 2) {
            s += ldabs(p[i]) + ldabs(p[i+1]);
        }
        return s;
    }
    for (ptrdiff_t i = 0; i < n; ++i) {
        p = (const T *)(x + i*incx);
        s += ldabs(p[0]) + ldabs(p[1]);
    }
    return s;
}

T qxasum_(const int *N, const C *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    if (n <= 0 || incx <= 0) return 0.0Q;

#ifdef _OPENMP
    if (n > MULTI_THREAD_MINIMAL) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            if (nthreads > 64) nthreads = 64;
            T partial[64] = {0};
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                ptrdiff_t chunk = (n + nth - 1) / nth;
                ptrdiff_t start = (ptrdiff_t)tid * chunk;
                ptrdiff_t end   = start + chunk;
                if (end > n) end = n;
                if (start < end)
                    partial[tid] = asum_kernel(end - start,
                                               x + start * incx, incx);
            }
            T s = 0.0Q;
            for (int i = 0; i < nthreads; ++i) s += partial[i];
            return s;
        }
    }
#endif
    return asum_kernel(n, x, incx);
}
