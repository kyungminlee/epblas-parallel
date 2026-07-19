/*
 * easum — kind10 port of OpenBLAS dasum.  sum(|x|).
 *
 * Reference bails on incx <= 0.
 */
#include <stddef.h>
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#include "eblas_tuning.h"

static T asum_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx)
{
    /* 6-accumulator unroll mirrors NETLIB DASUM and saturates x87 fadd
     * latency in flight.  fabsl() compiles to a single `fabs` x87
     * instruction — cheaper than a ternary which can lower to
     * cmov+neg or a conditional branch. */
    if (incx == 1) {
        T s0 = 0.0L, s1 = 0.0L, s2 = 0.0L, s3 = 0.0L, s4 = 0.0L, s5 = 0.0L;
        ptrdiff_t i = 0;
        for (; i + 5 < n; i += 6) {
            s0 += fabsl(x[i+0]);
            s1 += fabsl(x[i+1]);
            s2 += fabsl(x[i+2]);
            s3 += fabsl(x[i+3]);
            s4 += fabsl(x[i+4]);
            s5 += fabsl(x[i+5]);
        }
        T s = ((s0 + s1) + (s2 + s3)) + (s4 + s5);
        for (; i < n; ++i) s += fabsl(x[i]);
        return s;
    }
    T s = 0.0L;
    for (ptrdiff_t i = 0; i < n; ++i) s += fabsl(x[i*incx]);
    return s;
}

T easum_(const int *N, const T *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    if (n <= 0 || incx <= 0) return 0.0L;

#ifdef _OPENMP
    if (n > MULTI_THREAD_MINIMAL) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            T partial[L1_PARTIAL_MAX_THREADS] = {0};
            if (nthreads > L1_PARTIAL_MAX_THREADS) nthreads = L1_PARTIAL_MAX_THREADS;
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
            T s = 0.0L;
            for (int i = 0; i < nthreads; ++i) s += partial[i];
            return s;
        }
    }
#endif
    return asum_kernel(n, x, incx);
}
