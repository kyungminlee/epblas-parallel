/*
 * masum — multifloats DD port of OpenBLAS dasum.  sum(|x|).
 *
 * Faithful retype of kind10/easum.c: long double -> multifloats::float64x2,
 * fabsl -> mf::fabs. Same 6-accumulator unroll, threading, and threshold;
 * only the element type and its abs differ. (Port policy: the openblas
 * faithful-handport ADR kept with the workspace notes, outside this repo.)
 *
 * Reference bails on incx <= 0.
 */
#include <cstddef>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

#include "mblas_tuning.h"
#define MULTI_THREAD_MINIMAL MBLAS_MT_MIN_L1

static T asum_kernel(std::ptrdiff_t n, const T *x, std::ptrdiff_t incx)
{
    /* 6-accumulator unroll mirrors NETLIB DASUM, splitting the dependency
     * chain across independent DD accumulators. */
    if (incx == 1) {
        T s0 = 0.0, s1 = 0.0, s2 = 0.0, s3 = 0.0, s4 = 0.0, s5 = 0.0;
        std::ptrdiff_t i = 0;
        for (; i + 5 < n; i += 6) {
            s0 += mf::fabs(x[i+0]);
            s1 += mf::fabs(x[i+1]);
            s2 += mf::fabs(x[i+2]);
            s3 += mf::fabs(x[i+3]);
            s4 += mf::fabs(x[i+4]);
            s5 += mf::fabs(x[i+5]);
        }
        T s = ((s0 + s1) + (s2 + s3)) + (s4 + s5);
        for (; i < n; ++i) s += mf::fabs(x[i]);
        return s;
    }
    T s = 0.0;
    for (std::ptrdiff_t i = 0; i < n; ++i) s += mf::fabs(x[i*incx]);
    return s;
}

extern "C" T masum_(const int *N, const T *x, const int *INCX)
{
    std::ptrdiff_t n    = (std::ptrdiff_t)(*N);
    std::ptrdiff_t incx = (std::ptrdiff_t)(*INCX);
    if (n <= 0 || incx <= 0) return T(0.0);

#ifdef _OPENMP
    if (n > MULTI_THREAD_MINIMAL) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            T partial[MBLAS_L1_MAX_THREADS] = {};
            if (nthreads > MBLAS_L1_MAX_THREADS) nthreads = MBLAS_L1_MAX_THREADS;
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                std::ptrdiff_t chunk = (n + nth - 1) / nth;
                std::ptrdiff_t start = (std::ptrdiff_t)tid * chunk;
                std::ptrdiff_t end   = start + chunk;
                if (end > n) end = n;
                if (start < end)
                    partial[tid] = asum_kernel(end - start,
                                               x + start * incx, incx);
            }
            T s = 0.0;
            for (int i = 0; i < nthreads; ++i) s += partial[i];
            return s;
        }
    }
#endif
    return asum_kernel(n, x, incx);
}
