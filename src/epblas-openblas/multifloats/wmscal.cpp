/*
 * wmscal — kind10 port of OpenBLAS zdscal.  x := alpha*x, alpha REAL.
 *
 * NETLIB bail: incx <= 0 is a no-op (matches blas/src/wmscal.f and
 * OpenBLAS interface/scal.c).
 *
 * Kernel shape: treat each complex element as two consecutive long
 * doubles and walk by 2 in a pointer-incrementing loop with the two
 * fmuls written explicitly.  Borrowed from epblas-parallel/kind10/wmscal.c
 * — gives the compiler a tight 2-fmul-per-iter body instead of a
 * 2N-iteration single-fmul loop that may or may not get unrolled.
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
typedef multifloats::float64x2 T;

#include "mblas_tuning.h"
#define MULTI_THREAD_MINIMAL MBLAS_MT_MIN_L1

static void scal_kernel(ptrdiff_t n, T alpha, T *base, ptrdiff_t incx)
{
    if (incx == 1) {
        T *p = base;
        T *e = p + 2 * n;
        for (; p < e; p += 2) {
            p[0] *= alpha;
            p[1] *= alpha;
        }
        return;
    }
    /* incx > 0 only (NETLIB bails on incx <= 0). */
    for (ptrdiff_t i = 0; i < n; ++i) {
        T *p = base + 2 * i * incx;
        p[0] *= alpha;
        p[1] *= alpha;
    }
}

extern "C" void wmscal_(const int *N, const T *ALPHA, C *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    if (n <= 0 || incx <= 0) return;
    T alpha = *ALPHA;
    if (alpha == 1.0) return;

    T *base = (T *)x;

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
                                base + 2 * start * incx, incx);
            }
            return;
        }
    }
#endif
    scal_kernel(n, alpha, base, incx);
}
