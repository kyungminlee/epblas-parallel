/*
 * yscal — kind10 port of OpenBLAS zscal.  x := alpha*x, alpha complex.
 *
 * NETLIB bail: incx <= 0 is a no-op (matches blas/src/yscal.f and
 * OpenBLAS interface/scal.c).
 *
 * Kernel shape: manual __real__/__imag__ expansion with the imag-part
 * products written as `xi*ar + xr*ai` (X.im term first). gcc emits x87
 * products in source order; this order matches the value already on
 * top of the x87 stack after the first store, saving one `fxch` per
 * iter vs the natural `xr*ai + xi*ar` form (15 insns / 2 fxch → 14
 * insns / 1 fxch). Borrowed from epblas-parallel/kind10/yscal.c —
 * matches gfortran's ZSCAL codegen instruction-for-instruction.
 */
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef _Complex long double C;
typedef long double T;

#include "eblas_tuning.h"

static void scal_kernel(ptrdiff_t n, T ar, T ai, T *base, ptrdiff_t incx)
{
    if (incx == 1) {
        T *p = base;
        T *e = p + 2 * n;
        for (; p < e; p += 2) {
            const T xr = p[0], xi = p[1];
            p[0] = xr * ar - xi * ai;
            p[1] = xi * ar + xr * ai;
        }
        return;
    }
    /* incx > 0 only (NETLIB bails on incx <= 0). */
    for (ptrdiff_t i = 0; i < n; ++i) {
        T *p = base + 2 * i * incx;
        const T xr = p[0], xi = p[1];
        p[0] = xr * ar - xi * ai;
        p[1] = xi * ar + xr * ai;
    }
}

void yscal_(const int *N, const C *ALPHA, C *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    if (n <= 0 || incx <= 0) return;
    const T ar = __real__ *ALPHA;
    const T ai = __imag__ *ALPHA;
    if (ar == 1.0L && ai == 0.0L) return;

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
                    scal_kernel(end - start, ar, ai,
                                base + 2 * start * incx, incx);
            }
            return;
        }
    }
#endif
    scal_kernel(n, ar, ai, base, incx);
}
