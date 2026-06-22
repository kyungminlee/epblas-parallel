#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
/* erotm — kind10 real: apply modified Givens.
 *
 * Imag-part products written with the just-loaded operand (z) first so
 * gcc's x87 backend emits the multiply against top-of-stack and saves
 * a fxch — same pattern as yscal (Addendum 17). The flag-unswitched
 * paths each shrink from 14 insns + 2 fxch to 12 insns + 1 fxch in
 * the flag<0 branch. */
#include "../common/epblas_facade.h"
typedef long double T;

static inline void step(const T flag, const T h11, const T h12, const T h21, const T h22,
                        T *xi, T *yi)
{
    T w = *xi, z = *yi;
    if (flag < 0.0L)        { *xi = w * h11 + z * h12; *yi = z * h22 + w * h21; }
    else if (flag == 0.0L)  { *xi = w + z * h12;       *yi = z + w * h21; }
    else                    { *xi = w * h11 + z;       *yi = z * h22 - w; }
}

/* Unit-stride kernel, shared by the serial entry and the per-thread OMP slices. */
static void erotm_unit(ptrdiff_t n, T flag, T h11, T h12, T h21, T h22, T *x, T *y)
{
    for (ptrdiff_t i = 0; i < n; ++i) step(flag, h11, h12, h21, h22, &x[i], &y[i]);
}

#ifdef _OPENMP
/* Threaded modified Givens — same cache-bandwidth rationale as eaxpy_omp (see
 * eaxpy.c). Compute-bound (mul/add per element). Threshold set by par4<=ob4 (ob
 * keeps rotm serial at small N). Measured under iomp5: par4/ob4 1.09@1024, then
 * 0.95@1536 — break-even ~1536, stay serial through 1024. */
#define EROTM_OMP_MIN 1024
static bool erotm_omp(ptrdiff_t n, T flag, T h11, T h12, T h21, T h22, T *x, T *y)
{
    if (n <= EROTM_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) erotm_unit(hi - lo, flag, h11, h12, h21, h22, x + lo, y + lo);
    }
    return 1;
}
#endif

static void erotm_core(ptrdiff_t n, T *x, ptrdiff_t incx, T *y, ptrdiff_t incy,
                       const T *dparam)
{
    const T flag = dparam[0];
    if (n <= 0 || flag == -2.0L) return;
    const T h11 = dparam[1], h21 = dparam[2], h12 = dparam[3], h22 = dparam[4];
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (erotm_omp(n, flag, h11, h12, h21, h22, x, y)) return;
#endif
        erotm_unit(n, flag, h11, h12, h21, h22, x, y);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { step(flag, h11, h12, h21, h22, &x[ix], &y[iy]);
                                       ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_ROTM(erotm, T)
