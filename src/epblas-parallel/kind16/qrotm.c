/* qrotm — kind16 real: apply modified Givens. */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __float128 TR;

static inline void step(const TR flag, const TR h11, const TR h12, const TR h21, const TR h22,
                        TR *xi, TR *yi)
{
    TR w = *xi, z = *yi;
    if (flag < 0.0Q)        { *xi = w * h11 + z * h12; *yi = w * h21 + z * h22; }
    else if (flag == 0.0Q)  { *xi = w + z * h12;       *yi = w * h21 + z; }
    else                    { *xi = w * h11 + z;       *yi = -w + h22 * z; }
}

#ifdef _OPENMP
/* Threaded modified Givens — quad is compute-bound, so it threads (see
 * qaxpy.c). Each iteration is independent; index-from-i covers every stride. */
#define QROTM_OMP_MIN 128
__attribute__((noinline)) static bool qrotm_omp(ptrdiff_t n, TR *x, ptrdiff_t incx, TR *y, ptrdiff_t incy,
                                               TR flag, TR h11, TR h12, TR h21, TR h22)
{
    if (n <= QROTM_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    ptrdiff_t iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (ptrdiff_t i = 0; i < n; ++i)
        step(flag, h11, h12, h21, h22, &x[ix0 + i * incx], &y[iy0 + i * incy]);
    return 1;
}
#endif

static void qrotm_core(ptrdiff_t n, TR *x, ptrdiff_t incx, TR *y, ptrdiff_t incy,
                       const TR *dparam)
{
    const TR flag = dparam[0];
    if (n <= 0 || flag == -2.0Q) return;
    const TR h11 = dparam[1], h21 = dparam[2], h12 = dparam[3], h22 = dparam[4];
#ifdef _OPENMP
    if (qrotm_omp(n, x, incx, y, incy, flag, h11, h12, h21, h22)) return;
#endif
    if (incx == 1 && incy == 1) {
        for (ptrdiff_t i = 0; i < n; ++i) step(flag, h11, h12, h21, h22, &x[i], &y[i]);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { step(flag, h11, h12, h21, h22, &x[ix], &y[iy]);
                                       ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_ROTM(qrotm, TR)
