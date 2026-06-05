/* qrotm — kind16 real: apply modified Givens. */
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __float128 T;

static inline void step(const T flag, const T h11, const T h12, const T h21, const T h22,
                        T *xi, T *yi)
{
    T w = *xi, z = *yi;
    if (flag < 0.0Q)        { *xi = w * h11 + z * h12; *yi = w * h21 + z * h22; }
    else if (flag == 0.0Q)  { *xi = w + z * h12;       *yi = w * h21 + z; }
    else                    { *xi = w * h11 + z;       *yi = -w + h22 * z; }
}

#ifdef _OPENMP
/* Threaded modified Givens — quad is compute-bound, so it threads (see
 * qaxpy.c). Each iteration is independent; index-from-i covers every stride. */
#define QROTM_OMP_MIN 128
__attribute__((noinline)) static int qrotm_omp(int n, T *x, int incx, T *y, int incy,
                                               T flag, T h11, T h12, T h21, T h22)
{
    if (n <= QROTM_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    int ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    int iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; ++i)
        step(flag, h11, h12, h21, h22, &x[ix0 + i * incx], &y[iy0 + i * incy]);
    return 1;
}
#endif

void qrotm_(const int *n_, T *x, const int *incx_, T *y, const int *incy_,
            const T *dparam)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T flag = dparam[0];
    if (n <= 0 || flag == -2.0Q) return;
    const T h11 = dparam[1], h21 = dparam[2], h12 = dparam[3], h22 = dparam[4];
#ifdef _OPENMP
    if (qrotm_omp(n, x, incx, y, incy, flag, h11, h12, h21, h22)) return;
#endif
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; ++i) step(flag, h11, h12, h21, h22, &x[i], &y[i]);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { step(flag, h11, h12, h21, h22, &x[ix], &y[iy]);
                                       ix += incx; iy += incy; }
    }
}
