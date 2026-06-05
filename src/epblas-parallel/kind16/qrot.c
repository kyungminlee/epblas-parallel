/* qrot — kind16 real Givens rotation. */
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __float128 T;

#ifdef _OPENMP
/* Threaded Givens rotation — quad is compute-bound, so it threads (see
 * qaxpy.c). Each iteration is independent; index-from-i covers every stride. */
#define QROT_OMP_MIN 128
__attribute__((noinline)) static int qrot_omp(int n, T *x, int incx,
                                              T *y, int incy, T c, T s)
{
    if (n <= QROT_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    int ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    int iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; ++i) {
        int ix = ix0 + i * incx, iy = iy0 + i * incy;
        T tx = c * x[ix] + s * y[iy];
        y[iy] = c * y[iy] - s * x[ix];
        x[ix] = tx;
    }
    return 1;
}
#endif

void qrot_(const int *n_, T *x, const int *incx_, T *y, const int *incy_,
           const T *c_, const T *s_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T c = *c_, s = *s_;
    if (n <= 0) return;
#ifdef _OPENMP
    if (qrot_omp(n, x, incx, y, incy, c, s)) return;
#endif
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; ++i) {
            T tx = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = tx;
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) {
            T tx = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = tx;
            ix += incx; iy += incy;
        }
    }
}
