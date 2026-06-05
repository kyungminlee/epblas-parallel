/* qcopy — kind16 real: Y := X. */
#include <string.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __float128 T;

#ifdef _OPENMP
/* Threaded copy. Pure data movement, but threading still spreads the quad-wide
 * stream across cores for more memory bandwidth (ob threads it ~2x). Higher
 * crossover than the compute-bound RMW ops since small n stays in cache.
 * schedule(static) gives each thread a contiguous block (sequential stores).
 * Real-quad copy only wins past L2 (crossover ~8K; n=4096 still washes). */
#define QCOPY_OMP_MIN 8192
__attribute__((noinline)) static int qcopy_omp(int n, const T *x, int incx,
                                               T *y, int incy)
{
    if (n <= QCOPY_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    int ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    int iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; ++i) y[iy0 + i * incy] = x[ix0 + i * incx];
    return 1;
}
#endif

void qcopy_(const int *n_, const T *x, const int *incx_, T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
#ifdef _OPENMP
    if (qcopy_omp(n, x, incx, y, incy)) return;
#endif
    if (incx == 1 && incy == 1) memcpy(y, x, (size_t)n * sizeof(T));
    else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}
