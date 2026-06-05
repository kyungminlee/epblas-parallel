/* qswap — kind16 real: swap X ↔ Y. */
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __float128 T;

#ifdef _OPENMP
/* Threaded swap — two quad streams read+write; threading spreads them across
 * cores for memory bandwidth (see qcopy.c). Each iteration is independent.
 * Real-quad swap only wins past L2 (crossover ~8K; n=4096 still washes). */
#define QSWAP_OMP_MIN 8192
__attribute__((noinline)) static int qswap_omp(int n, T *x, int incx,
                                               T *y, int incy)
{
    if (n <= QSWAP_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    int ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    int iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; ++i) {
        int ix = ix0 + i * incx, iy = iy0 + i * incy;
        T t = x[ix]; x[ix] = y[iy]; y[iy] = t;
    }
    return 1;
}
#endif

void qswap_(const int *n_, T *x, const int *incx_, T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
#ifdef _OPENMP
    if (qswap_omp(n, x, incx, y, incy)) return;
#endif
    if (incx == 1 && incy == 1) {
        const int m = n % 3;
        for (int i = 0; i < m; ++i) { T t = x[i]; x[i] = y[i]; y[i] = t; }
        for (int i = m; i < n; i += 3) {
            T t0 = x[i    ]; x[i    ] = y[i    ]; y[i    ] = t0;
            T t1 = x[i + 1]; x[i + 1] = y[i + 1]; y[i + 1] = t1;
            T t2 = x[i + 2]; x[i + 2] = y[i + 2]; y[i + 2] = t2;
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { T t = x[ix]; x[ix] = y[iy]; y[iy] = t; ix += incx; iy += incy; }
    }
}
