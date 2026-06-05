/* qaxpy — kind16 real: Y := α·X + Y. */
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __float128 T;

#ifdef _OPENMP
/* Threaded elementwise AXPY. Unlike fp80 (memory-bound, kept serial), quad is
 * compute-bound under libquadmath, so even the RMW L1 ops thread profitably
 * (crossover ~n=128). Indices are computed from i so every stride combination
 * is handled by the one loop; the serial fast-paths below stay intact for the
 * sub-threshold / single-thread case. */
#define QAXPY_OMP_MIN 128
__attribute__((noinline)) static int qaxpy_omp(int n, T alpha,
                                               const T *x, int incx,
                                               T *y, int incy)
{
    if (n <= QAXPY_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    int ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    int iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; ++i) y[iy0 + i * incy] += alpha * x[ix0 + i * incx];
    return 1;
}
#endif

void qaxpy_(const int *n_, const T *alpha_,
            const T *x, const int *incx_,
            T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T alpha = *alpha_;
    if (n <= 0 || alpha == 0.0Q) return;
#ifdef _OPENMP
    if (qaxpy_omp(n, alpha, x, incx, y, incy)) return;
#endif
    if (incx == 1 && incy == 1) {
        const int m = n % 4;
        for (int i = 0; i < m; ++i) y[i] += alpha * x[i];
        for (int i = m; i < n; i += 4) {
            y[i    ] += alpha * x[i    ];
            y[i + 1] += alpha * x[i + 1];
            y[i + 2] += alpha * x[i + 2];
            y[i + 3] += alpha * x[i + 3];
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] += alpha * x[ix]; ix += incx; iy += incy; }
    }
}
