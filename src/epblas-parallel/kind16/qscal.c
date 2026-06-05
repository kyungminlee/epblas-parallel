/* qscal — kind16 real: X := α · X. */
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __float128 T;

#ifdef _OPENMP
/* Threaded elementwise SCAL — quad is compute-bound, so it threads (see
 * qaxpy.c). Index-from-i covers every stride; serial fast-paths preserved. */
#define QSCAL_OMP_MIN 128
__attribute__((noinline)) static int qscal_omp(int n, T alpha, T *x, int incx)
{
    if (n <= QSCAL_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    int ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; ++i) x[ix0 + i * incx] *= alpha;
    return 1;
}
#endif

void qscal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const T alpha = *alpha_;
    if (n <= 0 || alpha == 1.0Q) return;
#ifdef _OPENMP
    if (qscal_omp(n, alpha, x, incx)) return;
#endif
    if (incx == 1) {
        /* 5-way unroll matches NETLIB DSCAL. */
        const int m = n % 5;
        for (int i = 0; i < m; ++i) x[i] *= alpha;
        for (int i = m; i < n; i += 5) {
            x[i    ] *= alpha;
            x[i + 1] *= alpha;
            x[i + 2] *= alpha;
            x[i + 3] *= alpha;
            x[i + 4] *= alpha;
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) { x[ix] *= alpha; ix += incx; }
    }
}
