/* xscal — kind16 complex: X := α·X (α complex). */
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __complex128 T;

#ifdef _OPENMP
/* Threaded elementwise SCAL — quad is compute-bound, so it threads (see
 * qaxpy.c). Index-from-i covers every stride; serial fast-paths preserved. */
#define XSCAL_OMP_MIN 128
__attribute__((noinline)) static int xscal_omp(int n, T alpha, T *x, int incx)
{
    if (n <= XSCAL_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    int ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (int i = 0; i < n; ++i) x[ix0 + i * incx] *= alpha;
    return 1;
}
#endif

void xscal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const T alpha = *alpha_;
    if (n <= 0) return;
#ifdef _OPENMP
    if (xscal_omp(n, alpha, x, incx)) return;
#endif
    if (incx == 1) {
        T *end = x + n;
        for (T *p = x; p < end; ++p) *p *= alpha;
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) { x[ix] *= alpha; ix += incx; }
    }
}
