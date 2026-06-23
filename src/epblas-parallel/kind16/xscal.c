/* xscal — kind16 complex: X := α·X (α complex). */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __complex128 TC;

#ifdef _OPENMP
/* Threaded elementwise SCAL — quad is compute-bound, so it threads (see
 * qaxpy.c). Index-from-i covers every stride; serial fast-paths preserved. */
#define XSCAL_OMP_MIN 128
__attribute__((noinline)) static bool xscal_omp(ptrdiff_t n, TC alpha, TC *x, ptrdiff_t incx)
{
    if (n <= XSCAL_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (ptrdiff_t i = 0; i < n; ++i) x[ix0 + i * incx] *= alpha;
    return 1;
}
#endif

static void xscal_core(ptrdiff_t n, const TC *alpha_, TC *x, ptrdiff_t incx)
{
    const TC alpha = *alpha_;
    if (n <= 0) return;
#ifdef _OPENMP
    if (xscal_omp(n, alpha, x, incx)) return;
#endif
    if (incx == 1) {
        TC *end = x + n;
        for (TC *p = x; p < end; ++p) *p *= alpha;
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { x[ix] *= alpha; ix += incx; }
    }
}

EPBLAS_FACADE_SCAL(xscal, TC, TC)
