/* xaxpy — kind16 complex: Y := α·X + Y. */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __complex128 TC;

#ifdef _OPENMP
/* Threaded elementwise AXPY — quad is compute-bound, so it threads (see
 * qaxpy.c). Index-from-i covers every stride; serial fast-paths preserved. */
#define XAXPY_OMP_MIN 128
__attribute__((noinline)) static bool xaxpy_omp(ptrdiff_t n, TC alpha,
                                               const TC *x, ptrdiff_t incx,
                                               TC *y, ptrdiff_t incy)
{
    if (n <= XAXPY_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    ptrdiff_t iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (ptrdiff_t i = 0; i < n; ++i) y[iy0 + i * incy] += alpha * x[ix0 + i * incx];
    return 1;
}
#endif

static void xaxpy_core(ptrdiff_t n, const TC *alpha_,
                       const TC *x, ptrdiff_t incx,
                       TC *y, ptrdiff_t incy)
{
    const TC alpha = *alpha_;
    /* BLAS contract (reference ZAXPY: DCABS1(alpha)==0 => RETURN): a zero
     * scalar is a no-op, and must not turn a non-finite x into NaN in y.
     * Matches the qaxpy/eaxpy/yaxpy twins. */
    if (n <= 0 || alpha == (TC)0.0Q) return;
#ifdef _OPENMP
    if (xaxpy_omp(n, alpha, x, incx, y, incy)) return;
#endif
    if (incx == 1 && incy == 1) {
        const TC *xe = x + n;
        TC *yp = y;
        for (const TC *xp = x; xp < xe; ++xp, ++yp) *yp += alpha * (*xp);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { y[iy] += alpha * x[ix]; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_AXPY(xaxpy, TC, TC)
