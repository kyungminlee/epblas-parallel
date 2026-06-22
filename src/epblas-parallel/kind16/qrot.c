/* qrot — kind16 real Givens rotation. */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __float128 T;

#ifdef _OPENMP
/* Threaded Givens rotation — quad is compute-bound, so it threads (see
 * qaxpy.c). Each iteration is independent; index-from-i covers every stride. */
#define QROT_OMP_MIN 128
__attribute__((noinline)) static bool qrot_omp(ptrdiff_t n, T *x, ptrdiff_t incx,
                                              T *y, ptrdiff_t incy, T c, T s)
{
    if (n <= QROT_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    ptrdiff_t iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (ptrdiff_t i = 0; i < n; ++i) {
        ptrdiff_t ix = ix0 + i * incx, iy = iy0 + i * incy;
        T tx = c * x[ix] + s * y[iy];
        y[iy] = c * y[iy] - s * x[ix];
        x[ix] = tx;
    }
    return 1;
}
#endif

static void qrot_core(ptrdiff_t n, T *x, ptrdiff_t incx, T *y, ptrdiff_t incy,
                      const T *c_, const T *s_)
{
    const T c = *c_, s = *s_;
    if (n <= 0) return;
#ifdef _OPENMP
    if (qrot_omp(n, x, incx, y, incy, c, s)) return;
#endif
    if (incx == 1 && incy == 1) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            T tx = c * x[i] + s * y[i];
            y[i] = c * y[i] - s * x[i];
            x[i] = tx;
        }
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            T tx = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = tx;
            ix += incx; iy += incy;
        }
    }
}

EPBLAS_FACADE_ROT(qrot, T, T)
