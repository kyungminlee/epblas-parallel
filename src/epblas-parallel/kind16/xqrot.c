/* xqrot — kind16: complex Givens with real c, s (CSROT/ZDROT analog). */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __complex128 T;
typedef __float128 R;

#ifdef _OPENMP
/* Threaded complex Givens (real c, s) — quad is compute-bound, so it threads
 * (see qaxpy.c). Each iteration is independent; index-from-i covers all strides. */
#define XQROT_OMP_MIN 128
__attribute__((noinline)) static bool xqrot_omp(ptrdiff_t n, T *x, ptrdiff_t incx,
                                               T *y, ptrdiff_t incy, R c, R s)
{
    if (n <= XQROT_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
    ptrdiff_t iy0 = (incy < 0) ? (-n + 1) * incy : 0;
    #pragma omp parallel for schedule(static) num_threads(nthreads)
    for (ptrdiff_t i = 0; i < n; ++i) {
        ptrdiff_t ix = ix0 + i * incx, iy = iy0 + i * incy;
        T tx;
        __real__ tx = c * __real__ x[ix] + s * __real__ y[iy];
        __imag__ tx = c * __imag__ x[ix] + s * __imag__ y[iy];
        __real__ y[iy] = c * __real__ y[iy] - s * __real__ x[ix];
        __imag__ y[iy] = c * __imag__ y[iy] - s * __imag__ x[ix];
        x[ix] = tx;
    }
    return 1;
}
#endif

static void xqrot_core(ptrdiff_t n, T *x, ptrdiff_t incx, T *y, ptrdiff_t incy,
                       const R *c_, const R *s_)
{
    const R c = *c_, s = *s_;
    if (n <= 0) return;
#ifdef _OPENMP
    if (xqrot_omp(n, x, incx, y, incy, c, s)) return;
#endif
    if (incx == 1 && incy == 1) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            T tx;
            __real__ tx = c * __real__ x[i] + s * __real__ y[i];
            __imag__ tx = c * __imag__ x[i] + s * __imag__ y[i];
            __real__ y[i] = c * __real__ y[i] - s * __real__ x[i];
            __imag__ y[i] = c * __imag__ y[i] - s * __imag__ x[i];
            x[i] = tx;
        }
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            T tx;
            __real__ tx = c * __real__ x[ix] + s * __real__ y[iy];
            __imag__ tx = c * __imag__ x[ix] + s * __imag__ y[iy];
            __real__ y[iy] = c * __real__ y[iy] - s * __real__ x[ix];
            __imag__ y[iy] = c * __imag__ y[iy] - s * __imag__ x[ix];
            x[ix] = tx;
            ix += incx; iy += incy;
        }
    }
}

EPBLAS_FACADE_ROT(xqrot, R, T)
