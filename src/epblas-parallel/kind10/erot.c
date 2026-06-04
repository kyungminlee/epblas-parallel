#include <stddef.h>
/* erot — kind10 real Givens rotation. */
typedef long double T;

void erot_(const int *n_, T *x, const int *incx_, T *y, const int *incy_,
           const T *c_, const T *s_)
{
    const ptrdiff_t n = *n_, incx = *incx_, incy = *incy_;
    const T c = *c_, s = *s_;
    if (n <= 0) return;
    /* Load both elements into locals before storing: x and y are distinct
     * `T*` the compiler can't prove non-aliasing for, so reading them inline
     * across the interleaved writes forces an 80-bit reload each iteration.
     * Hoisting to locals (the epblas-openblas shape) keeps them on the x87
     * register stack. Same ops in the same order — bit-identical. */
    if (incx == 1 && incy == 1) {
        for (ptrdiff_t i = 0; i < n; ++i) {
            T xi = x[i], yi = y[i];
            x[i] = c * xi + s * yi;
            y[i] = c * yi - s * xi;
        }
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            T xi = x[ix], yi = y[iy];
            x[ix] = c * xi + s * yi;
            y[iy] = c * yi - s * xi;
            ix += incx; iy += incy;
        }
    }
}
