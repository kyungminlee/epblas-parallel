/* xqrot — kind16: complex Givens with real c, s (CSROT/ZDROT analog). */
#include <quadmath.h>
typedef __complex128 T;
typedef __float128 R;

void xqrot_(const int *n_, T *x, const int *incx_, T *y, const int *incy_,
            const R *c_, const R *s_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const R c = *c_, s = *s_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; ++i) {
            T tx;
            __real__ tx = c * __real__ x[i] + s * __real__ y[i];
            __imag__ tx = c * __imag__ x[i] + s * __imag__ y[i];
            __real__ y[i] = c * __real__ y[i] - s * __real__ x[i];
            __imag__ y[i] = c * __imag__ y[i] - s * __imag__ x[i];
            x[i] = tx;
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) {
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
