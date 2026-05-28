/* xqscal — kind16: X := α·X with α real __float128, X complex. */
#include <quadmath.h>
typedef __complex128 T;
typedef __float128 R;

void xqscal_(const int *n_, const R *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const R alpha = *alpha_;
    if (n <= 0 || alpha == 1.0Q) return;
    if (incx == 1) {
        T *end = x + n;
        for (T *p = x; p < end; ++p) {
            __real__ *p *= alpha;
            __imag__ *p *= alpha;
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) {
            __real__ x[ix] *= alpha;
            __imag__ x[ix] *= alpha;
            ix += incx;
        }
    }
}
