/* xcopy — kind16 complex: Y := X. */
#include <string.h>
#include <quadmath.h>
typedef __complex128 T;

void xcopy_(const int *n_, const T *x, const int *incx_, T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) memcpy(y, x, (size_t)n * sizeof(T));
    else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}
