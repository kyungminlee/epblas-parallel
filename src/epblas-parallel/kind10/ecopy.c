/* ecopy — kind10 real: Y := X. */
#include <string.h>
#include <stddef.h>
typedef long double T;

void ecopy_(const int *n_, const T *x, const int *incx_, T *y, const int *incy_)
{
    const ptrdiff_t n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        memcpy(y, x, (size_t)n * sizeof(T));
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}
