/*
 * mswap — multifloats real DD: swap X ↔ Y.
 */
#include <cstddef>
#include <multifloats.h>

namespace mf = multifloats;
using T = mf::float64x2;

extern "C" void mswap_(const int *n_,
                       T *x, const int *incx_,
                       T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; ++i) { T t = x[i]; x[i] = y[i]; y[i] = t; }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { T t = x[ix]; x[ix] = y[iy]; y[iy] = t; ix += incx; iy += incy; }
    }
}
