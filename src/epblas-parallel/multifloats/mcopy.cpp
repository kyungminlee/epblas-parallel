/*
 * mcopy — multifloats real DD: Y := X.
 * Memory-bandwidth bound; SIMD just widens the load/store width.
 */
#include <cstddef>
#include <cstring>
#include <multifloats.h>

namespace mf = multifloats;
using T = mf::float64x2;

extern "C" void mcopy_(const int *n_,
                       const T *x, const int *incx_,
                       T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        std::memcpy(y, x, static_cast<std::size_t>(n) * sizeof(T));
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}
