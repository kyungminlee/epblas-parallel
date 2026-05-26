/* qrotm — kind16 real: apply modified Givens. */
#include <quadmath.h>
typedef __float128 T;

static inline void step(const T flag, const T h11, const T h12, const T h21, const T h22,
                        T *xi, T *yi)
{
    T w = *xi, z = *yi;
    if (flag < 0.0Q)        { *xi = w * h11 + z * h12; *yi = w * h21 + z * h22; }
    else if (flag == 0.0Q)  { *xi = w + z * h12;       *yi = w * h21 + z; }
    else                    { *xi = w * h11 + z;       *yi = -w + h22 * z; }
}

void qrotm_(const int *n_, T *x, const int *incx_, T *y, const int *incy_,
            const T *dparam)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    const T flag = dparam[0];
    if (n <= 0 || flag == -2.0Q) return;
    const T h11 = dparam[1], h21 = dparam[2], h12 = dparam[3], h22 = dparam[4];
    if (incx == 1 && incy == 1) {
        for (int i = 0; i < n; ++i) step(flag, h11, h12, h21, h22, &x[i], &y[i]);
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { step(flag, h11, h12, h21, h22, &x[ix], &y[iy]);
                                       ix += incx; iy += incy; }
    }
}
