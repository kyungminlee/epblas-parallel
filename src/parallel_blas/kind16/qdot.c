/* qdot — kind16 real: returns Σ X·Y. */
#include <quadmath.h>
typedef __float128 T;

T qdot_(const int *n_, const T *x, const int *incx_,
        const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return 0.0Q;
    T s0 = 0.0Q, s1 = 0.0Q;
    if (incx == 1 && incy == 1) {
        T s2 = 0.0Q, s3 = 0.0Q, s4 = 0.0Q;
        int i = 0;
        for (; i + 4 < n; i += 5) {
            s0 += x[i    ] * y[i    ];
            s1 += x[i + 1] * y[i + 1];
            s2 += x[i + 2] * y[i + 2];
            s3 += x[i + 3] * y[i + 3];
            s4 += x[i + 4] * y[i + 4];
        }
        s0 += s2 + s3 + s4;
        for (; i < n; ++i) s0 += x[i] * y[i];
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { s0 += x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s0 + s1;
}
