/* xdotc — kind16 complex: returns Σ conj(X)·Y. */
#include <quadmath.h>
typedef __complex128 T;

T xdotc_(const int *n_, const T *x, const int *incx_,
         const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    T s = 0;
    if (n <= 0) return s;
    if (incx == 1 && incy == 1) {
        T s0 = (T)0.0Q, s1 = (T)0.0Q;
        int i = 0;
        for (; i + 1 < n; i += 2) {
            s0 += ~x[i    ] * y[i    ];
            s1 += ~x[i + 1] * y[i + 1];
        }
        s += s0 + s1;
        for (; i < n; ++i) s += ~x[i] * y[i];
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { s += ~x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s;
}
