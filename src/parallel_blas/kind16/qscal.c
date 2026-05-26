/* qscal — kind16 real: X := α · X. */
#include <quadmath.h>
typedef __float128 T;

void qscal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const T alpha = *alpha_;
    if (n <= 0 || alpha == 1.0Q) return;
    if (incx == 1) {
        /* 5-way unroll matches NETLIB DSCAL. */
        const int m = n % 5;
        for (int i = 0; i < m; ++i) x[i] *= alpha;
        for (int i = m; i < n; i += 5) {
            x[i    ] *= alpha;
            x[i + 1] *= alpha;
            x[i + 2] *= alpha;
            x[i + 3] *= alpha;
            x[i + 4] *= alpha;
        }
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) { x[ix] *= alpha; ix += incx; }
    }
}
