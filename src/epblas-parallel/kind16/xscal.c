/* xscal — kind16 complex: X := α·X (α complex). */
#include <quadmath.h>
typedef __complex128 T;

void xscal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    const T alpha = *alpha_;
    if (n <= 0) return;
    if (incx == 1) {
        T *end = x + n;
        for (T *p = x; p < end; ++p) *p *= alpha;
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (int i = 0; i < n; ++i) { x[ix] *= alpha; ix += incx; }
    }
}
