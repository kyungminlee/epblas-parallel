/* qxasum — kind16: Σ (|re(X)| + |im(X)|) for complex X. */
#include <quadmath.h>
/* fabsq via __builtin_fabsf128 — single `pand` instead of a libquadmath function call. */
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __complex128 T;
typedef __float128 R;

R qxasum_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx < 1) return 0.0Q;
    R s = 0.0Q;
    if (incx == 1) {
        for (int i = 0; i < n; ++i) s += fabsq(__real__ x[i]) + fabsq(__imag__ x[i]);
    } else {
        for (int i = 0, ix = 0; i < n; ++i, ix += incx)
            s += fabsq(__real__ x[ix]) + fabsq(__imag__ x[ix]);
    }
    return s;
}
