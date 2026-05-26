/* qasum — kind16 real: returns Σ |X|. */
#include <quadmath.h>
/* fabsq via __builtin_fabsf128 — single `pand` instead of a libquadmath function call. */
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __float128 T;

T qasum_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx < 1) return 0.0Q;
    T s0 = 0.0Q, s1 = 0.0Q;
    if (incx == 1) {
        int i = 0;
        for (; i + 1 < n; i += 2) { s0 += fabsq(x[i]); s1 += fabsq(x[i + 1]); }
        if (i < n) s0 += fabsq(x[i]);
    } else {
        for (int i = 0, ix = 0; i < n; ++i, ix += incx) s0 += fabsq(x[ix]);
    }
    return s0 + s1;
}
