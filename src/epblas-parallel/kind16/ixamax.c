/* ixamax — kind16 complex: 1-based argmax(|re|+|im|). */
#include <quadmath.h>
/* fabsq via __builtin_fabsf128 — single `pand` instead of a libquadmath function call. */
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __complex128 T;
typedef __float128 R;
int ixamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
    int best = 1;
    R bv = fabsq(__real__ x[0]) + fabsq(__imag__ x[0]);
    int ix = incx;
    for (int i = 2; i <= n; ++i) {
        R v = fabsq(__real__ x[ix]) + fabsq(__imag__ x[ix]);
        if (v > bv) { bv = v; best = i; }
        ix += incx;
    }
    return best;
}
