/* iqamax — kind16 real: 1-based argmax(|X|). */
#include <quadmath.h>
/* fabsq via __builtin_fabsf128 — single `pand` instead of a libquadmath function call. */
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __float128 T;
int iqamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
    int best = 1; T bv = fabsq(x[0]); int ix = incx;
    for (int i = 2; i <= n; ++i) { T v = fabsq(x[ix]); if (v > bv) { bv = v; best = i; } ix += incx; }
    return best;
}
