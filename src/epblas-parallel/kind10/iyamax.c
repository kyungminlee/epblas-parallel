/* iyamax — kind10 complex: 1-based argmax(|re|+|im|). */
#include <math.h>
typedef _Complex long double T;
typedef long double R;
int iyamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
    int best = 1;
    R bv = fabsl(__real__ x[0]) + fabsl(__imag__ x[0]);
    int ix = incx;
    for (int i = 2; i <= n; ++i) {
        R v = fabsl(__real__ x[ix]) + fabsl(__imag__ x[ix]);
        if (v > bv) { bv = v; best = i; }
        ix += incx;
    }
    return best;
}
