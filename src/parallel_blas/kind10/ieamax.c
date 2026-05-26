/* ieamax — kind10 real: 1-based argmax(|X|). */
#include <math.h>
typedef long double T;
int ieamax_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
    int best = 1; T bv = fabsl(x[0]); int ix = incx;
    for (int i = 2; i <= n; ++i) { T v = fabsl(x[ix]); if (v > bv) { bv = v; best = i; } ix += incx; }
    return best;
}
