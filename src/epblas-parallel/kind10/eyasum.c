/* eyasum — kind10: Σ (|re(X)| + |im(X)|) for complex X. */
#include <math.h>
typedef _Complex long double T;
typedef long double R;

R eyasum_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx < 1) return 0.0L;
    R s = 0.0L, s1 = 0.0L, s2 = 0.0L;
    if (incx == 1) {
        int i = 0;
        for (; i + 2 < n; i += 3) {
            s  += fabsl(__real__ x[i    ]) + fabsl(__imag__ x[i    ]);
            s1 += fabsl(__real__ x[i + 1]) + fabsl(__imag__ x[i + 1]);
            s2 += fabsl(__real__ x[i + 2]) + fabsl(__imag__ x[i + 2]);
        }
        s += s1 + s2;
        for (; i < n; ++i) s += fabsl(__real__ x[i]) + fabsl(__imag__ x[i]);
    } else {
        for (int i = 0, ix = 0; i < n; ++i, ix += incx)
            s += fabsl(__real__ x[ix]) + fabsl(__imag__ x[ix]);
    }
    return s;
}
