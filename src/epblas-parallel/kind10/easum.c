/* easum — kind10 real: returns Σ |X|. */
#include <math.h>
typedef long double T;

T easum_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n < 1 || incx < 1) return 0.0L;
    T s0 = 0.0L, s1 = 0.0L;
    if (incx == 1) {
        /* 6-accumulator unroll matches NETLIB DASUM. */
        T s2 = 0.0L, s3 = 0.0L, s4 = 0.0L, s5 = 0.0L;
        int i = 0;
        for (; i + 5 < n; i += 6) {
            s0 += fabsl(x[i    ]);
            s1 += fabsl(x[i + 1]);
            s2 += fabsl(x[i + 2]);
            s3 += fabsl(x[i + 3]);
            s4 += fabsl(x[i + 4]);
            s5 += fabsl(x[i + 5]);
        }
        s0 += s1 + s2 + s3 + s4 + s5;
        s1 = 0.0L;
        for (; i < n; ++i) s0 += fabsl(x[i]);
    } else {
        for (int i = 0, ix = 0; i < n; ++i, ix += incx) s0 += fabsl(x[ix]);
    }
    return s0 + s1;
}
