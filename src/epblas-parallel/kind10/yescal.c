#include <stddef.h>
/* yescal — kind10: X := α·X with α real, X complex (CSSCAL/ZDSCAL analog).
 *
 * Treat the complex array as a 2N-long real array of (re, im) pairs.
 * The previous `... * 1.0iL` idiom triggered gcc's complex-multiplication
 * expansion (4 fmul + 2 fadd including products by zero); writing the
 * real and imag scales explicitly produces a tight 2-fmul-per-element
 * inner loop. */
typedef _Complex long double T;
typedef long double R;

void yescal_(const int *n_, const R *alpha_, T *x, const int *incx_)
{
    const ptrdiff_t n = *n_, incx = *incx_;
    const R alpha = *alpha_;
    if (n <= 0 || alpha == 1.0L) return;
    long double *base = (long double *)x;
    if (incx == 1) {
        long double *p = base;
        long double *e = p + 2 * (long)n;
        for (; p < e; p += 2) {
            p[0] *= alpha;
            p[1] *= alpha;
        }
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            base[2*ix]     *= alpha;
            base[2*ix + 1] *= alpha;
            ix += incx;
        }
    }
}
