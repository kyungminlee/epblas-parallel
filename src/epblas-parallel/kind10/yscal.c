#include <stddef.h>
/* yscal — kind10 complex: X := α·X (α and X both complex).
 *
 * Manual __real__/__imag__ expansion with imag-part products written
 * as `xi*ar + xr*ai` (X.im term first). gcc emits x87 products in
 * source order; this order matches the value already on top of the
 * x87 stack after the first store, saving one `fxch` per iter
 * relative to the natural `xr*ai + xi*ar` form. With the default
 * order, the inner loop is 15 insns per element with 2 fxch; this
 * form is 14 insns with 1 fxch — matches gfortran's ZSCAL codegen.
 */
typedef _Complex long double T;

void yscal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const ptrdiff_t n = *n_, incx = *incx_;
    if (n <= 0) return;
    const long double ar = __real__ *alpha_;
    const long double ai = __imag__ *alpha_;
    if (ar == 1.0L && ai == 0.0L) return;
    long double *base = (long double *)x;
    if (incx == 1) {
        long double *p = base;
        long double *e = p + 2 * (long)n;
        for (; p < e; p += 2) {
            const long double xr = p[0], xi = p[1];
            p[0] = xr * ar - xi * ai;
            p[1] = xi * ar + xr * ai;
        }
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            const long double xr = base[2*ix], xi = base[2*ix + 1];
            base[2*ix]     = xr * ar - xi * ai;
            base[2*ix + 1] = xi * ar + xr * ai;
            ix += incx;
        }
    }
}
