/* xrotg — kind16 complex Givens generator.
 *
 * Same overhaul as kind10's yrotg: three hypotq() calls replaced by
 * direct Re²+Im² algebra + two sqrtq calls. Eliminates the libquadmath
 * hypotq cost — was the 0.77× speedup ceiling vs migrated.
 */
#include <quadmath.h>
typedef __complex128 T;
typedef __float128 R;

void xrotg_(T *a_, const T *b_, R *c, T *s)
{
    const T a = *a_, b = *b_;
    const R ar = __real__ a, ai = __imag__ a;
    const R br = __real__ b, bi = __imag__ b;
    const R g2 = br * br + bi * bi;
    if (g2 == 0.0Q) {
        *c = 1.0Q;
        __real__ *s = 0;
        __imag__ *s = 0;
        return;
    }
    const R f2 = ar * ar + ai * ai;
    if (f2 == 0.0Q) {
        *c = 0.0Q;
        const R d = sqrtq(g2);
        T conjb; __real__ conjb = br; __imag__ conjb = -bi;
        *s = conjb / d;
        __real__ *a_ = d;
        __imag__ *a_ = 0;
        return;
    }
    const R h2 = f2 + g2;
    *c = sqrtq(f2 / h2);
    *a_ = a / *c;
    const R d = sqrtq(f2 * h2);
    T conjb; __real__ conjb = br; __imag__ conjb = -bi;
    *s = conjb * (a / d);
}
