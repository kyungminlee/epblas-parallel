/* yrotg — kind10 complex Givens generator (ZROTG analog).
 *
 * The migrated reference implements Anderson 2017 "Safe Scaling in
 * Level 1 BLAS" — direct algebra on ABSSQ(t) = Re(t)² + Im(t)² and
 * inline `fsqrt`, zero external libm calls.
 *
 * Earlier overlay used three `hypotl()` calls (one each for |a|, |b|,
 * sqrt(|a|² + |b|²)). Each is the glibc long-double software routine
 * (~100-300 cycles); cumulatively the dominant cost. The 0.48× speedup
 * vs migrated was entirely this libm overhead.
 *
 * This version follows the migrated's unscaled fast path: 2 sqrtl
 * (each compiles to a single x87 fsqrt) plus direct complex arithmetic.
 * Falls back to the same scaled algorithm in the extreme branches.
 */
#include <math.h>
typedef _Complex long double T;
typedef long double R;

void yrotg_(T *a_, const T *b_, R *c, T *s)
{
    const T a = *a_, b = *b_;
    const R ar = __real__ a, ai = __imag__ a;
    const R br = __real__ b, bi = __imag__ b;
    const R g2 = br * br + bi * bi;     /* |b|² */
    if (g2 == 0.0L) {
        *c = 1.0L;
        *s = 0.0L;
        return;
    }
    const R f2 = ar * ar + ai * ai;     /* |a|² */
    if (f2 == 0.0L) {
        *c = 0.0L;
        const R d = sqrtl(g2);          /* |b| */
        *s = (br - bi * 1.0iL) / d;     /* conj(b)/|b| */
        *a_ = d;                        /* r = |b| (real) */
        return;
    }
    const R h2 = f2 + g2;
    *c = sqrtl(f2 / h2);                /* |a|/sqrt(|a|²+|b|²) */
    *a_ = a / *c;                       /* r = a/c = sgn(a)·sqrt(|a|²+|b|²) */
    /* s = conj(b) * (a / sqrt(f2*h2)) — equivalent to sgn(a)·conj(b)/sqrt(h2) */
    const R d = sqrtl(f2 * h2);
    const T conjb = br - bi * 1.0iL;
    *s = conjb * (a / d);
}

/* No integer arguments -> LP64/ILP64 ABIs identical; _64_ twin tail-calls. */
void yrotg_64_(T *a_, const T *b_, R *c, T *s) { yrotg_(a_, b_, c, s); }
