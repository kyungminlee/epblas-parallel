/* wrotg — multifloats: complex Givens generator (ZROTG analog).
 *   a, b complex; c real; s complex.
 *   c = |a| / sqrt(|a|² + |b|²)
 *   s = (a/|a|) · conj(b) / sqrt(|a|² + |b|²)
 *   r = (a/|a|) · sqrt(|a|² + |b|²);  a := r
 */
#include <multifloats.h>
#include "mf_kernels.h"
#include "mf_pred.h"
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;

namespace {
using mf_pred::eq0;
using mf_kernels::cconj;
using mf_kernels::cmul;
inline TC cdiv_real(TC const &a, R const &r) { return TC{ a.re / r, a.im / r }; }
}

/* Direct Re²+Im² algebra — avoids two cabsdd() calls (each is an
 * overflow-safe hypot, expensive in DD arithmetic). Same fix as
 * kind10/yrotg and kind16/xrotg. */
extern "C" void wrotg_(TC *a_, const TC *b_, R *c, TC *s)
{
    const TC a = *a_, b = *b_;
    const TC czero{R{0.0, 0.0}, R{0.0, 0.0}};
    const R g2 = b.re * b.re + b.im * b.im;     /* |b|² */
    if (eq0(g2)) {
        *c = R{1.0, 0.0};
        *s = czero;
        return;
    }
    const R f2 = a.re * a.re + a.im * a.im;     /* |a|² */
    if (eq0(f2)) {
        *c = R{0.0, 0.0};
        const R d = sqrtdd(g2);                 /* |b| */
        *s = cdiv_real(cconj(b), d);            /* conj(b)/|b| */
        a_->re = d;
        a_->im = R{0.0, 0.0};
        return;
    }
    const R h2 = f2 + g2;
    *c = sqrtdd(f2 / h2);                       /* |a|/sqrt(|a|²+|b|²) */
    *a_ = cdiv_real(a, *c);                     /* r = a/c */
    const R d = sqrtdd(f2 * h2);
    *s = cdiv_real(cmul(cconj(b), a), d);       /* conj(b)·a/sqrt(|a|²·(|a|²+|b|²)) */
}
/* ILP64 twin — no integer args, so the ABI is identical to LP64. */
extern "C" void wrotg_64_(TC *a_, const TC *b_, R *c, TC *s) { wrotg_(a_, b_, c, s); }
