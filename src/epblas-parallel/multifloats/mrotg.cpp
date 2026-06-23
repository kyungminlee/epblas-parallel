/* mrotg — real DD Givens generator.
 * Classical algorithm. Given (a, b), compute c, s, r=sign(a)·sqrt(a²+b²)
 * such that [c s; -s c]·[a; b] = [r; 0]. Returns r in a, z in b.
 */
#include <multifloats.h>
#include "mf_pred.h"
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using TR = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
namespace {
using mf_pred::lt;
using mf_pred::gt;
}

extern "C" void mrotg_(TR *a, TR *b, TR *c, TR *s)
{
    TR zero{0.0, 0.0}, one{1.0, 0.0};
    TR anorm = fabsdd(*a), bnorm = fabsdd(*b);
    if (eq0(bnorm)) { *c = one; *s = zero; *b = zero; return; }
    if (eq0(anorm)) { *c = zero; *s = one; *a = *b; *b = one; return; }
    TR r = sqrtdd(anorm * anorm + bnorm * bnorm);
    if (lt(*a, zero)) r = TR{-r.limbs[0], -r.limbs[1]};
    *c = *a / r;
    *s = *b / r;
    TR z = gt(anorm, bnorm) ? *s : (one / *c);
    *a = r;
    *b = z;
}
/* ILP64 twin — no integer args, so the ABI is identical to LP64. */
extern "C" void mrotg_64_(TR *a, TR *b, TR *c, TR *s) { mrotg_(a, b, c, s); }
