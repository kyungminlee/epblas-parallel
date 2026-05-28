/* mrotg — real DD Givens generator.
 * Classical algorithm. Given (a, b), compute c, s, r=sign(a)·sqrt(a²+b²)
 * such that [c s; -s c]·[a; b] = [r; 0]. Returns r in a, z in b.
 */
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_lt(T a, T b) {
    return a.limbs[0] < b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] < b.limbs[1]);
}
inline bool dd_gt(T a, T b) {
    return a.limbs[0] > b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] > b.limbs[1]);
}
}

extern "C" void mrotg_(T *a, T *b, T *c, T *s)
{
    T zero{0.0, 0.0}, one{1.0, 0.0};
    T anorm = fabsdd(*a), bnorm = fabsdd(*b);
    if (dd_iszero(bnorm)) { *c = one; *s = zero; *b = zero; return; }
    if (dd_iszero(anorm)) { *c = zero; *s = one; *a = *b; *b = one; return; }
    T r = sqrtdd(anorm * anorm + bnorm * bnorm);
    if (dd_lt(*a, zero)) r = T{-r.limbs[0], -r.limbs[1]};
    *c = *a / r;
    *s = *b / r;
    T z = dd_gt(anorm, bnorm) ? *s : (one / *c);
    *a = r;
    *b = z;
}
