/* mcabs1 — multifloats: |re(z)| + |im(z)| for a single complex DD. */
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
/* Inline magnitude — the public fabsdd() is an out-of-line library call, so
 * using it here emits two PLT calls (~2x slower than ob's inlined abs). The
 * branchy a < 0 ? -a : a uses only header-inline constexpr ops and yields the
 * identical canonical DD magnitude. Matches imamax/mrotmg. (A branchless SSE2
 * pack of both limbs was tried and measured WORSE — 4.8ns vs 3.2ns: the DD
 * two-sum add is scalar, so vector pack/unpack is pure overhead.) */
inline R r_abs(R a) { return a < 0 ? -a : a; }
}

extern "C" R mcabs1_(const T *z_)
{
    const T z = *z_;
    return r_abs(z.re) + r_abs(z.im);
}
