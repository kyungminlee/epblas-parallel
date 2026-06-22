/* mcabs1 — multifloats: |re(z)| + |im(z)| for a single complex DD. */
#include <multifloats.h>
#include "mf_pred.h"
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
/* Inline magnitude from mf_pred — the public fabsdd() is out-of-line (two PLT
 * calls, ~2x slower than ob's inlined abs); mf_pred::mag is header-inline and
 * yields the identical canonical DD magnitude. (A branchless SSE2 pack of both
 * limbs was tried and measured WORSE — 4.8ns vs 3.2ns: the DD two-sum add is
 * scalar, so vector pack/unpack is pure overhead.) */
using mf_pred::mag;
}

extern "C" R mcabs1_(const T *z_)
{
    const T z = *z_;
    return mag(z.re) + mag(z.im);
}
/* ILP64 twin — no integer args, so the ABI is identical to LP64. */
extern "C" R mcabs1_64_(const T *z_) { return mcabs1_(z_); }
