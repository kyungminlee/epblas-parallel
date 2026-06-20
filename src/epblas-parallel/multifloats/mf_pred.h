/*
 * mf_pred.h — scalar predicate vocabulary for real double-double (float64x2).
 *
 * The Givens/argmax/abs routines (mrotg/mrotmg/mrotm/wrotg/iwamax/imamax/...)
 * each re-derived the same handful of limb-lexicographic comparisons against a
 * normalized DD. They are precision-/algorithm-agnostic integer-of-limbs tests,
 * so they live here once. This also hosts the zero/one equality checks the L2/L3
 * routines each re-derived (eq0/eq1 for real DD, ceq0/ceq1 for complex DD; the
 * old per-file dd_iszero/dd_isone/cdd_iszero/cdd_isone with their divergent
 * T-by-value / R-by-ref / (double,double) signatures all collapse onto these).
 * Distinct from the SIMD-DD arithmetic in mf_simd_fast.h / mf_simd_exact.h.
 *
 * A normalized float64x2 keeps |limbs[1]| <= ulp(limbs[0]), so the sign and
 * ordering are decided by limbs[0] first and limbs[1] only on a hi-limb tie.
 * Cold path (one call per generate / O(N) per argmax), so these stay as plain
 * branchy scalar tests, bit-for-bit with the originals.
 */
#pragma once

#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf_pred {

using dd  = multifloats::float64x2;
using cdd = multifloats::complex64x2;

inline bool eq0(dd x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }

/* == 1 (exactly one, hi==1 lo==0). The to-zero/to-one pair the L2/L3 routines
 * test alpha/beta/diagonal against — same limb-lexicographic family as eq0. */
inline bool eq1(dd x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }

/* Two-loose-limbs overload for the deinterleaved SoA rank-update kernels
 * (mspr/msyr/mspr2/msyr2) that hold hi/lo in separate arrays. */
inline bool eq0(double h, double l) { return h == 0.0 && l == 0.0; }

/* Complex DD == 0 / == 1 (re,im each a float64x2). */
inline bool ceq0(cdd const &z) { return eq0(z.re) && eq0(z.im); }
inline bool ceq1(cdd const &z) { return eq1(z.re) && eq0(z.im); }

inline bool lt0(dd x) { return x.limbs[0] < 0.0 || (x.limbs[0] == 0.0 && x.limbs[1] < 0.0); }

inline bool gt0(dd x) { return x.limbs[0] > 0.0 || (x.limbs[0] == 0.0 && x.limbs[1] > 0.0); }

inline bool lt(dd a, dd b) {
    return a.limbs[0] < b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] < b.limbs[1]);
}

inline bool gt(dd a, dd b) {
    return a.limbs[0] > b.limbs[0]
        || (a.limbs[0] == b.limbs[0] && a.limbs[1] > b.limbs[1]);
}

/* Inline DD magnitude: deliberately NOT named `abs` — the library's inline
 * multifloats::abs(float64x2) is ADL-reachable for a float64x2 argument and would
 * make every unqualified abs() call ambiguous. `mag` keeps the exact branchy
 * a<0?-a:a semantics the callers were bit-exact against (distinct from the
 * library's fabs-based abs), header-inline so no out-of-line fabsdd PLT call. */
inline dd mag(dd a) { return a < 0 ? -a : a; }

inline bool abs_gt(dd a, dd b) { return gt(mag(a), mag(b)); }

}  // namespace mf_pred
