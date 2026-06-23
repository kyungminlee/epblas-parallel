/* mrotmg — multifloats real DD: generate modified Givens.
 * Port of LAPACK reference DROTMG. Computes the H matrix and updated
 * (d1, d2, x1) such that H applied to (sqrt(d1)·x1, sqrt(d2)·y1) zeros
 * the second component.
 */
#include <multifloats.h>
#include "mf_pred.h"
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using TR = mf::float64x2;

namespace {
/* lt0/eq0/gt0/mag/abs_gt from mf_pred (limb-lexicographic; mag stays inline to
 * dodge the out-of-line fabsdd PLT calls — see mf_pred.h). */
using mf_pred::lt0;
using mf_pred::eq0;
using mf_pred::gt0;
using mf_pred::mag;
using mf_pred::abs_gt;
}

extern "C" void mrotmg_(TR *d1_, TR *d2_, TR *x1_, const TR *y1_, TR *dparam)
{
    const TR zero{0.0, 0.0}, one{1.0, 0.0}, two{2.0, 0.0};
    const TR gam{4096.0, 0.0};
    const TR gamsq{16777216.0, 0.0};
    const TR rgamsq{5.9604645e-8, 0.0};
    TR d1 = *d1_, d2 = *d2_, x1 = *x1_, y1 = *y1_;
    TR flag, h11{}, h12{}, h21{}, h22{};

    if (lt0(d1)) {
        flag = TR{-1.0, 0.0};
        h11 = h12 = h21 = h22 = zero;
        d1 = d2 = x1 = zero;
    } else {
        TR p2 = d2 * y1;
        if (eq0(p2)) { dparam[0] = TR{-2.0, 0.0}; return; }
        TR p1 = d1 * x1;
        TR q2 = p2 * y1;
        TR q1 = p1 * x1;
        if (abs_gt(q1, q2)) {
            h21 = TR{-y1.limbs[0], -y1.limbs[1]} / x1;
            h12 = p2 / p1;
            TR u = one - h12 * h21;
            if (gt0(u)) {
                flag = zero;
                d1 = d1 / u; d2 = d2 / u; x1 = x1 * u;
            } else {
                flag = TR{-1.0, 0.0};
                h11 = h12 = h21 = h22 = zero;
                d1 = d2 = x1 = zero;
            }
        } else {
            if (lt0(q2)) {
                flag = TR{-1.0, 0.0};
                h11 = h12 = h21 = h22 = zero;
                d1 = d2 = x1 = zero;
            } else {
                flag = one;
                h11 = p1 / p2;
                h22 = x1 / y1;
                TR u = one + h11 * h22;
                TR tmp = d2 / u;
                d2 = d1 / u;
                d1 = tmp;
                x1 = y1 * u;
            }
        }
        /* SCALE-CHECK */
        if (!eq0(d1)) {
            while ((abs_gt(rgamsq, d1) || !abs_gt(gamsq, d1))) {
                if (eq0(flag)) { h11 = one; h22 = one; flag = TR{-1.0, 0.0}; }
                else              { h21 = TR{-1.0, 0.0}; h12 = one; flag = TR{-1.0, 0.0}; }
                if (abs_gt(rgamsq, d1)) { d1 = d1 * gam * gam; x1 = x1 / gam;
                    h11 = h11 / gam; h12 = h12 / gam; }
                else                       { d1 = d1 / (gam * gam); x1 = x1 * gam;
                    h11 = h11 * gam; h12 = h12 * gam; }
                if (abs_gt(rgamsq, d1) || !abs_gt(gamsq, d1)) continue; else break;
            }
        }
        if (!eq0(d2)) {
            while (abs_gt(rgamsq, mag(d2)) || !abs_gt(gamsq, mag(d2))) {
                if (eq0(flag)) { h11 = one; h22 = one; flag = TR{-1.0, 0.0}; }
                else              { h21 = TR{-1.0, 0.0}; h12 = one; flag = TR{-1.0, 0.0}; }
                if (abs_gt(rgamsq, mag(d2))) { d2 = d2 * gam * gam;
                    h21 = h21 / gam; h22 = h22 / gam; }
                else                                { d2 = d2 / (gam * gam);
                    h21 = h21 * gam; h22 = h22 * gam; }
                if (abs_gt(rgamsq, mag(d2)) || !abs_gt(gamsq, mag(d2))) continue; else break;
            }
        }
    }
    dparam[0] = flag;
    if (lt0(flag))      { dparam[1]=h11; dparam[2]=h21; dparam[3]=h12; dparam[4]=h22; }
    else if (eq0(flag)) { dparam[3]=h12; dparam[2]=h21; }
    else                   { dparam[1]=h11; dparam[4]=h22; }
    *d1_ = d1; *d2_ = d2; *x1_ = x1;
}
/* ILP64 twin — no integer args, so the ABI is identical to LP64. */
extern "C" void mrotmg_64_(TR *d1_, TR *d2_, TR *x1_, const TR *y1_, TR *dparam)
{ mrotmg_(d1_, d2_, x1_, y1_, dparam); }
