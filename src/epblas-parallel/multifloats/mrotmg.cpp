/* mrotmg — multifloats real DD: generate modified Givens.
 * Port of LAPACK reference DROTMG. Computes the H matrix and updated
 * (d1, d2, x1) such that H applied to (sqrt(d1)·x1, sqrt(d2)·y1) zeros
 * the second component.
 */
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
inline bool dd_lt0(T x) { return x.limbs[0] < 0.0; }
inline bool dd_eq0(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_gt0(T x) { return x.limbs[0] > 0.0 || (x.limbs[0] == 0.0 && x.limbs[1] > 0.0); }
/* Inline magnitude — the public fabsdd() is an out-of-line library call (the
 * scalar generator emitted 27 PLT calls to it → ~1.27x vs ob's inlined abs).
 * a < 0 ? -a : a uses only header-inline ops and yields the identical canonical
 * magnitude (same idiom as imamax t_abs; verified bit-exact). */
inline T dd_abs(T a) { return a < 0 ? -a : a; }
inline bool dd_abs_gt(T a, T b) {
    T aa = dd_abs(a), bb = dd_abs(b);
    return aa.limbs[0] > bb.limbs[0]
        || (aa.limbs[0] == bb.limbs[0] && aa.limbs[1] > bb.limbs[1]);
}
}

extern "C" void mrotmg_(T *d1_, T *d2_, T *x1_, const T *y1_, T *dparam)
{
    const T zero{0.0, 0.0}, one{1.0, 0.0}, two{2.0, 0.0};
    const T gam{4096.0, 0.0};
    const T gamsq{16777216.0, 0.0};
    const T rgamsq{5.9604645e-8, 0.0};
    T d1 = *d1_, d2 = *d2_, x1 = *x1_, y1 = *y1_;
    T flag, h11{}, h12{}, h21{}, h22{};

    if (dd_lt0(d1)) {
        flag = T{-1.0, 0.0};
        h11 = h12 = h21 = h22 = zero;
        d1 = d2 = x1 = zero;
    } else {
        T p2 = d2 * y1;
        if (dd_eq0(p2)) { dparam[0] = T{-2.0, 0.0}; return; }
        T p1 = d1 * x1;
        T q2 = p2 * y1;
        T q1 = p1 * x1;
        if (dd_abs_gt(q1, q2)) {
            h21 = T{-y1.limbs[0], -y1.limbs[1]} / x1;
            h12 = p2 / p1;
            T u = one - h12 * h21;
            if (dd_gt0(u)) {
                flag = zero;
                d1 = d1 / u; d2 = d2 / u; x1 = x1 * u;
            } else {
                flag = T{-1.0, 0.0};
                h11 = h12 = h21 = h22 = zero;
                d1 = d2 = x1 = zero;
            }
        } else {
            if (dd_lt0(q2)) {
                flag = T{-1.0, 0.0};
                h11 = h12 = h21 = h22 = zero;
                d1 = d2 = x1 = zero;
            } else {
                flag = one;
                h11 = p1 / p2;
                h22 = x1 / y1;
                T u = one + h11 * h22;
                T tmp = d2 / u;
                d2 = d1 / u;
                d1 = tmp;
                x1 = y1 * u;
            }
        }
        /* SCALE-CHECK */
        if (!dd_eq0(d1)) {
            while ((dd_abs_gt(rgamsq, d1) || !dd_abs_gt(gamsq, d1))) {
                if (dd_eq0(flag)) { h11 = one; h22 = one; flag = T{-1.0, 0.0}; }
                else              { h21 = T{-1.0, 0.0}; h12 = one; flag = T{-1.0, 0.0}; }
                if (dd_abs_gt(rgamsq, d1)) { d1 = d1 * gam * gam; x1 = x1 / gam;
                    h11 = h11 / gam; h12 = h12 / gam; }
                else                       { d1 = d1 / (gam * gam); x1 = x1 * gam;
                    h11 = h11 * gam; h12 = h12 * gam; }
                if (dd_abs_gt(rgamsq, d1) || !dd_abs_gt(gamsq, d1)) continue; else break;
            }
        }
        if (!dd_eq0(d2)) {
            while (dd_abs_gt(rgamsq, dd_abs(d2)) || !dd_abs_gt(gamsq, dd_abs(d2))) {
                if (dd_eq0(flag)) { h11 = one; h22 = one; flag = T{-1.0, 0.0}; }
                else              { h21 = T{-1.0, 0.0}; h12 = one; flag = T{-1.0, 0.0}; }
                if (dd_abs_gt(rgamsq, dd_abs(d2))) { d2 = d2 * gam * gam;
                    h21 = h21 / gam; h22 = h22 / gam; }
                else                                { d2 = d2 / (gam * gam);
                    h21 = h21 * gam; h22 = h22 * gam; }
                if (dd_abs_gt(rgamsq, dd_abs(d2)) || !dd_abs_gt(gamsq, dd_abs(d2))) continue; else break;
            }
        }
    }
    dparam[0] = flag;
    if (dd_lt0(flag))      { dparam[1]=h11; dparam[2]=h21; dparam[3]=h12; dparam[4]=h22; }
    else if (dd_eq0(flag)) { dparam[3]=h12; dparam[2]=h21; }
    else                   { dparam[1]=h11; dparam[4]=h22; }
    *d1_ = d1; *d2_ = d2; *x1_ = x1;
}
