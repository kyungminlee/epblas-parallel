/* qrotmg — kind16 real: generate modified Givens. */
#include <quadmath.h>
/* fabsq via __builtin_fabsf128 — single `pand` instead of a libquadmath function call. */
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __float128 T;

void qrotmg_(T *d1_, T *d2_, T *x1_, const T *y1_, T *dparam)
{
    T d1 = *d1_, d2 = *d2_, x1 = *x1_;
    const T y1 = *y1_;
    const T gam = 4096.0Q, gamsq = 16777216.0Q, rgamsq = 5.9604645e-8Q;
    T flag, h11 = 0, h12 = 0, h21 = 0, h22 = 0;

    if (d1 < 0.0Q) {
        flag = -1.0Q;
        d1 = d2 = x1 = 0.0Q;
    } else {
        T p2 = d2 * y1;
        if (p2 == 0.0Q) { dparam[0] = -2.0Q; return; }
        T p1 = d1 * x1;
        T q1 = p1 * x1, q2 = p2 * y1;
        if (fabsq(q1) > fabsq(q2)) {
            h21 = -y1 / x1;
            h12 = p2 / p1;
            T u = 1.0Q - h12 * h21;
            if (u > 0.0Q) { flag = 0.0Q; d1 /= u; d2 /= u; x1 *= u; }
            else          { flag = -1.0Q; h12 = h21 = 0.0Q; d1 = d2 = x1 = 0.0Q; }
        } else {
            if (q2 < 0.0Q) { flag = -1.0Q; d1 = d2 = x1 = 0.0Q; }
            else {
                flag = 1.0Q;
                h11 = p1 / p2;
                h22 = x1 / y1;
                T u = 1.0Q + h11 * h22;
                T t = d2 / u; d2 = d1 / u; d1 = t; x1 = y1 * u;
            }
        }
        while (d1 != 0.0Q && (fabsq(d1) <= rgamsq || fabsq(d1) >= gamsq)) {
            if (flag == 0.0Q) { h11 = 1.0Q; h22 = 1.0Q; flag = -1.0Q; }
            else              { h21 = -1.0Q; h12 = 1.0Q; flag = -1.0Q; }
            if (fabsq(d1) <= rgamsq) { d1 *= gam*gam; x1 /= gam; h11 /= gam; h12 /= gam; }
            else                     { d1 /= gam*gam; x1 *= gam; h11 *= gam; h12 *= gam; }
        }
        while (d2 != 0.0Q && (fabsq(d2) <= rgamsq || fabsq(d2) >= gamsq)) {
            if (flag == 0.0Q) { h11 = 1.0Q; h22 = 1.0Q; flag = -1.0Q; }
            else              { h21 = -1.0Q; h12 = 1.0Q; flag = -1.0Q; }
            if (fabsq(d2) <= rgamsq) { d2 *= gam*gam; h21 /= gam; h22 /= gam; }
            else                     { d2 /= gam*gam; h21 *= gam; h22 *= gam; }
        }
    }
    dparam[0] = flag;
    if (flag < 0.0Q)       { dparam[1]=h11; dparam[2]=h21; dparam[3]=h12; dparam[4]=h22; }
    else if (flag == 0.0Q) { dparam[3]=h12; dparam[2]=h21; }
    else                   { dparam[1]=h11; dparam[4]=h22; }
    *d1_ = d1; *d2_ = d2; *x1_ = x1;
}
/* ILP64 twin — no integer args, so the ABI is identical to LP64. */
void qrotmg_64_(T *d1_, T *d2_, T *x1_, const T *y1_, T *dparam) {
    qrotmg_(d1_, d2_, x1_, y1_, dparam);
}
