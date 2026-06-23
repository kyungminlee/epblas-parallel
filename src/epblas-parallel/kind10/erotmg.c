/* erotmg — kind10 real: generate modified Givens. Port of LAPACK DROTMG. */
#include <math.h>
typedef long double TR;

void erotmg_(TR *d1_, TR *d2_, TR *x1_, const TR *y1_, TR *dparam)
{
    TR d1 = *d1_, d2 = *d2_, x1 = *x1_;
    const TR y1 = *y1_;
    const TR gam = 4096.0L, gamsq = 16777216.0L, rgamsq = 5.9604645e-8L;
    TR flag, h11 = 0, h12 = 0, h21 = 0, h22 = 0;

    if (d1 < 0.0L) {
        flag = -1.0L;
        d1 = d2 = x1 = 0.0L;
    } else {
        TR p2 = d2 * y1;
        if (p2 == 0.0L) { dparam[0] = -2.0L; return; }
        TR p1 = d1 * x1;
        TR q1 = p1 * x1, q2 = p2 * y1;
        if (fabsl(q1) > fabsl(q2)) {
            h21 = -y1 / x1;
            h12 = p2 / p1;
            TR u = 1.0L - h12 * h21;
            if (u > 0.0L) { flag = 0.0L; d1 /= u; d2 /= u; x1 *= u; }
            else          { flag = -1.0L; h12 = h21 = 0.0L; d1 = d2 = x1 = 0.0L; }
        } else {
            if (q2 < 0.0L) { flag = -1.0L; d1 = d2 = x1 = 0.0L; }
            else {
                flag = 1.0L;
                h11 = p1 / p2;
                h22 = x1 / y1;
                TR u = 1.0L + h11 * h22;
                TR t = d2 / u; d2 = d1 / u; d1 = t; x1 = y1 * u;
            }
        }
        /* Scale check loops */
        while (d1 != 0.0L && (fabsl(d1) <= rgamsq || fabsl(d1) >= gamsq)) {
            if (flag == 0.0L) { h11 = 1.0L; h22 = 1.0L; flag = -1.0L; }
            else              { h21 = -1.0L; h12 = 1.0L; flag = -1.0L; }
            if (fabsl(d1) <= rgamsq) { d1 *= gam*gam; x1 /= gam; h11 /= gam; h12 /= gam; }
            else                     { d1 /= gam*gam; x1 *= gam; h11 *= gam; h12 *= gam; }
        }
        while (d2 != 0.0L && (fabsl(d2) <= rgamsq || fabsl(d2) >= gamsq)) {
            if (flag == 0.0L) { h11 = 1.0L; h22 = 1.0L; flag = -1.0L; }
            else              { h21 = -1.0L; h12 = 1.0L; flag = -1.0L; }
            if (fabsl(d2) <= rgamsq) { d2 *= gam*gam; h21 /= gam; h22 /= gam; }
            else                     { d2 /= gam*gam; h21 *= gam; h22 *= gam; }
        }
    }
    dparam[0] = flag;
    if (flag < 0.0L)       { dparam[1]=h11; dparam[2]=h21; dparam[3]=h12; dparam[4]=h22; }
    else if (flag == 0.0L) { dparam[3]=h12; dparam[2]=h21; }
    else                   { dparam[1]=h11; dparam[4]=h22; }
    *d1_ = d1; *d2_ = d2; *x1_ = x1;
}

/* No integer arguments -> LP64/ILP64 ABIs identical; _64_ twin tail-calls. */
void erotmg_64_(TR *d1_, TR *d2_, TR *x1_, const TR *y1_, TR *dparam)
{ erotmg_(d1_, d2_, x1_, y1_, dparam); }
