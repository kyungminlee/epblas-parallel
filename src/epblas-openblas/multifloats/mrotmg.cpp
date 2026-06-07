/*
 * mrotmg — kind10 port of OpenBLAS drotmg.  Generate modified Givens.
 *
 * In/out scalars dd1, dd2, dx1; in scalar dy1; out array dparam[5].
 * Reference: blas/src/mrotmg.f.
 */
#include <multifloats.h>
#include <multifloats/float64x2.h>
typedef multifloats::float64x2 T;

static inline T ldabs(T x) { return x < 0 ? -x : x; }

extern "C" void mrotmg_(T *DD1, T *DD2, T *DX1, const T *DY1, T *DPARAM)
{
    static const T ZERO = 0.0, ONE = 1.0, TWO = 2.0;
    static const T GAM    = 4096.0;
    static const T GAMSQ  = 16777216.0;            /* GAM*GAM */
    static const T RGAMSQ = 5.9604645e-8;          /* 1/GAMSQ approx */

    T dd1 = *DD1, dd2 = *DD2, dx1 = *DX1, dy1 = *DY1;
    T dflag = -ONE;
    T dh11 = ZERO, dh12 = ZERO, dh21 = ZERO, dh22 = ZERO;
    T dp1, dp2, dq1, dq2, du, dtemp;

    if (dd1 < ZERO) {
        /* GO ZERO-H-D-AND-DX1 */
        dflag = -ONE;
        dh11 = ZERO; dh12 = ZERO; dh21 = ZERO; dh22 = ZERO;
        dd1 = ZERO; dd2 = ZERO; dx1 = ZERO;
    } else {
        dp2 = dd2 * dy1;
        if (dp2 == ZERO) {
            dflag = -TWO;
            DPARAM[0] = dflag;
            *DD1 = dd1; *DD2 = dd2; *DX1 = dx1;
            return;
        }
        dp1 = dd1 * dx1;
        dq2 = dp2 * dy1;
        dq1 = dp1 * dx1;

        if (ldabs(dq1) > ldabs(dq2)) {
            dh21 = -dy1 / dx1;
            dh12 = dp2 / dp1;
            du = ONE - dh12 * dh21;
            if (du > ZERO) {
                dflag = ZERO;
                dd1 = dd1 / du;
                dd2 = dd2 / du;
                dx1 = dx1 * du;
            } else {
                /* Edge-case safety. */
                dflag = -ONE;
                dh11 = ZERO; dh12 = ZERO; dh21 = ZERO; dh22 = ZERO;
                dd1 = ZERO; dd2 = ZERO; dx1 = ZERO;
            }
        } else {
            if (dq2 < ZERO) {
                /* GO ZERO-H-D-AND-DX1 */
                dflag = -ONE;
                dh11 = ZERO; dh12 = ZERO; dh21 = ZERO; dh22 = ZERO;
                dd1 = ZERO; dd2 = ZERO; dx1 = ZERO;
            } else {
                dflag = ONE;
                dh11 = dp1 / dp2;
                dh22 = dx1 / dy1;
                du = ONE + dh11 * dh22;
                dtemp = dd2 / du;
                dd2 = dd1 / du;
                dd1 = dtemp;
                dx1 = dy1 * du;
            }
        }

        /* SCALE-CHECK on dd1 */
        if (dd1 != ZERO) {
            while ((dd1 <= RGAMSQ) || (dd1 >= GAMSQ)) {
                if (dflag == ZERO) {
                    dh11 = ONE; dh22 = ONE;
                    dflag = -ONE;
                } else {
                    dh21 = -ONE; dh12 = ONE;
                    dflag = -ONE;
                }
                if (dd1 <= RGAMSQ) {
                    dd1 = dd1 * (GAM * GAM);
                    dh11 = dh11 / GAM;
                    dh12 = dh12 / GAM;
                } else {
                    dd1 = dd1 / (GAM * GAM);
                    dh11 = dh11 * GAM;
                    dh12 = dh12 * GAM;
                }
            }
        }
        /* SCALE-CHECK on dd2 */
        if (dd2 != ZERO) {
            while ((ldabs(dd2) <= RGAMSQ) || (ldabs(dd2) >= GAMSQ)) {
                if (dflag == ZERO) {
                    dh11 = ONE; dh22 = ONE;
                    dflag = -ONE;
                } else {
                    dh21 = -ONE; dh12 = ONE;
                    dflag = -ONE;
                }
                if (ldabs(dd2) <= RGAMSQ) {
                    dd2 = dd2 * (GAM * GAM);
                    dh21 = dh21 / GAM;
                    dh22 = dh22 / GAM;
                } else {
                    dd2 = dd2 / (GAM * GAM);
                    dh21 = dh21 * GAM;
                    dh22 = dh22 * GAM;
                }
            }
        }
    }

    if (dflag < ZERO) {
        DPARAM[1] = dh11; DPARAM[2] = dh21;
        DPARAM[3] = dh12; DPARAM[4] = dh22;
    } else if (dflag == ZERO) {
        DPARAM[2] = dh21; DPARAM[3] = dh12;
    } else {
        DPARAM[1] = dh11; DPARAM[4] = dh22;
    }
    DPARAM[0] = dflag;
    *DD1 = dd1; *DD2 = dd2; *DX1 = dx1;
}
