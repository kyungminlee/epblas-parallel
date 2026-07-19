/*
 * mrotg — kind10 port of LAPACK 3.12.1 drotg (Anderson 2017 safe scaling).
 *
 * Generates plane rotation:
 *   [  c  s ] [ a ] = [ r ]
 *   [ -s  c ] [ b ]   [ 0 ]
 *
 * Inputs/outputs (all REAL(KIND=10)):
 *   a (in/out): on entry the input a; on exit the rotated r.
 *   b (in/out): on entry the input b; on exit z (auxiliary).
 *   c (out):   the cosine.
 *   s (out):   the sine.
 *
 * Reference: blas/src/mrotg.f90.
 */
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <cfloat>

typedef multifloats::float64x2 T;

static T safmin, safmax;
static int safscale_initialized = 0;

static void safscale_init(void)
{
    /* DD adaptation: the double-double type rides on the binary64 exponent
     * range, so the safe-scaling constants use DBL_* (NOT host fp80 LDBL_*). */
    int min_exp = DBL_MIN_EXP;
    int max_exp = DBL_MAX_EXP;
    /* safmin = radix^max(minexp - 1, 1 - maxexp)
     * safmax = radix^max(1 - minexp, maxexp - 1) */
    int s_min = (min_exp - 1) > (1 - max_exp) ? (min_exp - 1) : (1 - max_exp);
    int s_max = (1 - min_exp) > (max_exp - 1) ? (1 - min_exp) : (max_exp - 1);
    safmin = multifloats::ldexpdd(1.0, s_min);
    safmax = multifloats::ldexpdd(1.0, s_max);
    safscale_initialized = 1;
}

static inline T dd_abs(T x) { return multifloats::fabs(x); }
static inline T ldsign1(T x) { return x < 0 ? -1.0 : 1.0; }

extern "C" void mrotg_(T *a, T *b, T *c, T *s)
{
    if (!safscale_initialized) safscale_init();
    T av = *a, bv = *b;
    T anorm = dd_abs(av), bnorm = dd_abs(bv);

    if (bnorm == 0.0) {
        *c = 1.0; *s = 0.0; *b = 0.0;
    } else if (anorm == 0.0) {
        *c = 0.0; *s = 1.0;
        *a = bv;
        *b = 1.0;
    } else {
        T scl = anorm > bnorm ? anorm : bnorm;
        if (scl > safmax) scl = safmax;
        if (scl < safmin) scl = safmin;
        T sigma = anorm > bnorm ? ldsign1(av) : ldsign1(bv);
        T ar = av / scl, br = bv / scl;
        T r = sigma * (scl * multifloats::sqrt(ar*ar + br*br));
        T cv = av / r, sv = bv / r;
        T z;
        if (anorm > bnorm) {
            z = sv;
        } else if (cv != 0.0) {
            z = T(1.0) / cv;
        } else {
            z = 1.0;
        }
        *a = r;
        *b = z;
        *c = cv;
        *s = sv;
    }
}
