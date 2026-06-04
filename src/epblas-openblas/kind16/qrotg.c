/*
 * qrotg — kind16 port of LAPACK 3.12.1 drotg (Anderson 2017 safe scaling).
 *
 * Generates plane rotation:
 *   [  c  s ] [ a ] = [ r ]
 *   [ -s  c ] [ b ]   [ 0 ]
 *
 * Inputs/outputs (all REAL(KIND=16)):
 *   a (in/out): on entry the input a; on exit the rotated r.
 *   b (in/out): on entry the input b; on exit z (auxiliary).
 *   c (out):   the cosine.
 *   s (out):   the sine.
 *
 * Reference: blas/src/qrotg.f90.
 */
#include <math.h>
#include <quadmath.h>
#include <float.h>

typedef __float128 T;

static T safmin, safmax;
static int safscale_initialized = 0;

static void safscale_init(void)
{
    int min_exp = LDBL_MIN_EXP;
    int max_exp = LDBL_MAX_EXP;
    /* safmin = radix^max(minexp - 1, 1 - maxexp)
     * safmax = radix^max(1 - minexp, maxexp - 1) */
    int s_min = (min_exp - 1) > (1 - max_exp) ? (min_exp - 1) : (1 - max_exp);
    int s_max = (1 - min_exp) > (max_exp - 1) ? (1 - min_exp) : (max_exp - 1);
    safmin = ldexpq(1.0Q, s_min);
    safmax = ldexpq(1.0Q, s_max);
    safscale_initialized = 1;
}

static inline T ldabs(T x) { return x < 0 ? -x : x; }
static inline T ldsign1(T x) { return x < 0 ? -1.0Q : 1.0Q; }

void qrotg_(T *a, T *b, T *c, T *s)
{
    if (!safscale_initialized) safscale_init();
    T av = *a, bv = *b;
    T anorm = ldabs(av), bnorm = ldabs(bv);

    if (bnorm == 0.0Q) {
        *c = 1.0Q; *s = 0.0Q; *b = 0.0Q;
    } else if (anorm == 0.0Q) {
        *c = 0.0Q; *s = 1.0Q;
        *a = bv;
        *b = 1.0Q;
    } else {
        T scl = anorm > bnorm ? anorm : bnorm;
        if (scl > safmax) scl = safmax;
        if (scl < safmin) scl = safmin;
        T sigma = anorm > bnorm ? ldsign1(av) : ldsign1(bv);
        T ar = av / scl, br = bv / scl;
        T r = sigma * (scl * sqrtq(ar*ar + br*br));
        T cv = av / r, sv = bv / r;
        T z;
        if (anorm > bnorm) {
            z = sv;
        } else if (cv != 0.0Q) {
            z = 1.0Q / cv;
        } else {
            z = 1.0Q;
        }
        *a = r;
        *b = z;
        *c = cv;
        *s = sv;
    }
}
