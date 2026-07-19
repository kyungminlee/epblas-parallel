/*
 * erotg — kind10 port of LAPACK 3.12.1 drotg (Anderson 2017 safe scaling).
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
 * Reference: blas/src/erotg.f90.
 */
#include <math.h>
#include <float.h>

typedef long double T;

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
    safmin = ldexpl(1.0L, s_min);
    safmax = ldexpl(1.0L, s_max);
    safscale_initialized = 1;
}

static inline T ldabs(T x) { return __builtin_fabsl(x); }  /* branchless x87 fabs */
static inline T ldsign1(T x) { return x < 0 ? -1.0L : 1.0L; }

void erotg_(T *a, T *b, T *c, T *s)
{
    if (!safscale_initialized) safscale_init();
    T av = *a, bv = *b;
    T anorm = ldabs(av), bnorm = ldabs(bv);

    if (bnorm == 0.0L) {
        *c = 1.0L; *s = 0.0L; *b = 0.0L;
    } else if (anorm == 0.0L) {
        *c = 0.0L; *s = 1.0L;
        *a = bv;
        *b = 1.0L;
    } else {
        T scl = anorm > bnorm ? anorm : bnorm;
        if (scl > safmax) scl = safmax;
        if (scl < safmin) scl = safmin;
        T sigma = anorm > bnorm ? ldsign1(av) : ldsign1(bv);
        T ar = av / scl, br = bv / scl;
        T r = sigma * (scl * sqrtl(ar*ar + br*br));
        T cv = av / r, sv = bv / r;
        T z;
        if (anorm > bnorm) {
            z = sv;
        } else if (cv != 0.0L) {
            z = 1.0L / cv;
        } else {
            z = 1.0L;
        }
        *a = r;
        *b = z;
        *c = cv;
        *s = sv;
    }
}
