/*
 * enrm2 — kind10 port of LAPACK 3.12.1 dnrm2 (Blue's algorithm).
 *
 * Three-accumulator scaled sum-of-squares to avoid overflow/underflow.
 * Reference: blas/src/enrm2.f90.
 *
 * Scaling constants computed once per process from <float.h> /
 * ldexpl() so they survive across calls without recomputing.
 */
#include <stddef.h>
#include <math.h>
#include <float.h>

typedef long double T;

static T btsml, btbig, bssml, bsbig, maxN;
static int blue_initialized = 0;

static void blue_init(void)
{
    /* LDBL_MIN_EXP = minimum normalized exponent (radix^(e-1) form) =
     * -16381 on x86 extended.  LDBL_MAX_EXP = 16384.  LDBL_MANT_DIG = 64. */
    int min_exp = LDBL_MIN_EXP;
    int max_exp = LDBL_MAX_EXP;
    int dig     = LDBL_MANT_DIG;

    /* Blue's thresholds: ceil/floor of half * range. */
    int e_btsml = (min_exp - 1 + 1) / 2;        /* ceil((min_exp-1)/2) */
    if (((min_exp - 1) & 1) && (min_exp - 1) < 0) e_btsml = (min_exp - 1) / 2;
    /* Use safer compute via ldexpl directly with the exact LAPACK
     * formula:
     *     btsml = radix^ceiling((minexp - 1) * 0.5)
     *     btbig = radix^floor   ((maxexp - dig + 1) * 0.5)
     *     bssml = radix^(-floor ((minexp - dig)     * 0.5))
     *     bsbig = radix^(-ceiling((maxexp + dig - 1) * 0.5))
     * Helper macros for ceil/floor of integer halves. */
    #define CEIL2(x)  ( ((x) >= 0) ? ((x) + 1) / 2 : -((-(x)) / 2) )
    #define FLOOR2(x) ( ((x) >= 0) ? (x) / 2 : -(((-(x)) + 1) / 2) )

    btsml = ldexpl(1.0L,  CEIL2(min_exp - 1));
    btbig = ldexpl(1.0L,  FLOOR2(max_exp - dig + 1));
    bssml = ldexpl(1.0L, -FLOOR2(min_exp - dig));
    bsbig = ldexpl(1.0L, -CEIL2(max_exp + dig - 1));
    maxN  = LDBL_MAX;
    #undef CEIL2
    #undef FLOOR2

    blue_initialized = 1;
}

static inline T ldabs(T x) { return x < 0 ? -x : x; }

T enrm2_(const int *N, const T *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);

    if (n <= 0) return 0.0L;
    if (!blue_initialized) blue_init();

    T asml = 0.0L, amed = 0.0L, abig = 0.0L;
    int notbig = 1;

    ptrdiff_t ix = 0;
    if (incx < 0) ix = -(n - 1) * incx;

    for (ptrdiff_t i = 0; i < n; ++i) {
        T ax = ldabs(x[ix]);
        if (ax > btbig) {
            T t = ax * bsbig;
            abig += t * t;
            notbig = 0;
        } else if (ax < btsml) {
            if (notbig) {
                T t = ax * bssml;
                asml += t * t;
            }
        } else {
            amed += ax * ax;
        }
        ix += incx;
    }

    T scl, sumsq;
    if (abig > 0.0L) {
        if (amed > 0.0L || amed > maxN || amed != amed) {
            abig = abig + (amed * bsbig) * bsbig;
        }
        scl = 1.0L / bsbig;
        sumsq = abig;
    } else if (asml > 0.0L) {
        if (amed > 0.0L || amed > maxN || amed != amed) {
            T amed_s = sqrtl(amed);
            T asml_s = sqrtl(asml) / bssml;
            T ymin, ymax;
            if (asml_s > amed_s) { ymin = amed_s; ymax = asml_s; }
            else                 { ymin = asml_s; ymax = amed_s; }
            scl = 1.0L;
            sumsq = ymax * ymax * (1.0L + (ymin/ymax) * (ymin/ymax));
        } else {
            scl = 1.0L / bssml;
            sumsq = asml;
        }
    } else {
        scl = 1.0L;
        sumsq = amed;
    }
    return scl * sqrtl(sumsq);
}
