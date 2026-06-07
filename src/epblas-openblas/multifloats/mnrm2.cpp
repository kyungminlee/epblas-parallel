/*
 * mnrm2 — multifloats DD port of LAPACK 3.12.1 dnrm2 (Blue's algorithm).
 *
 * Three-accumulator scaled sum-of-squares to avoid overflow/underflow.
 * Faithful retype of kind10/enrm2.c (long double -> float64x2).
 *
 * ADAPTATION vs the verbatim kind10 source: Blue's scaling constants are
 * derived from the *double* exponent range (DBL_MIN_EXP/DBL_MAX_EXP), not
 * the host long double's. A double-double value is two binary64 limbs, so
 * its exponent range is binary64's (~1.8e308 / 2.2e-308); using the fp80
 * LDBL_* range would push the overflow-guard thresholds far beyond DBL_MAX
 * and defeat the scaling. mant_dig uses the per-limb 53 (conservative —
 * the thresholds only bracket the asml/amed/abig buckets). Constants are
 * computed once per process via ldexpdd().
 */
#include <cstddef>
#include <cfloat>
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using T = mf::float64x2;

static T btsml, btbig, bssml, bsbig, maxN;
static int blue_initialized = 0;

static void blue_init(void)
{
    /* Double-double rides on the binary64 exponent range. */
    int min_exp = DBL_MIN_EXP;
    int max_exp = DBL_MAX_EXP;
    int dig     = DBL_MANT_DIG;

    /* Blue's thresholds: ceil/floor of half * range. */
    #define CEIL2(x)  ( ((x) >= 0) ? ((x) + 1) / 2 : -((-(x)) / 2) )
    #define FLOOR2(x) ( ((x) >= 0) ? (x) / 2 : -(((-(x)) + 1) / 2) )

    btsml = mf::ldexpdd(T(1.0),  CEIL2(min_exp - 1));
    btbig = mf::ldexpdd(T(1.0),  FLOOR2(max_exp - dig + 1));
    bssml = mf::ldexpdd(T(1.0), -FLOOR2(min_exp - dig));
    bsbig = mf::ldexpdd(T(1.0), -CEIL2(max_exp + dig - 1));
    maxN  = T(DBL_MAX);
    #undef CEIL2
    #undef FLOOR2

    blue_initialized = 1;
}

static inline T ldabs(T x) { return x < 0.0 ? -x : x; }

extern "C" T mnrm2_(const int *N, const T *x, const int *INCX)
{
    std::ptrdiff_t n    = (std::ptrdiff_t)(*N);
    std::ptrdiff_t incx = (std::ptrdiff_t)(*INCX);

    if (n <= 0) return T(0.0);
    if (!blue_initialized) blue_init();

    T asml = 0.0, amed = 0.0, abig = 0.0;
    int notbig = 1;

    std::ptrdiff_t ix = 0;
    if (incx < 0) ix = -(n - 1) * incx;

    for (std::ptrdiff_t i = 0; i < n; ++i) {
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
    if (abig > 0.0) {
        if (amed > 0.0 || amed > maxN || amed != amed) {
            abig = abig + (amed * bsbig) * bsbig;
        }
        scl = T(1.0) / bsbig;
        sumsq = abig;
    } else if (asml > 0.0) {
        if (amed > 0.0 || amed > maxN || amed != amed) {
            T amed_s = mf::sqrt(amed);
            T asml_s = mf::sqrt(asml) / bssml;
            T ymin, ymax;
            if (asml_s > amed_s) { ymin = amed_s; ymax = asml_s; }
            else                 { ymin = asml_s; ymax = amed_s; }
            scl = 1.0;
            sumsq = ymax * ymax * (T(1.0) + (ymin/ymax) * (ymin/ymax));
        } else {
            scl = T(1.0) / bssml;
            sumsq = asml;
        }
    } else {
        scl = 1.0;
        sumsq = amed;
    }
    return scl * mf::sqrt(sumsq);
}
