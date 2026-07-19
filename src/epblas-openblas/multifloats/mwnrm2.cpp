/*
 * mwnrm2 — multifloats DD port of LAPACK 3.12.1 dznrm2 (Blue's algorithm).
 *
 * Real Euclidean norm of a complex-DD vector.  Each complex element
 * contributes |Re|^2 + |Im|^2; Blue's three-accumulator scaling is applied
 * to Re and Im independently.  Faithful retype of kind10/eynrm2.c: the
 * complex element (complex64x2 = two float64x2 limbs) is read through a
 * float64x2* alias, exactly as the kind10 source reads through long double*.
 *
 * Blue's constants use the binary64 exponent range — see mnrm2.cpp.
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
    int min_exp = DBL_MIN_EXP;
    int max_exp = DBL_MAX_EXP;
    int dig     = DBL_MANT_DIG;
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

static inline T dd_abs(T x) { return mf::fabs(x); }

extern "C" T mwnrm2_(const int *N, const mf::complex64x2 *x, const int *INCX)
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
        const T *p = reinterpret_cast<const T *>(x + ix);
        for (int c = 0; c < 2; ++c) {
            T ax = dd_abs(p[c]);
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
        }
        ix += incx;
    }

    T scl, sumsq;
    if (abig > 0.0) {
        if (amed > 0.0 || amed > maxN || amed != amed)
            abig = abig + (amed * bsbig) * bsbig;
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
