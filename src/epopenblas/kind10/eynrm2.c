/*
 * eynrm2 — kind10 port of LAPACK 3.12.1 dznrm2 (Blue's algorithm).
 *
 * Real Euclidean norm of a complex vector.  Each complex element
 * contributes |Re|^2 + |Im|^2 to the sum; we apply Blue's
 * three-accumulator scaling to Re and Im independently.
 *
 * Reference: blas/src/eynrm2.f90.
 */
#include <stddef.h>
#include <math.h>
#include <float.h>

typedef _Complex long double C;
typedef long double T;

static T btsml, btbig, bssml, bsbig, maxN;
static int blue_initialized = 0;

static void blue_init(void)
{
    int min_exp = LDBL_MIN_EXP;
    int max_exp = LDBL_MAX_EXP;
    int dig     = LDBL_MANT_DIG;
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

T eynrm2_(const int *N, const C *x, const int *INCX)
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
        const T *p = (const T *)(x + ix);
        for (int c = 0; c < 2; ++c) {
            T ax = ldabs(p[c]);
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
    if (abig > 0.0L) {
        if (amed > 0.0L || amed > maxN || amed != amed)
            abig = abig + (amed * bsbig) * bsbig;
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
