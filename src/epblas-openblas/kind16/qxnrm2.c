/*
 * qxnrm2 — kind16 port of LAPACK 3.12.1 dznrm2 (Blue's algorithm).
 *
 * Real Euclidean norm of a complex vector.  Each complex element
 * contributes |Re|^2 + |Im|^2 to the sum; we apply Blue's
 * three-accumulator scaling to Re and Im independently.
 *
 * Reference: blas/src/qxnrm2.f90.
 */
#include <stddef.h>
#include <quadmath.h>
#include <math.h>
#include <float.h>

typedef __complex128 C;
typedef __float128 T;

static T btsml, btbig, bssml, bsbig, maxN;
static int blue_initialized = 0;

static void blue_init(void)
{
    int min_exp = LDBL_MIN_EXP;
    int max_exp = LDBL_MAX_EXP;
    int dig     = LDBL_MANT_DIG;
    #define CEIL2(x)  ( ((x) >= 0) ? ((x) + 1) / 2 : -((-(x)) / 2) )
    #define FLOOR2(x) ( ((x) >= 0) ? (x) / 2 : -(((-(x)) + 1) / 2) )
    btsml = ldexpq(1.0Q,  CEIL2(min_exp - 1));
    btbig = ldexpq(1.0Q,  FLOOR2(max_exp - dig + 1));
    bssml = ldexpq(1.0Q, -FLOOR2(min_exp - dig));
    bsbig = ldexpq(1.0Q, -CEIL2(max_exp + dig - 1));
    maxN  = LDBL_MAX;
    #undef CEIL2
    #undef FLOOR2
    blue_initialized = 1;
}

static inline T q_abs(T x) { return __builtin_fabsf128(x); }

T qxnrm2_(const int *N, const C *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    if (n <= 0) return 0.0Q;
    if (!blue_initialized) blue_init();

    T asml = 0.0Q, amed = 0.0Q, abig = 0.0Q;
    int notbig = 1;

    ptrdiff_t ix = 0;
    if (incx < 0) ix = -(n - 1) * incx;

    for (ptrdiff_t i = 0; i < n; ++i) {
        const T *p = (const T *)(x + ix);
        for (int c = 0; c < 2; ++c) {
            T ax = q_abs(p[c]);
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
    if (abig > 0.0Q) {
        if (amed > 0.0Q || amed > maxN || amed != amed)
            abig = abig + (amed * bsbig) * bsbig;
        scl = 1.0Q / bsbig;
        sumsq = abig;
    } else if (asml > 0.0Q) {
        if (amed > 0.0Q || amed > maxN || amed != amed) {
            T amed_s = sqrtq(amed);
            T asml_s = sqrtq(asml) / bssml;
            T ymin, ymax;
            if (asml_s > amed_s) { ymin = amed_s; ymax = asml_s; }
            else                 { ymin = asml_s; ymax = amed_s; }
            scl = 1.0Q;
            sumsq = ymax * ymax * (1.0Q + (ymin/ymax) * (ymin/ymax));
        } else {
            scl = 1.0Q / bssml;
            sumsq = asml;
        }
    } else {
        scl = 1.0Q;
        sumsq = amed;
    }
    return scl * sqrtq(sumsq);
}
