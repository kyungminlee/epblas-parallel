/* eynrm2 — kind10: ||X||₂ for complex X (real result).
 *
 * Blue's algorithm — single pass, three buckets, two values per element
 * (real and imaginary parts). Matches migrated reference.
 */
#include <math.h>
#include <float.h>
typedef _Complex long double T;
typedef long double R;

static R btsml, btbig, bssml, bsbig, maxN;
static int blue_inited = 0;

static __attribute__((cold)) void blue_init(void)
{
    btsml = ldexpl(1.0L, -8191);
    btbig = ldexpl(1.0L,  8160);
    bssml = ldexpl(1.0L,  8222);
    bsbig = ldexpl(1.0L, -8224);
    maxN  = LDBL_MAX;
    blue_inited = 1;
}

static inline R sq(R x) { return x * x; }
static inline R ldabs(R x) { return x < 0 ? -x : x; }

R eynrm2_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n <= 0) return 0.0L;
    if (!blue_inited) blue_init();

    R abig = 0.0L, amed = 0.0L, asml = 0.0L;
    int notbig = 1;
    int ix = (incx < 0) ? -(n - 1) * incx : 0;
    /* Hot loop transcribed from the epblas-openblas port: a complex element
     * is two reals read through `p[c]`, and the three-way Blue bucketing is
     * inlined here with the accumulators as plain locals.  This exact shape
     * is what lets gcc keep the dominant medium-magnitude accumulator `amed`
     * (and the `btsml` threshold) on the x87 register stack — only the
     * rare-path constant spills.  An earlier by-pointer/macro form spilled
     * `amed` instead, costing a per-element 80-bit load/store (~1.8x slower).
     * Same ops in the same order, so bit-identical to the reference. */
    for (int i = 0; i < n; ++i) {
        const R *p = (const R *)&x[ix];
        for (int c = 0; c < 2; ++c) {
            R ax = ldabs(p[c]);
            if (ax > btbig) {
                R t = ax * bsbig;
                abig += t * t;
                notbig = 0;
            } else if (ax < btsml) {
                if (notbig) {
                    R t = ax * bssml;
                    asml += t * t;
                }
            } else {
                amed += ax * ax;
            }
        }
        ix += incx;
    }

    R scl, sumsq;
    if (abig > 0.0L) {
        if (amed > 0.0L || amed > maxN || amed != amed) {
            abig += (amed * bsbig) * bsbig;
        }
        scl   = 1.0L / bsbig;
        sumsq = abig;
    } else if (asml > 0.0L) {
        if (amed > 0.0L || amed > maxN || amed != amed) {
            R sa = sqrtl(amed);
            R ss = sqrtl(asml) / bssml;
            R ymin, ymax;
            if (ss > sa) { ymin = sa; ymax = ss; }
            else         { ymin = ss; ymax = sa; }
            scl   = 1.0L;
            sumsq = sq(ymax) * (1.0L + sq(ymin / ymax));
        } else {
            scl   = 1.0L / bssml;
            sumsq = asml;
        }
    } else {
        scl   = 1.0L;
        sumsq = amed;
    }
    return scl * sqrtl(sumsq);
}
