/* enrm2 — kind10 real: returns ||X||₂ = sqrt(Σ X·X).
 *
 * Blue's algorithm (Anderson 2017 / Blue 1978). Same algorithm as the
 * migrated reference: single pass over X with three magnitude-bucketed
 * accumulators (abig/amed/asml). The naive two-pass scaled version
 * touched X twice and lost ~12% to the migrated reference.
 *
 * For x87 long double (REAL(KIND=10)):
 *   minexp = -16381, maxexp = 16384, digits = 64 (binary)
 */
#include <math.h>
#include <float.h>
typedef long double T;

static T btsml, btbig, bssml, bsbig, maxN;
static int blue_inited = 0;

static __attribute__((cold)) void blue_init(void)
{
    /* Constants from Fortran:
     *   btsml = 2^ceil((minexp-1)/2)            = 2^-8191
     *   btbig = 2^floor((maxexp-digits+1)/2)    = 2^floor(16321/2) = 2^8160
     *   bssml = 2^(-floor((minexp-digits)/2))   = 2^(-floor(-8222))  = 2^8222
     *   bsbig = 2^(-ceil((maxexp+digits-1)/2))  = 2^(-ceil(8223))   = 2^-8224
     */
    btsml = ldexpl(1.0L, -8191);
    btbig = ldexpl(1.0L,  8160);
    bssml = ldexpl(1.0L,  8222);
    bsbig = ldexpl(1.0L, -8224);
    maxN  = LDBL_MAX;
    blue_inited = 1;
}

static inline T sq(T x) { return x * x; }

T enrm2_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n <= 0) return 0.0L;
    if (!blue_inited) blue_init();

    T abig = 0.0L, amed = 0.0L, asml = 0.0L;
    int notbig = 1;
    int ix = (incx < 0) ? -(n - 1) * incx : 0;
    for (int i = 0; i < n; ++i) {
        T ax = fabsl(x[ix]);
        if (ax > btbig) {
            abig += sq(ax * bsbig);
            notbig = 0;
        } else if (ax < btsml) {
            if (notbig) asml += sq(ax * bssml);
        } else {
            amed += sq(ax);
        }
        ix += incx;
    }

    T scl, sumsq;
    if (abig > 0.0L) {
        if (amed > 0.0L || amed > maxN || amed != amed) {
            abig += (amed * bsbig) * bsbig;
        }
        scl   = 1.0L / bsbig;
        sumsq = abig;
    } else if (asml > 0.0L) {
        if (amed > 0.0L || amed > maxN || amed != amed) {
            T sa = sqrtl(amed);
            T ss = sqrtl(asml) / bssml;
            T ymin, ymax;
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
