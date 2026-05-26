/* qxnrm2 — kind16: ||X||₂ for complex X (real result).
 *
 * Blue's algorithm: single pass, three magnitude-bucketed accumulators.
 * Same as qnrm2 but processes Re/Im as two values per element.
 */
#include <quadmath.h>
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __complex128 T;
typedef __float128 R;

static R btsml, btbig, bssml, bsbig, maxN;
static int blue_inited = 0;

static __attribute__((cold)) void blue_init(void)
{
    btsml = scalbnq(1.0Q, -8191);
    btbig = scalbnq(1.0Q,  8136);
    bssml = scalbnq(1.0Q,  8247);
    bsbig = scalbnq(1.0Q, -8248);
    maxN  = FLT128_MAX;
    blue_inited = 1;
}

static inline R sq(R x) { return x * x; }

/* Bucket one scalar component into (abig, amed, asml). */
static inline void blue_bucket(R ax, R *abig, R *amed, R *asml, int *notbig)
{
    if (ax > btbig) {
        *abig += sq(ax * bsbig);
        *notbig = 0;
    } else if (ax < btsml) {
        if (*notbig) *asml += sq(ax * bssml);
    } else {
        *amed += sq(ax);
    }
}

R qxnrm2_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n <= 0) return 0.0Q;
    if (!blue_inited) blue_init();

    R abig = 0.0Q, amed = 0.0Q, asml = 0.0Q;
    int notbig = 1;
    int ix = (incx < 0) ? -(n - 1) * incx : 0;
    for (int i = 0; i < n; ++i) {
        blue_bucket(fabsq(__real__ x[ix]), &abig, &amed, &asml, &notbig);
        blue_bucket(fabsq(__imag__ x[ix]), &abig, &amed, &asml, &notbig);
        ix += incx;
    }

    R scl, sumsq;
    if (abig > 0.0Q) {
        if (amed > 0.0Q || amed > maxN || amed != amed) {
            abig += (amed * bsbig) * bsbig;
        }
        scl   = 1.0Q / bsbig;
        sumsq = abig;
    } else if (asml > 0.0Q) {
        if (amed > 0.0Q || amed > maxN || amed != amed) {
            R sa = sqrtq(amed);
            R ss = sqrtq(asml) / bssml;
            R ymin, ymax;
            if (ss > sa) { ymin = sa; ymax = ss; }
            else         { ymin = ss; ymax = sa; }
            scl   = 1.0Q;
            sumsq = sq(ymax) * (1.0Q + sq(ymin / ymax));
        } else {
            scl   = 1.0Q / bssml;
            sumsq = asml;
        }
    } else {
        scl   = 1.0Q;
        sumsq = amed;
    }
    return scl * sqrtq(sumsq);
}
