/* qnrm2 — kind16 real: returns ||X||₂.
 *
 * Blue's algorithm (Anderson 2017, "Safe Scaling in the Level 1 BLAS";
 * Blue 1978, "A Portable Fortran Program to Find the Euclidean Norm of
 * a Vector"). Same algorithm as the migrated reference — single pass
 * over X with three accumulators (abig/amed/asml) bucketed by magnitude
 * to avoid both overflow and underflow.
 *
 * Previous overlay used the naive two-pass scaled algorithm. For
 * __float128 every element is read via __addtf3/__multf3/__lttf2 — a
 * second pass doubled the soft-float call count.
 */
#include <quadmath.h>
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __float128 T;

/* Blue's scaling constants — radix² powers chosen so that
 *   x in [btsml, btbig]  : accumulate x² directly (amed)
 *   x > btbig            : scale down by bsbig before squaring
 *   x < btsml            : scale up   by bssml before squaring
 * The constants depend only on the floating-point format, so cache
 * after first call. For __float128 (IEEE binary128):
 *   minexp = -16381, maxexp = 16384, digits = 113 (binary)
 */
static T btsml, btbig, bssml, bsbig, maxN;
static int blue_inited = 0;

static __attribute__((cold)) void blue_init(void)
{
    /* 2^k via repeated squaring — avoids ldexpq function call.
     * For the small exponents we need (max |k| ≈ 16384), this is fine. */
    /* Constants from Fortran:
     *   btsml = 2^ceil((minexp-1)/2)            = 2^-8191
     *   btbig = 2^floor((maxexp-digits+1)/2)    = 2^8136
     *   bssml = 2^(-floor((minexp-digits)/2))   = 2^8247
     *   bsbig = 2^(-ceil((maxexp+digits-1)/2))  = 2^-8248
     */
    btsml = scalbnq(1.0Q, -8191);
    btbig = scalbnq(1.0Q,  8136);
    bssml = scalbnq(1.0Q,  8247);
    bsbig = scalbnq(1.0Q, -8248);
    maxN  = FLT128_MAX;
    blue_inited = 1;
}

static inline T sq(T x) { return x * x; }

T qnrm2_(const int *n_, const T *x, const int *incx_)
{
    const int n = *n_, incx = *incx_;
    if (n <= 0) return 0.0Q;
    if (!blue_inited) blue_init();

    T abig = 0.0Q, amed = 0.0Q, asml = 0.0Q;
    int notbig = 1;
    int ix = (incx < 0) ? -(n - 1) * incx : 0;
    for (int i = 0; i < n; ++i) {
        T ax = fabsq(x[ix]);
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
    if (abig > 0.0Q) {
        if (amed > 0.0Q || amed > maxN || amed != amed) {
            abig += (amed * bsbig) * bsbig;
        }
        scl   = 1.0Q / bsbig;
        sumsq = abig;
    } else if (asml > 0.0Q) {
        if (amed > 0.0Q || amed > maxN || amed != amed) {
            T sa = sqrtq(amed);
            T ss = sqrtq(asml) / bssml;
            T ymin, ymax;
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
