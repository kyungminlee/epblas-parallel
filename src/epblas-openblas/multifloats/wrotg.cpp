/*
 * wrotg — multifloats DD port of LAPACK 3.12.1 zrotg (Anderson 2017 scaling).
 *
 * Generates plane rotation with real cosine and complex sine:
 *   [ c          s ] [a] = [r]
 *   [-conjg(s)   c ] [b]   [0]
 *
 * Inputs/outputs:
 *   a (in/out): on entry the input a; on exit r.
 *   b (in):    the scalar b.
 *   c (out):   real cosine.
 *   s (out):   complex sine.
 *
 * Reference: blas/src/yrotg.f90.
 *
 * DD adaptation: the double-double type rides on the binary64 exponent
 * range, so Anderson's safe-scaling constants use DBL_* (NOT the host
 * fp80 LDBL_*).
 */
#include <complex>
#include <cfloat>
#include <multifloats.h>
#include <multifloats/float64x2.h>

namespace mf = multifloats;
using C = std::complex<mf::float64x2>;
using T = mf::float64x2;

static T safmin, safmax, rtmin;
static int safscale_initialized = 0;

static void safscale_init(void)
{
    int min_exp = DBL_MIN_EXP;
    int max_exp = DBL_MAX_EXP;
    int s_min = (min_exp - 1) > (1 - max_exp) ? (min_exp - 1) : (1 - max_exp);
    int s_max = (1 - min_exp) > (max_exp - 1) ? (1 - min_exp) : (max_exp - 1);
    safmin = mf::ldexpdd(T(1.0), s_min);
    safmax = mf::ldexpdd(T(1.0), s_max);
    rtmin  = mf::sqrt(safmin);
    safscale_initialized = 1;
}

static inline T ldabs(T x) { return x < 0.0 ? -x : x; }
static inline T lmax(T a, T b) { return a > b ? a : b; }
static inline T lmin(T a, T b) { return a < b ? a : b; }
static inline T abssq(C t) {
    return t.real()*t.real() + t.imag()*t.imag();
}
static inline T re(C z) { return z.real(); }
static inline T im(C z) { return z.imag(); }
static inline int ceq0(C z) { return re(z) == 0.0 && im(z) == 0.0; }

extern "C" void wrotg_(C *Ain, C *Bin, T *Cout, C *Sout)
{
    if (!safscale_initialized) safscale_init();
    C f = *Ain;
    C g = *Bin;
    C r, s;
    T c;

    if (ceq0(g)) {
        c = 1.0;
        s = C(0.0, 0.0);
        r = f;
    } else if (ceq0(f)) {
        c = 0.0;
        T g1;
        T gr = re(g), gi = im(g);
        T agi = ldabs(gi), agr = ldabs(gr);
        if (gr == 0.0) {
            r = agi;
            s = std::conj(g) / r;
        } else if (gi == 0.0) {
            r = agr;
            s = std::conj(g) / r;
        } else {
            g1 = lmax(agr, agi);
            T rtmax = mf::sqrt(safmax / 2.0);
            if (g1 > rtmin && g1 < rtmax) {
                T g2 = abssq(g);
                T d = mf::sqrt(g2);
                s = std::conj(g) / d;
                r = d;
            } else {
                T u = lmin(safmax, lmax(safmin, g1));
                C gs = g / u;
                T g2 = abssq(gs);
                T d = mf::sqrt(g2);
                s = std::conj(gs) / d;
                r = d * u;
            }
        }
    } else {
        T f1 = lmax(ldabs(re(f)), ldabs(im(f)));
        T g1 = lmax(ldabs(re(g)), ldabs(im(g)));
        T rtmax = mf::sqrt(safmax / 4.0);
        if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
            T f2 = abssq(f);
            T g2 = abssq(g);
            T h2 = f2 + g2;
            if (f2 >= h2 * safmin) {
                c = mf::sqrt(f2 / h2);
                r = f / c;
                T rtmax2 = rtmax * 2.0;
                if (f2 > rtmin && h2 < rtmax2) {
                    s = std::conj(g) * (f / mf::sqrt(f2 * h2));
                } else {
                    s = std::conj(g) * (r / h2);
                }
            } else {
                T d = mf::sqrt(f2 * h2);
                c = f2 / d;
                if (c >= safmin) {
                    r = f / c;
                } else {
                    r = f * (h2 / d);
                }
                s = std::conj(g) * (f / d);
            }
        } else {
            T u = lmin(safmax, lmax(safmin, lmax(f1, g1)));
            C gs = g / u;
            T g2 = abssq(gs);
            T w;
            C fs;
            T f2, h2;
            if (f1 / u < rtmin) {
                T v = lmin(safmax, lmax(safmin, f1));
                w = v / u;
                fs = f / v;
                f2 = abssq(fs);
                h2 = f2 * w * w + g2;
            } else {
                w = 1.0;
                fs = f / u;
                f2 = abssq(fs);
                h2 = f2 + g2;
            }
            if (f2 >= h2 * safmin) {
                c = mf::sqrt(f2 / h2);
                r = fs / c;
                T rtmax2 = rtmax * 2.0;
                if (f2 > rtmin && h2 < rtmax2) {
                    s = std::conj(gs) * (fs / mf::sqrt(f2 * h2));
                } else {
                    s = std::conj(gs) * (r / h2);
                }
            } else {
                T d = mf::sqrt(f2 * h2);
                c = f2 / d;
                if (c >= safmin) {
                    r = fs / c;
                } else {
                    r = fs * (h2 / d);
                }
                s = std::conj(gs) * (fs / d);
            }
            c = c * w;
            r = r * u;
        }
    }
    *Ain = r;
    *Cout = c;
    *Sout = s;
}
