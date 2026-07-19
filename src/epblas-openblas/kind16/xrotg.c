/*
 * xrotg — kind16 port of LAPACK 3.12.1 zrotg (Anderson 2017 scaling).
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
 * Reference: blas/src/xrotg.f90.
 */
#include <math.h>
#include <quadmath.h>
#include <float.h>

typedef __complex128 C;
typedef __float128 T;

static T safmin, safmax, rtmin;
static int safscale_initialized = 0;

static void safscale_init(void)
{
    int min_exp = LDBL_MIN_EXP;
    int max_exp = LDBL_MAX_EXP;
    int s_min = (min_exp - 1) > (1 - max_exp) ? (min_exp - 1) : (1 - max_exp);
    int s_max = (1 - min_exp) > (max_exp - 1) ? (1 - min_exp) : (max_exp - 1);
    safmin = ldexpq(1.0Q, s_min);
    safmax = ldexpq(1.0Q, s_max);
    rtmin  = sqrtq(safmin);
    safscale_initialized = 1;
}

static inline T q_abs(T x) { return __builtin_fabsf128(x); }
static inline T lmax(T a, T b) { return a > b ? a : b; }
static inline T lmin(T a, T b) { return a < b ? a : b; }
static inline T abssq(C t) {
    const T *p = (const T *)&t;
    return p[0]*p[0] + p[1]*p[1];
}
static inline C cconjq(C z) {
    C r; __real__ r = __real__ z; __imag__ r = -__imag__ z; return r;
}
static inline T re(C z) { return __real__ z; }
static inline T im(C z) { return __imag__ z; }
static inline int ceq0(C z) { return re(z) == 0.0Q && im(z) == 0.0Q; }

void xrotg_(C *Ain, C *Bin, T *Cout, C *Sout)
{
    if (!safscale_initialized) safscale_init();
    C f = *Ain;
    C g = *Bin;
    C r, s;
    T c;

    if (ceq0(g)) {
        c = 1.0Q;
        s = 0.0Q;
        r = f;
    } else if (ceq0(f)) {
        c = 0.0Q;
        T g1;
        T gr = re(g), gi = im(g);
        T agi = q_abs(gi), agr = q_abs(gr);
        if (gr == 0.0Q) {
            r = agi;
            s = cconjq(g) / r;
        } else if (gi == 0.0Q) {
            r = agr;
            s = cconjq(g) / r;
        } else {
            g1 = lmax(agr, agi);
            T rtmax = sqrtq(safmax / 2.0Q);
            if (g1 > rtmin && g1 < rtmax) {
                T g2 = abssq(g);
                T d = sqrtq(g2);
                s = cconjq(g) / d;
                r = d;
            } else {
                T u = lmin(safmax, lmax(safmin, g1));
                C gs = g / u;
                T g2 = abssq(gs);
                T d = sqrtq(g2);
                s = cconjq(gs) / d;
                r = d * u;
            }
        }
    } else {
        T f1 = lmax(q_abs(re(f)), q_abs(im(f)));
        T g1 = lmax(q_abs(re(g)), q_abs(im(g)));
        T rtmax = sqrtq(safmax / 4.0Q);
        if (f1 > rtmin && f1 < rtmax && g1 > rtmin && g1 < rtmax) {
            T f2 = abssq(f);
            T g2 = abssq(g);
            T h2 = f2 + g2;
            if (f2 >= h2 * safmin) {
                c = sqrtq(f2 / h2);
                r = f / c;
                T rtmax2 = rtmax * 2.0Q;
                if (f2 > rtmin && h2 < rtmax2) {
                    s = cconjq(g) * (f / sqrtq(f2 * h2));
                } else {
                    s = cconjq(g) * (r / h2);
                }
            } else {
                T d = sqrtq(f2 * h2);
                c = f2 / d;
                if (c >= safmin) {
                    r = f / c;
                } else {
                    r = f * (h2 / d);
                }
                s = cconjq(g) * (f / d);
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
                w = 1.0Q;
                fs = f / u;
                f2 = abssq(fs);
                h2 = f2 + g2;
            }
            if (f2 >= h2 * safmin) {
                c = sqrtq(f2 / h2);
                r = fs / c;
                T rtmax2 = rtmax * 2.0Q;
                if (f2 > rtmin && h2 < rtmax2) {
                    s = cconjq(gs) * (fs / sqrtq(f2 * h2));
                } else {
                    s = cconjq(gs) * (r / h2);
                }
            } else {
                T d = sqrtq(f2 * h2);
                c = f2 / d;
                if (c >= safmin) {
                    r = fs / c;
                } else {
                    r = fs * (h2 / d);
                }
                s = cconjq(gs) * (fs / d);
            }
            c = c * w;
            r = r * u;
        }
    }
    *Ain = r;
    *Cout = c;
    *Sout = s;
}
