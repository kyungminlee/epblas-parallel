/* qxnrm2 — kind16: ||X||₂ for complex X (real result).
 *
 * Blue's algorithm: single pass, three magnitude-bucketed accumulators.
 * Same as qnrm2 but processes Re/Im as two values per element.
 */
#include <stddef.h>
#include <stdbool.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __complex128 TC;
typedef __float128 R;

static R btsml, btbig, bssml, bsbig, maxN;
static R btsml2, btbig2;   /* squared thresholds — see blue_bucket / fast path */
static bool blue_inited = 0;

static __attribute__((cold)) void blue_init(void)
{
    btsml = scalbnq(1.0Q, -8191);
    btbig = scalbnq(1.0Q,  8136);
    bssml = scalbnq(1.0Q,  8247);
    bsbig = scalbnq(1.0Q, -8248);
    maxN  = FLT128_MAX;
    /* Squared bucket thresholds. The buckets square each component anyway, so
     * route on v*v vs these instead of |v| vs the linear thresholds — dropping
     * the per-element fabsq. For quad this matters more than for fp80: the
     * comparisons are libquadmath calls, so fewer of them is a real win. Exact
     * powers of two (2^16272 / 2^-16382), both in __float128 range. */
    btbig2 = btbig * btbig;
    btsml2 = btsml * btsml;
    blue_inited = 1;
}

static inline R sq(R x) { return x * x; }

/* Bucket one scalar component into (abig, amed, asml). Routes on the square
 * s = v*v (Inf if v overflows on square → big bucket; 0 if it underflows →
 * small bucket), so no abs is needed; the big/small buckets rescale from the
 * raw v before squaring (sign irrelevant once squared). */
static inline void blue_bucket(R v, R *abig, R *amed, R *asml, bool *notbig)
{
    R s = v * v;
    if (s > btbig2) {
        *abig += sq(v * bsbig);
        *notbig = 0;
    } else if (s < btsml2) {
        if (*notbig) *asml += sq(v * bssml);
    } else {
        *amed += s;
    }
}

/* Combine the three magnitude buckets into the final norm (Anderson 2017). */
static R qxnrm2_finalize(R abig, R amed, R asml)
{
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

#ifdef _OPENMP
/* Bucket-accumulate a unit-stride chunk (Re+Im per element) into
 * (abig, amed, asml). The notbig flag is chunk-local: asml is only consumed by
 * the finalize when the GLOBAL abig==0 (every chunk kept notbig==1), so this is
 * exact. */
static void qxnrm2_bucket(ptrdiff_t n, const TC *x, R *abig_, R *amed_, R *asml_)
{
    R abig = 0.0Q, amed = 0.0Q, asml = 0.0Q;
    bool notbig = 1;
    for (ptrdiff_t i = 0; i < n; ++i) {
        blue_bucket(__real__ x[i], &abig, &amed, &asml, &notbig);
        blue_bucket(__imag__ x[i], &abig, &amed, &asml, &notbig);
    }
    *abig_ = abig; *amed_ = amed; *asml_ = asml;
}

/* Threaded reduction for large unit-stride X (see qnrm2 for the rationale). */
#define QXNRM2_OMP_MIN 128
#define QXNRM2_MAX_CPUS 64
__attribute__((noinline)) static bool qxnrm2_omp(ptrdiff_t n, const TC *x, R *out)
{
    if (n <= QXNRM2_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > QXNRM2_MAX_CPUS) nthreads = QXNRM2_MAX_CPUS;
    R pbig[QXNRM2_MAX_CPUS] = {0}, pmed[QXNRM2_MAX_CPUS] = {0}, psml[QXNRM2_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) qxnrm2_bucket(hi - lo, x + lo, &pbig[tid], &pmed[tid], &psml[tid]);
    }
    R abig = 0.0Q, amed = 0.0Q, asml = 0.0Q;
    for (ptrdiff_t i = 0; i < nthreads; ++i) { abig += pbig[i]; amed += pmed[i]; asml += psml[i]; }
    *out = qxnrm2_finalize(abig, amed, asml);
    return 1;
}
#endif

static R qxnrm2_core(ptrdiff_t n, const TC *x, ptrdiff_t incx)
{
    if (n <= 0) return 0.0Q;
    if (!blue_inited) blue_init();

#ifdef _OPENMP
    if (incx == 1) {
        R r;
        if (qxnrm2_omp(n, x, &r)) return r;
    }
#endif

    /* Fast path — no per-element thresholds. Accumulate the raw sum of squares
     * across four independent chains (two complex elts/iter, Re/Im × even/odd)
     * to expose ILP to the out-of-order engine across the libquadmath add/mul
     * calls. If the total is finite and nonzero, nothing overflowed when
     * squared and not everything underflowed, so the naive norm is accurate —
     * the common case, with ZERO comparisons (libquadmath calls) in the loop.
     * A non-finite-or-zero total (near-FLT128_MAX / all-tiny / Inf / NaN) is
     * rare and falls through to Blue's bucketing below. Reorders the sum vs the
     * reference (4 chains) — within fuzz tolerance. */
    R c0 = 0.0Q, c1 = 0.0Q, c2 = 0.0Q, c3 = 0.0Q;
    ptrdiff_t i = 0;
    if (incx == 1) {
        ptrdiff_t n2 = n & ~(ptrdiff_t)1;
        for (; i < n2; i += 2) {
            const R *p = (const R *)&x[i];
            c0 += p[0] * p[0];
            c1 += p[1] * p[1];
            c2 += p[2] * p[2];
            c3 += p[3] * p[3];
        }
        for (; i < n; ++i) {
            const R *p = (const R *)&x[i];
            c0 += p[0] * p[0];
            c1 += p[1] * p[1];
        }
    } else {
        ptrdiff_t ix = (incx < 0) ? -(n - 1) * incx : 0;
        for (; i < n; ++i) {
            const R *p = (const R *)&x[ix];
            c0 += p[0] * p[0];
            c1 += p[1] * p[1];
            ix += incx;
        }
    }
    R s = (c0 + c1) + (c2 + c3);
    if (s > 0.0Q && s <= maxN)        /* finite, nonzero, no overflow */
        return sqrtq(s);

    /* Robust fallback: Blue's three-bucket scaling. Reached only for huge /
     * all-tiny / Inf / NaN data. */
    R abig = 0.0Q, amed = 0.0Q, asml = 0.0Q;
    bool notbig = 1;
    ptrdiff_t ix = (incx < 0) ? -(n - 1) * incx : 0;
    for (ptrdiff_t k = 0; k < n; ++k) {
        const R *p = (const R *)&x[ix];
        blue_bucket(p[0], &abig, &amed, &asml, &notbig);
        blue_bucket(p[1], &abig, &amed, &asml, &notbig);
        ix += incx;
    }
    return qxnrm2_finalize(abig, amed, asml);
}

EPBLAS_FACADE_ASUM(qxnrm2, R, TC)
