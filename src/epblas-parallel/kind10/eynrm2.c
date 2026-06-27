/* eynrm2 — kind10: ||X||₂ for complex X (real result).
 *
 * Blue's algorithm — single pass, three buckets, two values per element
 * (real and imaginary parts). Matches migrated reference.
 */
#include <math.h>
#include <stdbool.h>
#include <float.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define EYNRM2_OMP_MIN  10000   /* below this, run the serial path (n in elements) */
#define EYNRM2_MAX_CPUS 64
#endif
#include "../common/epblas_facade.h"
typedef _Complex long double TC;
typedef long double R;

static R btsml, btbig, bssml, bsbig, maxN;
static R btsml2, btbig2;   /* squared thresholds — see hot loop */
static ptrdiff_t blue_inited = 0;

static __attribute__((cold)) void blue_init(void)
{
    btsml = ldexpl(1.0L, -8191);
    btbig = ldexpl(1.0L,  8160);
    bssml = ldexpl(1.0L,  8222);
    bsbig = ldexpl(1.0L, -8224);
    maxN  = LDBL_MAX;
    /* Squared bucket thresholds. The hot loop squares each component anyway
     * (for amed), so it routes on the square vs these instead of on |ax| vs
     * the linear thresholds — eliminating the per-element abs (fld-dup + fabs)
     * and the x87 stack-shuffle (fxch/fstp) it forced. Exact powers of two, so
     * btbig2/btsml2 are representable (2^16320 / 2^-16382, both in range). */
    btbig2 = btbig * btbig;
    btsml2 = btsml * btsml;
    blue_inited = 1;
}

static inline R sq(R x) { return x * x; }
static inline R ldabs(R x) { return __builtin_fabsl(x); }

/* Combine the three magnitude buckets into the final norm (Anderson 2017). */
static R eynrm2_finalize(R abig, R amed, R asml)
{
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

#ifdef _OPENMP
/* Bucket-accumulate a unit-stride chunk of `nel` complex elements (2 reals each)
 * into (abig, amed, asml). Same register-resident hot-loop shape as the serial
 * path (see comment below). `notbig` is chunk-local & exact: asml is consumed by
 * the finalizer only when the GLOBAL abig==0. */
static void eynrm2_bucket(ptrdiff_t nel, const TC *x, R *abig_, R *amed_, R *asml_)
{
    R abig = 0.0L, amed = 0.0L, asml = 0.0L;
    bool notbig = 1;
    for (ptrdiff_t i = 0; i < nel; ++i) {
        const R *p = (const R *)&x[i];
        for (ptrdiff_t c = 0; c < 2; ++c) {
            R v = p[c];
            R s = v * v;                    /* Inf if huge, 0 if tiny — caught below */
            if (s > btbig2) {
                R t = v * bsbig;
                abig += t * t;
                notbig = 0;
            } else if (s < btsml2) {
                if (notbig) {
                    R t = v * bssml;
                    asml += t * t;
                }
            } else {
                amed += s;                  /* medium: square is exact, reuse it */
            }
        }
    }
    *abig_ = abig; *amed_ = amed; *asml_ = asml;
}

/* Threaded reduction for large unit-stride X. Each thread buckets its own chunk
 * of complex elements; partial sums combine exactly. Reduction order differs
 * from serial (not bit-identical), but within fuzz tolerance. */
__attribute__((noinline)) static bool eynrm2_omp(ptrdiff_t n, const TC *x, R *out)
{
    if (n <= EYNRM2_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > EYNRM2_MAX_CPUS) nthreads = EYNRM2_MAX_CPUS;
    R pbig[EYNRM2_MAX_CPUS] = {0}, pmed[EYNRM2_MAX_CPUS] = {0}, psml[EYNRM2_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) eynrm2_bucket(hi - lo, x + lo, &pbig[tid], &pmed[tid], &psml[tid]);
    }
    R abig = 0.0L, amed = 0.0L, asml = 0.0L;
    for (ptrdiff_t i = 0; i < nthreads; ++i) { abig += pbig[i]; amed += pmed[i]; asml += psml[i]; }
    *out = eynrm2_finalize(abig, amed, asml);
    return 1;
}
#endif

static R eynrm2_core(ptrdiff_t n, const TC *x, ptrdiff_t incx)
{
    if (n <= 0) return 0.0L;
    if (!blue_inited) blue_init();

#ifdef _OPENMP
    if (incx == 1) {
        R r;
        if (eynrm2_omp(n, x, &r)) return r;
    }
#endif

    /* Fast path — no per-element thresholds at all. Accumulate the raw sum of
     * squares across four independent chains (two complex elements per
     * iteration: Re/Im × even/odd) so the fp80 fadd latencies (~3-5 cyc)
     * overlap instead of serializing through one accumulator — the only ILP
     * lever on x87 (no SIMD). With the thresholds gone there is stack room for
     * the four chains (an earlier split *with* thresholds resident spilled).
     *
     * If the total is finite and nonzero then no component overflowed when
     * squared and not every component underflowed to zero, so the naive norm
     * is accurate and we are done — the common case, with ZERO compares in the
     * loop body. A non-finite-or-zero total (data near LDBL_MAX, or all tiny,
     * or Inf/NaN inputs) is rare and falls through to Blue's three-bucket
     * scaling below, which pays for its threshold work only when it matters.
     * Reorders the sum vs the reference (4 chains) — within fuzz tolerance. */
    R c0 = 0.0L, c1 = 0.0L, c2 = 0.0L, c3 = 0.0L;
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
    if (s > 0.0L && s <= maxN)        /* finite, nonzero, no overflow */
        return sqrtl(s);

    /* Robust fallback: Blue's three-bucket scaling. Reached only for huge /
     * all-tiny / Inf / NaN data. Routes on the square v*v against the squared
     * thresholds (the abs is unnecessary once squared); the big/small buckets
     * rescale from the raw v before squaring, preserving the overflow guard. */
    R abig = 0.0L, amed = 0.0L, asml = 0.0L;
    bool notbig = 1;
    ptrdiff_t ix = (incx < 0) ? -(n - 1) * incx : 0;
    for (ptrdiff_t k = 0; k < n; ++k) {
        const R *p = (const R *)&x[ix];
        for (ptrdiff_t c = 0; c < 2; ++c) {
            R v = p[c];
            R s2 = v * v;
            if (s2 > btbig2) {
                R t = v * bsbig;
                abig += t * t;
                notbig = 0;
            } else if (s2 < btsml2) {
                if (notbig) {
                    R t = v * bssml;
                    asml += t * t;
                }
            } else {
                amed += s2;
            }
        }
        ix += incx;
    }

    return eynrm2_finalize(abig, amed, asml);
}

EPBLAS_FACADE_ASUM(eynrm2, R, TC)
