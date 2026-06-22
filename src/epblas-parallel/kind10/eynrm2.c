/* eynrm2 — kind10: ||X||₂ for complex X (real result).
 *
 * Blue's algorithm — single pass, three buckets, two values per element
 * (real and imaginary parts). Matches migrated reference.
 */
#include <math.h>
#include <float.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define EYNRM2_OMP_MIN  10000   /* below this, run the serial path (n in elements) */
#define EYNRM2_MAX_CPUS 64
#endif
#include "../common/epblas_facade.h"
typedef _Complex long double T;
typedef long double R;

static R btsml, btbig, bssml, bsbig, maxN;
static ptrdiff_t blue_inited = 0;

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
static void eynrm2_bucket(ptrdiff_t nel, const T *x, R *abig_, R *amed_, R *asml_)
{
    R abig = 0.0L, amed = 0.0L, asml = 0.0L;
    ptrdiff_t notbig = 1;
    for (ptrdiff_t i = 0; i < nel; ++i) {
        const R *p = (const R *)&x[i];
        for (ptrdiff_t c = 0; c < 2; ++c) {
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
    }
    *abig_ = abig; *amed_ = amed; *asml_ = asml;
}

/* Threaded reduction for large unit-stride X. Each thread buckets its own chunk
 * of complex elements; partial sums combine exactly. Reduction order differs
 * from serial (not bit-identical), but within fuzz tolerance. */
__attribute__((noinline)) static int eynrm2_omp(ptrdiff_t n, const T *x, R *out)
{
    if (n <= EYNRM2_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > EYNRM2_MAX_CPUS) nthreads = EYNRM2_MAX_CPUS;
    R pbig[EYNRM2_MAX_CPUS] = {0}, pmed[EYNRM2_MAX_CPUS] = {0}, psml[EYNRM2_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) eynrm2_bucket(hi - lo, x + lo, &pbig[tid], &pmed[tid], &psml[tid]);
    }
    R abig = 0.0L, amed = 0.0L, asml = 0.0L;
    for (int i = 0; i < nthreads; ++i) { abig += pbig[i]; amed += pmed[i]; asml += psml[i]; }
    *out = eynrm2_finalize(abig, amed, asml);
    return 1;
}
#endif

static R eynrm2_core(ptrdiff_t n, const T *x, ptrdiff_t incx)
{
    if (n <= 0) return 0.0L;
    if (!blue_inited) blue_init();

#ifdef _OPENMP
    if (incx == 1) {
        R r;
        if (eynrm2_omp(n, x, &r)) return r;
    }
#endif

    R abig = 0.0L, amed = 0.0L, asml = 0.0L;
    ptrdiff_t notbig = 1;
    ptrdiff_t ix = (incx < 0) ? -(n - 1) * incx : 0;
    /* Hot loop transcribed from the epblas-openblas port: a complex element
     * is two reals read through `p[c]`, and the three-way Blue bucketing is
     * inlined here with the accumulators as plain locals.  This exact shape
     * is what lets gcc keep the dominant medium-magnitude accumulator `amed`
     * (and the `btsml` threshold) on the x87 register stack — only the
     * rare-path constant spills.  An earlier by-pointer/macro form spilled
     * `amed` instead, costing a per-element 80-bit load/store (~1.8x slower).
     * Same ops in the same order, so bit-identical to the reference. */
    for (ptrdiff_t i = 0; i < n; ++i) {
        const R *p = (const R *)&x[ix];
        for (ptrdiff_t c = 0; c < 2; ++c) {
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

    return eynrm2_finalize(abig, amed, asml);
}

EPBLAS_FACADE_ASUM(eynrm2, R, T)
