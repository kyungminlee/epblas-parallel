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
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define ENRM2_OMP_MIN  10000   /* below this, run the serial path */
#define ENRM2_MAX_CPUS 64
#endif
#include "../common/epblas_facade.h"
typedef long double TR;

static TR btsml, btbig, bssml, bsbig, maxN;
static ptrdiff_t blue_inited = 0;

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

static inline TR sq(TR x) { return x * x; }

/* Combine the three magnitude buckets into the final norm (Anderson 2017). */
static TR enrm2_finalize(TR abig, TR amed, TR asml)
{
    TR scl, sumsq;
    if (abig > 0.0L) {
        if (amed > 0.0L || amed > maxN || amed != amed) {
            abig += (amed * bsbig) * bsbig;
        }
        scl   = 1.0L / bsbig;
        sumsq = abig;
    } else if (asml > 0.0L) {
        if (amed > 0.0L || amed > maxN || amed != amed) {
            TR sa = sqrtl(amed);
            TR ss = sqrtl(asml) / bssml;
            TR ymin, ymax;
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
/* Bucket-accumulate a unit-stride chunk into (abig, amed, asml). The `notbig`
 * flag is chunk-local: asml is only consumed by enrm2_finalize when the GLOBAL
 * abig==0 (no big element anywhere → every chunk kept notbig==1), so a
 * per-chunk notbig is exact. */
static void enrm2_bucket(ptrdiff_t n, const TR *x, TR *abig_, TR *amed_, TR *asml_)
{
    TR abig = 0.0L, amed = 0.0L, asml = 0.0L;
    bool notbig = 1;
    for (ptrdiff_t i = 0; i < n; ++i) {
        TR ax = fabsl(x[i]);
        if (ax > btbig) { abig += sq(ax * bsbig); notbig = 0; }
        else if (ax < btsml) { if (notbig) asml += sq(ax * bssml); }
        else amed += sq(ax);
    }
    *abig_ = abig; *amed_ = amed; *asml_ = asml;
}

/* Threaded reduction for large unit-stride X. Each thread buckets its own chunk;
 * the three partial sums combine exactly as analysed above. Reduction order
 * differs from serial (not bit-identical), but within fuzz tolerance. */
__attribute__((noinline)) static bool enrm2_omp(ptrdiff_t n, const TR *x, TR *out)
{
    if (n <= ENRM2_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > ENRM2_MAX_CPUS) nthreads = ENRM2_MAX_CPUS;
    TR pbig[ENRM2_MAX_CPUS] = {0}, pmed[ENRM2_MAX_CPUS] = {0}, psml[ENRM2_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) enrm2_bucket(hi - lo, x + lo, &pbig[tid], &pmed[tid], &psml[tid]);
    }
    TR abig = 0.0L, amed = 0.0L, asml = 0.0L;
    for (ptrdiff_t i = 0; i < nthreads; ++i) { abig += pbig[i]; amed += pmed[i]; asml += psml[i]; }
    *out = enrm2_finalize(abig, amed, asml);
    return 1;
}
#endif

static TR enrm2_core(ptrdiff_t n, const TR *x, ptrdiff_t incx)
{
    if (n <= 0) return 0.0L;
    if (!blue_inited) blue_init();

#ifdef _OPENMP
    if (incx == 1) {
        TR r;
        if (enrm2_omp(n, x, &r)) return r;
    }
#endif

    TR abig = 0.0L, amed = 0.0L, asml = 0.0L;
    bool notbig = 1;
    ptrdiff_t ix = (incx < 0) ? -(n - 1) * incx : 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        TR ax = fabsl(x[ix]);
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

    return enrm2_finalize(abig, amed, asml);
}

EPBLAS_FACADE_ASUM(enrm2, TR, TR)
