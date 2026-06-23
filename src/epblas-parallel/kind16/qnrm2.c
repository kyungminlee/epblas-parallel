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
typedef __float128 TR;

/* Blue's scaling constants — radix² powers chosen so that
 *   x in [btsml, btbig]  : accumulate x² directly (amed)
 *   x > btbig            : scale down by bsbig before squaring
 *   x < btsml            : scale up   by bssml before squaring
 * The constants depend only on the floating-point format, so cache
 * after first call. For __float128 (IEEE binary128):
 *   minexp = -16381, maxexp = 16384, digits = 113 (binary)
 */
static TR btsml, btbig, bssml, bsbig, maxN;
static bool blue_inited = 0;

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

static inline TR sq(TR x) { return x * x; }

/* Combine the three magnitude buckets into the final norm (Anderson 2017). */
static TR qnrm2_finalize(TR abig, TR amed, TR asml)
{
    TR scl, sumsq;
    if (abig > 0.0Q) {
        if (amed > 0.0Q || amed > maxN || amed != amed) {
            abig += (amed * bsbig) * bsbig;
        }
        scl   = 1.0Q / bsbig;
        sumsq = abig;
    } else if (asml > 0.0Q) {
        if (amed > 0.0Q || amed > maxN || amed != amed) {
            TR sa = sqrtq(amed);
            TR ss = sqrtq(asml) / bssml;
            TR ymin, ymax;
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
/* Bucket-accumulate a unit-stride chunk into (abig, amed, asml). The `notbig`
 * flag is chunk-local: asml is only consumed by qnrm2_finalize when the GLOBAL
 * abig==0 (no big element anywhere → every chunk kept notbig==1), so a
 * per-chunk notbig is exact. */
static void qnrm2_bucket(ptrdiff_t n, const TR *x, TR *abig_, TR *amed_, TR *asml_)
{
    TR abig = 0.0Q, amed = 0.0Q, asml = 0.0Q;
    bool notbig = 1;
    for (ptrdiff_t i = 0; i < n; ++i) {
        TR ax = fabsq(x[i]);
        if (ax > btbig) { abig += sq(ax * bsbig); notbig = 0; }
        else if (ax < btsml) { if (notbig) asml += sq(ax * bssml); }
        else amed += sq(ax);
    }
    *abig_ = abig; *amed_ = amed; *asml_ = asml;
}

/* Threaded reduction for large unit-stride X. libquadmath makes every |x| and
 * square a heavy soft-float op, so the reference threads the norm and pulls
 * ~4x ahead at large n; par mirrors it. Each thread buckets its own chunk; the
 * three partial sums combine exactly as analysed above. Reduction order
 * differs from serial (not bit-identical), but within fuzz tolerance. */
#define QNRM2_OMP_MIN 128
#define QNRM2_MAX_CPUS 64
__attribute__((noinline)) static bool qnrm2_omp(ptrdiff_t n, const TR *x, TR *out)
{
    if (n <= QNRM2_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > QNRM2_MAX_CPUS) nthreads = QNRM2_MAX_CPUS;
    TR pbig[QNRM2_MAX_CPUS] = {0}, pmed[QNRM2_MAX_CPUS] = {0}, psml[QNRM2_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) qnrm2_bucket(hi - lo, x + lo, &pbig[tid], &pmed[tid], &psml[tid]);
    }
    TR abig = 0.0Q, amed = 0.0Q, asml = 0.0Q;
    for (ptrdiff_t i = 0; i < nthreads; ++i) { abig += pbig[i]; amed += pmed[i]; asml += psml[i]; }
    *out = qnrm2_finalize(abig, amed, asml);
    return 1;
}
#endif

static TR qnrm2_core(ptrdiff_t n, const TR *x, ptrdiff_t incx)
{
    if (n <= 0) return 0.0Q;
    if (!blue_inited) blue_init();

#ifdef _OPENMP
    if (incx == 1) {
        TR r;
        if (qnrm2_omp(n, x, &r)) return r;
    }
#endif

    TR abig = 0.0Q, amed = 0.0Q, asml = 0.0Q;
    bool notbig = 1;
    ptrdiff_t ix = (incx < 0) ? -(n - 1) * incx : 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        TR ax = fabsq(x[ix]);
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
    return qnrm2_finalize(abig, amed, asml);
}

EPBLAS_FACADE_ASUM(qnrm2, TR, TR)
