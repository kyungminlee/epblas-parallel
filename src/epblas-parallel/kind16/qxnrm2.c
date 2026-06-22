/* qxnrm2 — kind16: ||X||₂ for complex X (real result).
 *
 * Blue's algorithm: single pass, three magnitude-bucketed accumulators.
 * Same as qnrm2 but processes Re/Im as two values per element.
 */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
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
static void qxnrm2_bucket(ptrdiff_t n, const T *x, R *abig_, R *amed_, R *asml_)
{
    R abig = 0.0Q, amed = 0.0Q, asml = 0.0Q;
    int notbig = 1;
    for (ptrdiff_t i = 0; i < n; ++i) {
        blue_bucket(fabsq(__real__ x[i]), &abig, &amed, &asml, &notbig);
        blue_bucket(fabsq(__imag__ x[i]), &abig, &amed, &asml, &notbig);
    }
    *abig_ = abig; *amed_ = amed; *asml_ = asml;
}

/* Threaded reduction for large unit-stride X (see qnrm2 for the rationale). */
#define QXNRM2_OMP_MIN 128
#define QXNRM2_MAX_CPUS 64
__attribute__((noinline)) static int qxnrm2_omp(ptrdiff_t n, const T *x, R *out)
{
    if (n <= QXNRM2_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
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
    for (int i = 0; i < nthreads; ++i) { abig += pbig[i]; amed += pmed[i]; asml += psml[i]; }
    *out = qxnrm2_finalize(abig, amed, asml);
    return 1;
}
#endif

static R qxnrm2_core(ptrdiff_t n, const T *x, ptrdiff_t incx)
{
    if (n <= 0) return 0.0Q;
    if (!blue_inited) blue_init();

#ifdef _OPENMP
    if (incx == 1) {
        R r;
        if (qxnrm2_omp(n, x, &r)) return r;
    }
#endif

    R abig = 0.0Q, amed = 0.0Q, asml = 0.0Q;
    int notbig = 1;
    ptrdiff_t ix = (incx < 0) ? -(n - 1) * incx : 0;
    for (ptrdiff_t i = 0; i < n; ++i) {
        blue_bucket(fabsq(__real__ x[ix]), &abig, &amed, &asml, &notbig);
        blue_bucket(fabsq(__imag__ x[ix]), &abig, &amed, &asml, &notbig);
        ix += incx;
    }
    return qxnrm2_finalize(abig, amed, asml);
}

EPBLAS_FACADE_ASUM(qxnrm2, R, T)
