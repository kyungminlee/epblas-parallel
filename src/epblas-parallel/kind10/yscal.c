#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
/* yscal — kind10 complex: X := α·X (α and X both complex).
 *
 * Manual __real__/__imag__ expansion with imag-part products written
 * as `xi*ar + xr*ai` (X.im term first). gcc emits x87 products in
 * source order; this order matches the value already on top of the
 * x87 stack after the first store, saving one `fxch` per iter
 * relative to the natural `xr*ai + xi*ar` form. With the default
 * order, the inner loop is 15 insns per element with 2 fxch; this
 * form is 14 insns with 1 fxch — matches gfortran's ZSCAL codegen.
 */
#include "../common/epblas_facade.h"
typedef _Complex long double T;

/* Unit-stride kernel over the (re,im)-pair view, shared by serial + OMP slices.
 *
 * 4-way unrolled (over complex elements) to amortize per-element loop overhead;
 * fp80 has no SIMD so this is the only lever above a 1-element loop. The four
 * elements are emitted SEQUENTIALLY — each completes both its stores before the
 * next begins — so only one complex element's working set is live at a time and
 * peak x87 stack depth is unchanged (ar,ai pinned + xr,xi + temps). This is the
 * key distinction from INTERLEAVING two complex MAC chains, which overruns the
 * 8-deep stack and spills (cf. ysyr2k 2-col interleave); sequential unrolling
 * does not. Product order is preserved bit-for-bit (xr*ar-xi*ai, xi*ar+xr*ai),
 * keeping the 1-fxch codegen. Same N=64k-OMP4 / serial loop-overhead win as the
 * erot kernel. */
static void yscal_unit(ptrdiff_t n, long double ar, long double ai, long double *p)
{
    ptrdiff_t i, n1 = n & -4;
    for (i = 0; i < n1; i += 4, p += 8) {
        const long double r0 = p[0], m0 = p[1];
        p[0] = r0 * ar - m0 * ai;  p[1] = m0 * ar + r0 * ai;
        const long double r1 = p[2], m1 = p[3];
        p[2] = r1 * ar - m1 * ai;  p[3] = m1 * ar + r1 * ai;
        const long double r2 = p[4], m2 = p[5];
        p[4] = r2 * ar - m2 * ai;  p[5] = m2 * ar + r2 * ai;
        const long double r3 = p[6], m3 = p[7];
        p[6] = r3 * ar - m3 * ai;  p[7] = m3 * ar + r3 * ai;
    }
    for (; i < n; ++i, p += 2) {
        const long double xr = p[0], xi = p[1];
        p[0] = xr * ar - xi * ai;
        p[1] = xi * ar + xr * ai;
    }
}

#ifdef _OPENMP
/* Threaded unit-stride complex SCAL — same cache-bandwidth rationale as
 * eaxpy_omp (see eaxpy.c); complex alpha = 4 mul + 2 add/elem, more compute than
 * yescal so the break-even count is a touch lower. Threshold set by par4<=ob4
 * (ob keeps scal serial at small N). Measured under iomp5: par4/ob4 1.15@1024,
 * 1.04@1536, then 0.99@2048 — break-even ~2048, stay serial through 1536. */
#define YSCAL_OMP_MIN 1536
static int yscal_omp(ptrdiff_t n, long double ar, long double ai, long double *base)
{
    if (n <= YSCAL_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) yscal_unit(hi - lo, ar, ai, base + 2 * lo);
    }
    return 1;
}
#endif

static void yscal_core(ptrdiff_t n, const T *alpha_, T *x, ptrdiff_t incx)
{
    if (n <= 0) return;
    const long double ar = __real__ *alpha_;
    const long double ai = __imag__ *alpha_;
    if (ar == 1.0L && ai == 0.0L) return;
    long double *base = (long double *)x;
    if (incx == 1) {
#ifdef _OPENMP
        if (yscal_omp(n, ar, ai, base)) return;
#endif
        yscal_unit(n, ar, ai, base);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            const long double xr = base[2*ix], xi = base[2*ix + 1];
            base[2*ix]     = xr * ar - xi * ai;
            base[2*ix + 1] = xi * ar + xr * ai;
            ix += incx;
        }
    }
}

EPBLAS_FACADE_SCAL(yscal, T, T)
