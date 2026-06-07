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
typedef _Complex long double T;

/* Unit-stride kernel over the (re,im)-pair view, shared by serial + OMP slices. */
static void yscal_unit(ptrdiff_t n, long double ar, long double ai, long double *p)
{
    long double *e = p + 2 * (long)n;
    for (; p < e; p += 2) {
        const long double xr = p[0], xi = p[1];
        p[0] = xr * ar - xi * ai;
        p[1] = xi * ar + xr * ai;
    }
}

#ifdef _OPENMP
/* Threaded unit-stride complex SCAL — same cache-bandwidth rationale as
 * eaxpy_omp (see eaxpy.c); complex element = 2x bytes so the break-even shifts
 * to ~half the count: measured proto4/par1 ~0.93 at N=256, <1.0 to 4M (~0.55),
 * no upper bound. Break-even ~N=230; 384 keeps margin. */
#define YSCAL_OMP_MIN 384
static int yscal_omp(ptrdiff_t n, long double ar, long double ai, long double *base)
{
    if (n <= YSCAL_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) yscal_unit(hi - lo, ar, ai, base + 2 * lo);
    }
    return 1;
}
#endif

void yscal_(const int *n_, const T *alpha_, T *x, const int *incx_)
{
    const ptrdiff_t n = *n_, incx = *incx_;
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
