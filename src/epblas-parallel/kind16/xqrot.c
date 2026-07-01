/* xqrot — kind16: complex Givens with real c, s (CSROT/ZDROT analog). */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __complex128 TC;
typedef __float128 R;

/* Real coefficients on complex data: treat each complex element as two
 * independent reals and run the plain real rotation over them.  The
 * contiguous case becomes one flat 2n real loop (the epblas-openblas shape) —
 * load-first, no __complex128 member juggling.  Same ops in the same order —
 * bit-identical.  n is the COMPLEX count; the kernel walks 2n reals.
 *
 * NOT unrolled — deliberately.  quad has no SIMD: every mul/add is an opaque
 * libquadmath call (__mulkf3/__addkf3) that clobbers all caller-saved xmm
 * regs.  Loop overhead (1 add/cmp/jne per element) is negligible against 6
 * such calls, so unrolling buys nothing and instead keeps several elements
 * live across the call boundaries, forcing extra stack spills (~3.4 vs ~2.3
 * vmovdqa/call, measured) → the opposite of the fp80 yerot case where inline
 * x87 ops make loop overhead the lever.  A simple 1-element loop matches the
 * ob kernel's spill count and closes the serial par/ob gap. */
static void xqrot_unit(ptrdiff_t n, R c, R s, R *px, R *py)
{
    const long two_n = 2L * n;
    for (long i = 0; i < two_n; ++i) {
        R xi = px[i], yi = py[i];
        px[i] = c * xi + s * yi;
        py[i] = c * yi - s * xi;
    }
}

#ifdef _OPENMP
/* Threaded complex Givens (real c, s) — quad is compute-bound, so it threads
 * (see qaxpy.c).  Each iteration is independent; partition the flat real
 * kernel over disjoint element ranges. */
#define XQROT_OMP_MIN 128
static bool xqrot_omp(ptrdiff_t n, R c, R s, R *px, R *py)
{
    if (n <= XQROT_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) xqrot_unit(hi - lo, c, s, px + 2 * lo, py + 2 * lo);
    }
    return 1;
}
#endif

static void xqrot_core(ptrdiff_t n, TC *x, ptrdiff_t incx, TC *y, ptrdiff_t incy,
                       const R *c_, const R *s_)
{
    const R c = *c_, s = *s_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        R *px = (R *)x, *py = (R *)y;
#ifdef _OPENMP
        if (xqrot_omp(n, c, s, px, py)) return;
#endif
        xqrot_unit(n, c, s, px, py);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            R *px = (R *)&x[ix], *py = (R *)&y[iy];
            R xr = px[0], xi = px[1], yr = py[0], yi = py[1];
            px[0] = c * xr + s * yr; px[1] = c * xi + s * yi;
            py[0] = c * yr - s * xr; py[1] = c * yi - s * xi;
            ix += incx; iy += incy;
        }
    }
}

EPBLAS_FACADE_ROT(xqrot, R, TC)
