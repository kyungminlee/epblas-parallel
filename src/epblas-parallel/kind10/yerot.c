#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
/* yerot — kind10: complex Givens with real c, s. */
typedef _Complex long double TC;
typedef long double R;

/* Real coefficients on complex data: treat each complex element as two
 * independent reals and run the plain real rotation over them.  The
 * contiguous case becomes one flat 2n real loop (the epblas-openblas
 * shape) — load-first, no _Complex multiply, accumulators stay on the
 * x87 register stack.  Same ops in the same order — bit-identical.
 * n is the COMPLEX count; the kernel walks 2n reals.
 *
 * 4-way unrolled (sequentially) to amortize per-element loop overhead — the
 * only lever for fp80, which has no SIMD. Reinterpreting complex as 2n reals
 * makes this the plain real rotation, so it is identical to the erot kernel and
 * (being real) never nears the x87 8-deep cap. Closes the same N=64k-OMP4 /
 * serial loop-overhead residual as erot/yscal. */
static void yerot_unit(ptrdiff_t n, R c, R s, R *px, R *py)
{
    const long two_n = 2L * n;
    long i, m = two_n & -4;
    for (i = 0; i < m; i += 4) {
        R x0 = px[i+0], y0 = py[i+0];
        px[i+0] = c * x0 + s * y0;  py[i+0] = c * y0 - s * x0;
        R x1 = px[i+1], y1 = py[i+1];
        px[i+1] = c * x1 + s * y1;  py[i+1] = c * y1 - s * x1;
        R x2 = px[i+2], y2 = py[i+2];
        px[i+2] = c * x2 + s * y2;  py[i+2] = c * y2 - s * x2;
        R x3 = px[i+3], y3 = py[i+3];
        px[i+3] = c * x3 + s * y3;  py[i+3] = c * y3 - s * x3;
    }
    for (; i < two_n; ++i) {
        R xi = px[i], yi = py[i];
        px[i] = c * xi + s * yi;
        py[i] = c * yi - s * xi;
    }
}

#ifdef _OPENMP
/* Threaded complex Givens — same cache-bandwidth rationale as eaxpy_omp (see
 * eaxpy.c). Compute-bound (4 mul + 2 add per real, 2n reals) and a complex
 * element is 2x bytes, so it threads from low N: measured proto4/par1 ~1.02 at
 * N=128, 0.76 at 192, <1.0 to 4M (~0.60), no upper bound. Break-even ~N=130;
 * 256 keeps margin. */
#define YEROT_OMP_MIN 256
static bool yerot_omp(ptrdiff_t n, R c, R s, R *px, R *py)
{
    if (n <= YEROT_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) yerot_unit(hi - lo, c, s, px + 2 * lo, py + 2 * lo);
    }
    return 1;
}
#endif

static void yerot_core(ptrdiff_t n, TC *x, ptrdiff_t incx, TC *y, ptrdiff_t incy,
                       const R *c_, const R *s_)
{
    const R c = *c_, s = *s_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
        R *px = (R *)x, *py = (R *)y;
#ifdef _OPENMP
        if (yerot_omp(n, c, s, px, py)) return;
#endif
        yerot_unit(n, c, s, px, py);
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

EPBLAS_FACADE_ROT(yerot, R, TC)
