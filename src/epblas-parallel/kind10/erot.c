#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
/* erot — kind10 real Givens rotation. */
typedef long double T;

/* Unit-stride kernel, shared by the serial entry and the per-thread OMP slices.
 * Load both elements into locals before storing: x and y are distinct `T*` the
 * compiler can't prove non-aliasing for, so reading them inline across the
 * interleaved writes forces an 80-bit reload each iteration. Hoisting to locals
 * (the epblas-openblas shape) keeps them on the x87 register stack. Same ops in
 * the same order — bit-identical.
 *
 * 4-way unrolled. fp80 has no SIMD, so the only lever over the 1-element loop is
 * amortizing the per-element loop overhead (add/cmp/jb ~3 insns vs the ~16-insn
 * rotation body). The four elements are emitted SEQUENTIALLY — each completes its
 * c,s products and both stores before the next begins — so peak x87 stack depth
 * is unchanged (c,s pinned + one element's working set); unlike a complex MAC,
 * real rotation never approaches the 8-deep cap, so this cannot spill. This wins
 * the L2-band size (N~64k OMP4), where the per-element overhead is exposed; at
 * cache-overflow (~1M) it is bandwidth-bound and the unroll is neutral. */
static void erot_unit(ptrdiff_t n, T c, T s, T *x, T *y)
{
    ptrdiff_t i, n1 = n & -4;
    for (i = 0; i < n1; i += 4) {
        T x0 = x[i+0], y0 = y[i+0];
        x[i+0] = c * x0 + s * y0;  y[i+0] = c * y0 - s * x0;
        T x1 = x[i+1], y1 = y[i+1];
        x[i+1] = c * x1 + s * y1;  y[i+1] = c * y1 - s * x1;
        T x2 = x[i+2], y2 = y[i+2];
        x[i+2] = c * x2 + s * y2;  y[i+2] = c * y2 - s * x2;
        T x3 = x[i+3], y3 = y[i+3];
        x[i+3] = c * x3 + s * y3;  y[i+3] = c * y3 - s * x3;
    }
    for (; i < n; ++i) {
        T xi = x[i], yi = y[i];
        x[i] = c * xi + s * yi;
        y[i] = c * yi - s * xi;
    }
}

#ifdef _OPENMP
/* Threaded real Givens — same cache-bandwidth rationale as eaxpy_omp (see
 * eaxpy.c). Compute-bound (4 mul + 2 add per element); measured proto4/par1
 * 4 mul + 2 add per element, so heavy enough to break even early. Threshold set
 * by par4<=ob4 (ob keeps rot serial at small N). Measured under iomp5: par4/ob4
 * 1.15@1024, then 1.009@1536 and 0.95@3072 — break-even ~1536, stay serial
 * through 1024. */
#define EROT_OMP_MIN 1024
static bool erot_omp(ptrdiff_t n, T c, T s, T *x, T *y)
{
    if (n <= EROT_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) erot_unit(hi - lo, c, s, x + lo, y + lo);
    }
    return 1;
}
#endif

static void erot_core(ptrdiff_t n, T *x, ptrdiff_t incx, T *y, ptrdiff_t incy,
                      const T *c_, const T *s_)
{
    const T c = *c_, s = *s_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (erot_omp(n, c, s, x, y)) return;
#endif
        erot_unit(n, c, s, x, y);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            T xi = x[ix], yi = y[iy];
            x[ix] = c * xi + s * yi;
            y[iy] = c * yi - s * xi;
            ix += incx; iy += incy;
        }
    }
}

EPBLAS_FACADE_ROT(erot, T, T)
