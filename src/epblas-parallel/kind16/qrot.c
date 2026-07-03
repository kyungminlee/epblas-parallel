/* qrot — kind16 real Givens rotation. */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __float128 TR;

/* Unit-stride kernel, shared by the serial entry and the per-thread OMP
 * slabs. Each iteration is an independent chain of libquadmath calls, so
 * any slab partition is bit-identical to one serial pass. */
static void qrot_unit(ptrdiff_t n, TR *x, TR *y, TR c, TR s)
{
    for (ptrdiff_t i = 0; i < n; ++i) {
        TR tx = c * x[i] + s * y[i];
        y[i] = c * y[i] - s * x[i];
        x[i] = tx;
    }
}

#ifdef _OPENMP
/* Threaded Givens rotation — quad is compute-bound, so it threads (see
 * qaxpy.c). Unit stride hands each thread a contiguous slab of the SAME
 * kernel the serial path runs (qscal_omp's shape — the rolled index-from-i
 * `parallel for` body pays two index multiplies per iteration). Strided
 * keeps the index-from-i parallel for (covers every stride). */
#define QROT_OMP_MIN 128
__attribute__((noinline)) static bool qrot_omp(ptrdiff_t n, TR *x, ptrdiff_t incx,
                                              TR *y, ptrdiff_t incy, TR c, TR s)
{
    if (n <= QROT_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (incx == 1 && incy == 1) {
        #pragma omp parallel num_threads(nthreads)
        {
            ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
            ptrdiff_t lo = blas_part_bound(n, tid, nth);
            ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
            if (lo < hi) qrot_unit(hi - lo, x + lo, y + lo, c, s);
        }
    } else {
        ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy0 = (incy < 0) ? (-n + 1) * incy : 0;
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for (ptrdiff_t i = 0; i < n; ++i) {
            ptrdiff_t ix = ix0 + i * incx, iy = iy0 + i * incy;
            TR tx = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = tx;
        }
    }
    return 1;
}
#endif

static void qrot_core(ptrdiff_t n, TR *x, ptrdiff_t incx, TR *y, ptrdiff_t incy,
                      const TR *c_, const TR *s_)
{
    const TR c = *c_, s = *s_;
    if (n <= 0) return;
#ifdef _OPENMP
    if (qrot_omp(n, x, incx, y, incy, c, s)) return;
#endif
    if (incx == 1 && incy == 1) {
        qrot_unit(n, x, y, c, s);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            TR tx = c * x[ix] + s * y[iy];
            y[iy] = c * y[iy] - s * x[ix];
            x[ix] = tx;
            ix += incx; iy += incy;
        }
    }
}

EPBLAS_FACADE_ROT(qrot, TR, TR)
