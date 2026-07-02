/* qscal — kind16 real: X := α · X. */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __float128 TR;

/* Unit-stride kernel (5-way unroll matching NETLIB DSCAL), shared by the
 * serial entry and the per-thread OMP slabs. Each *= is an independent
 * __multf3 call, so any slab partition is bit-identical to one serial pass. */
static void qscal_unit(ptrdiff_t n, TR alpha, TR *x)
{
    const ptrdiff_t m = n % 5;
    for (ptrdiff_t i = 0; i < m; ++i) x[i] *= alpha;
    for (ptrdiff_t i = m; i < n; i += 5) {
        x[i    ] *= alpha;
        x[i + 1] *= alpha;
        x[i + 2] *= alpha;
        x[i + 3] *= alpha;
        x[i + 4] *= alpha;
    }
}

#ifdef _OPENMP
/* Threaded elementwise SCAL — quad is compute-bound, so it threads (see
 * qaxpy.c). Unit stride hands each thread a contiguous slab of the SAME
 * 5-way kernel the serial path runs (escal_omp's shape): the old
 * `parallel for` over `x[ix0 + i*incx]` ran a rolled one-per-iteration
 * body per thread, giving back ~3% to ob4 at 64k-1M. Strided keeps the
 * index-from-i parallel for (covers every stride). */
#define QSCAL_OMP_MIN 128
__attribute__((noinline)) static bool qscal_omp(ptrdiff_t n, TR alpha, TR *x, ptrdiff_t incx)
{
    if (n <= QSCAL_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (incx == 1) {
        #pragma omp parallel num_threads(nthreads)
        {
            ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
            ptrdiff_t lo = blas_part_bound(n, tid, nth);
            ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
            if (lo < hi) qscal_unit(hi - lo, alpha, x + lo);
        }
    } else {
        ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for (ptrdiff_t i = 0; i < n; ++i) x[ix0 + i * incx] *= alpha;
    }
    return 1;
}
#endif

static void qscal_core(ptrdiff_t n, const TR *alpha_, TR *x, ptrdiff_t incx)
{
    const TR alpha = *alpha_;
    if (n <= 0 || alpha == 1.0Q) return;
#ifdef _OPENMP
    if (qscal_omp(n, alpha, x, incx)) return;
#endif
    if (incx == 1) {
        qscal_unit(n, alpha, x);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { x[ix] *= alpha; ix += incx; }
    }
}

EPBLAS_FACADE_SCAL(qscal, TR, TR)
