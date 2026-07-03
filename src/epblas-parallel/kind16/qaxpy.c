/* qaxpy — kind16 real: Y := α·X + Y. */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __float128 TR;

/* Unit-stride kernel (4-way unroll), shared by the serial entry and the
 * per-thread OMP slabs. Each += is an independent __multf3/__addtf3 pair,
 * so any slab partition is bit-identical to one serial pass. */
static void qaxpy_unit(ptrdiff_t n, TR alpha, const TR *x, TR *y)
{
    const ptrdiff_t m = n % 4;
    for (ptrdiff_t i = 0; i < m; ++i) y[i] += alpha * x[i];
    for (ptrdiff_t i = m; i < n; i += 4) {
        y[i    ] += alpha * x[i    ];
        y[i + 1] += alpha * x[i + 1];
        y[i + 2] += alpha * x[i + 2];
        y[i + 3] += alpha * x[i + 3];
    }
}

#ifdef _OPENMP
/* Threaded elementwise AXPY. Unlike fp80 (memory-bound, kept serial), quad is
 * compute-bound under libquadmath, so even the RMW L1 ops thread profitably
 * (crossover ~n=128). Unit stride hands each thread a contiguous slab of the
 * SAME 4-way kernel the serial path runs (qscal_omp's shape — the rolled
 * one-per-iteration `parallel for` body gave back ~3% to ob4 there). Strided
 * keeps the index-from-i parallel for (covers every stride); the serial
 * fast-paths below stay intact for the sub-threshold / single-thread case. */
#define QAXPY_OMP_MIN 128
__attribute__((noinline)) static bool qaxpy_omp(ptrdiff_t n, TR alpha,
                                               const TR *x, ptrdiff_t incx,
                                               TR *y, ptrdiff_t incy)
{
    if (n <= QAXPY_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (incx == 1 && incy == 1) {
        #pragma omp parallel num_threads(nthreads)
        {
            ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
            ptrdiff_t lo = blas_part_bound(n, tid, nth);
            ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
            if (lo < hi) qaxpy_unit(hi - lo, alpha, x + lo, y + lo);
        }
    } else {
        ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy0 = (incy < 0) ? (-n + 1) * incy : 0;
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for (ptrdiff_t i = 0; i < n; ++i) y[iy0 + i * incy] += alpha * x[ix0 + i * incx];
    }
    return 1;
}
#endif

static void qaxpy_core(ptrdiff_t n, const TR *alpha_,
                       const TR *x, ptrdiff_t incx,
                       TR *y, ptrdiff_t incy)
{
    const TR alpha = *alpha_;
    if (n <= 0 || alpha == 0.0Q) return;
#ifdef _OPENMP
    if (qaxpy_omp(n, alpha, x, incx, y, incy)) return;
#endif
    if (incx == 1 && incy == 1) {
        qaxpy_unit(n, alpha, x, y);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { y[iy] += alpha * x[ix]; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_AXPY(qaxpy, TR, TR)
