/* qcopy — kind16 real: Y := X. */
#include <stddef.h>
#include <string.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __float128 TR;

#ifdef _OPENMP
/* Threaded copy. Pure data movement, but threading still spreads the quad-wide
 * stream across cores for more memory bandwidth (ob threads it ~2x). Higher
 * crossover than the compute-bound RMW ops since small n stays in cache.
 * schedule(static) gives each thread a contiguous block (sequential stores).
 * Real-quad copy only wins past L2 (crossover ~8K; n=4096 still washes). */
#define QCOPY_OMP_MIN 8192
#define QCOPY_MEMCPY_SLAB_MIN 262144
__attribute__((noinline)) static bool qcopy_omp(ptrdiff_t n, const TR *x, ptrdiff_t incx,
                                               TR *y, ptrdiff_t incy)
{
    if (n <= QCOPY_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (incx == 1 && incy == 1 && n >= QCOPY_MEMCPY_SLAB_MIN) {
        /* Per-thread contiguous memcpy slab (qscal_omp's shape), large-n
         * only: past L3 glibc memcpy's non-temporal stores skip the RFO
         * traffic the elementwise loop pays (omp4 1M par/ob 1.00 -> 0.77
         * measured), but in the cache-resident band memcpy LOSES ~2-4% to
         * the rolled loop (omp4 64K 1.00 -> 1.02, repeatable), so that band
         * falls through to the rolled form below. Pure data movement,
         * trivially partition-invariant. */
        #pragma omp parallel num_threads(nthreads)
        {
            ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
            ptrdiff_t lo = blas_part_bound(n, tid, nth);
            ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
            if (lo < hi) memcpy(y + lo, x + lo, (size_t)(hi - lo) * sizeof(TR));
        }
    } else {
        ptrdiff_t ix0 = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy0 = (incy < 0) ? (-n + 1) * incy : 0;
        #pragma omp parallel for schedule(static) num_threads(nthreads)
        for (ptrdiff_t i = 0; i < n; ++i) y[iy0 + i * incy] = x[ix0 + i * incx];
    }
    return 1;
}
#endif

static void qcopy_core(ptrdiff_t n, const TR *x, ptrdiff_t incx, TR *y, ptrdiff_t incy)
{
    if (n <= 0) return;
#ifdef _OPENMP
    if (qcopy_omp(n, x, incx, y, incy)) return;
#endif
    if (incx == 1 && incy == 1) memcpy(y, x, (size_t)n * sizeof(TR));
    else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_COPY(qcopy, TR)
