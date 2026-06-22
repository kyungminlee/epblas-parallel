/* ecopy — kind10 real: Y := X. */
#include <string.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef long double T;

#ifdef _OPENMP
/* Threaded unit-stride COPY — cache-bandwidth rationale as eaxpy_omp (see
 * eaxpy.c), but COPY needs an UPPER bound. par's serial copy is a single memcpy
 * that already saturates main-memory bandwidth one core, and it already beats
 * OpenBLAS 3-6x serially; in the cache regime (~N=2K..512K) per-core bandwidth
 * still scales (measured par4/par1 ~0.27-0.74), but past ~1M the arrays exceed
 * L3 and extra threads only add fork/contention overhead (par4/par1 ~1.05-1.16,
 * a net LOSS). So engage only inside [2048, 524288]; serial elsewhere. Each
 * thread memcpy's its own slice. */
#define ECOPY_OMP_MIN 2048
#define ECOPY_OMP_MAX 524288
static bool ecopy_omp(ptrdiff_t n, const T *x, T *y)
{
    if (n <= ECOPY_OMP_MIN || n > ECOPY_OMP_MAX ||
        !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) memcpy(y + lo, x + lo, (size_t)(hi - lo) * sizeof(T));
    }
    return 1;
}
#endif

static void ecopy_core(ptrdiff_t n, const T *x, ptrdiff_t incx, T *y, ptrdiff_t incy)
{
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (ecopy_omp(n, x, y)) return;
#endif
        memcpy(y, x, (size_t)n * sizeof(T));
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_COPY(ecopy, T)
