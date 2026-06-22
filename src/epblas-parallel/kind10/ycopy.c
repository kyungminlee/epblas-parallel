/* ycopy — kind10 complex: Y := X. */
#include <string.h>
#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef _Complex long double T;

#ifdef _OPENMP
/* Threaded unit-stride complex COPY — cache-bandwidth rationale as ecopy_omp
 * (see ecopy.c): COPY needs an UPPER bound because par's serial memcpy already
 * saturates main-memory bandwidth on one core. In the cache regime threading
 * still helps (measured proto4/par1 ~0.30 at N=16K..64K, 0.63 at 131072), but
 * past that the arrays exceed L3 and extra threads only add contention
 * (proto4/par1 ~1.27 at 262144). A complex element is 2x the bytes of a real
 * one, so the upper bound is ~half ecopy's: engage only inside [1024, 131072].
 * Each thread memcpy's its own slice. */
#define YCOPY_OMP_MIN 1024
#define YCOPY_OMP_MAX 131072
static int ycopy_omp(ptrdiff_t n, const T *x, T *y)
{
    if (n <= YCOPY_OMP_MIN || n > YCOPY_OMP_MAX ||
        blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
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

static void ycopy_core(ptrdiff_t n, const T *x, ptrdiff_t incx, T *y, ptrdiff_t incy)
{
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (ycopy_omp(n, x, y)) return;
#endif
        memcpy(y, x, (size_t)n * sizeof(T));
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}

EPBLAS_FACADE_COPY(ycopy, T)
