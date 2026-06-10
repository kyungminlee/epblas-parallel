/* qswap — kind16 real: swap X ↔ Y. */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __float128 T;

/* Unit-stride 4-way kernel + strided fallback, shared by the serial entry and
 * the per-thread OMP slices (mirrors ob swap_kernel / kind10 eswap_unit). The
 * old path had two deficits: the serial unit loop was an int-indexed 3-way that
 * amortized loop overhead over 3 not 4 and carried a separate counter (~4-8%
 * slower serial), and the OMP body was an un-unrolled scalar indexed swap that
 * under-threaded at large N. One unrolled kernel fixes both. Independent
 * elements ⇒ bit-exact regardless of grouping. */
static void qswap_kernel(ptrdiff_t n, T *x, ptrdiff_t incx, T *y, ptrdiff_t incy)
{
    if (incx == 1 && incy == 1) {
        const ptrdiff_t n1 = n & -4;
        ptrdiff_t i;
        for (i = 0; i < n1; i += 4) {
            T t0 = x[i], t1 = x[i + 1], t2 = x[i + 2], t3 = x[i + 3];
            x[i] = y[i]; x[i + 1] = y[i + 1]; x[i + 2] = y[i + 2]; x[i + 3] = y[i + 3];
            y[i] = t0; y[i + 1] = t1; y[i + 2] = t2; y[i + 3] = t3;
        }
        for (; i < n; ++i) { T t = x[i]; x[i] = y[i]; y[i] = t; }
        return;
    }
    for (ptrdiff_t i = 0; i < n; ++i) {
        T t = x[i * incx]; x[i * incx] = y[i * incy]; y[i * incy] = t;
    }
}

#ifdef _OPENMP
/* Threaded swap — two quad streams read+write; threading spreads them across
 * cores for memory bandwidth (see qcopy.c). Each thread swaps a contiguous
 * slice via the unrolled kernel. Real-quad swap only wins past L2 (crossover
 * ~8K; n=4096 still washes). */
#define QSWAP_OMP_MIN 8192
__attribute__((noinline)) static int qswap_omp(ptrdiff_t n, T *x, ptrdiff_t incx,
                                               T *y, ptrdiff_t incy)
{
    if (n <= QSWAP_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi)
            qswap_kernel(hi - lo, x + lo * incx, incx, y + lo * incy, incy);
    }
    return 1;
}
#endif

void qswap_(const int *n_, T *x, const int *incx_, T *y, const int *incy_)
{
    const ptrdiff_t n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
    /* Shift to the first logical element for negative strides, then walk forward
     * with i*inc inside the kernel (matches ob). */
    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;
#ifdef _OPENMP
    if (qswap_omp(n, x, incx, y, incy)) return;
#endif
    qswap_kernel(n, x, incx, y, incy);
}
