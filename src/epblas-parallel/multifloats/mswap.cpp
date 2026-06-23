/*
 * mswap — multifloats real DD: swap X ↔ Y.
 *
 * 4-way unrolled kernel (matches ob swap_kernel): DD has no SIMD, so the only
 * lever on this bandwidth/loop-overhead-bound op is amortizing the loop control
 * with independent temps. Threaded path splits into contiguous chunks, each
 * running the same unrolled kernel.
 */
#include <cstddef>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define MSWAP_OMP_MIN 8192
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;

static void mswap_kernel(std::ptrdiff_t n, TR *x, std::ptrdiff_t incx,
                                            TR *y, std::ptrdiff_t incy)
{
    if (incx == 1 && incy == 1) {
        std::ptrdiff_t i, n1 = n & -4;
        for (i = 0; i < n1; i += 4) {
            TR t0 = x[i+0], t1 = x[i+1], t2 = x[i+2], t3 = x[i+3];
            x[i+0] = y[i+0]; x[i+1] = y[i+1];
            x[i+2] = y[i+2]; x[i+3] = y[i+3];
            y[i+0] = t0; y[i+1] = t1; y[i+2] = t2; y[i+3] = t3;
        }
        for (; i < n; ++i) { TR t = x[i]; x[i] = y[i]; y[i] = t; }
        return;
    }
    for (std::ptrdiff_t i = 0; i < n; ++i) {
        TR t = x[i*incx]; x[i*incx] = y[i*incy]; y[i*incy] = t;
    }
}

static void mswap_core(std::ptrdiff_t n,
                       TR *x, std::ptrdiff_t incx,
                       TR *y, std::ptrdiff_t incy)
{
    if (n <= 0) return;
    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

#ifdef _OPENMP
    /* Memory-bandwidth bound (2 reads + 2 writes); disjoint contiguous slices. */
    if (n > MSWAP_OMP_MIN && blas_omp_should_thread()) {
        std::ptrdiff_t nthreads = blas_omp_max_threads();
        #pragma omp parallel num_threads(nthreads)
        {
            std::ptrdiff_t tid = omp_get_thread_num();
            std::ptrdiff_t nth = omp_get_num_threads();
            std::ptrdiff_t chunk = (n + nth - 1) / nth;
            std::ptrdiff_t start = (std::ptrdiff_t)tid * chunk;
            std::ptrdiff_t end   = start + chunk;
            if (end > n) end = n;
            if (start < end)
                mswap_kernel(end - start, x + start * incx, incx,
                                          y + start * incy, incy);
        }
        return;
    }
#endif
    mswap_kernel(n, x, incx, y, incy);
}

extern "C" { EPBLAS_FACADE_SWAP(mswap, TR) }
