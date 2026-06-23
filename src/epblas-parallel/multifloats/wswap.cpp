/*
 * wswap — multifloats complex DD: swap X ↔ Y.
 */
#include <cstddef>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define WSWAP_OMP_MIN 8192
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TC = mf::complex64x2;

static void wswap_core(std::ptrdiff_t n,
                       TC *x, std::ptrdiff_t incx,
                       TC *y, std::ptrdiff_t incy)
{
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (n > WSWAP_OMP_MIN && blas_omp_should_thread()) {
            std::ptrdiff_t nthreads = blas_omp_max_threads();
            #pragma omp parallel for schedule(static) num_threads(nthreads)
            for (std::ptrdiff_t i = 0; i < n; ++i) { TC t = x[i]; x[i] = y[i]; y[i] = t; }
            return;
        }
#endif
        for (std::ptrdiff_t i = 0; i < n; ++i) { TC t = x[i]; x[i] = y[i]; y[i] = t; }
    } else {
        std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) { TC t = x[ix]; x[ix] = y[iy]; y[iy] = t; ix += incx; iy += incy; }
    }
}

extern "C" {
EPBLAS_FACADE_SWAP(wswap, TC)
}
