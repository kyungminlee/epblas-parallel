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

namespace mf = multifloats;
using T = mf::complex64x2;

extern "C" void wswap_(const int *n_,
                       T *x, const int *incx_,
                       T *y, const int *incy_)
{
    const std::ptrdiff_t n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (n > WSWAP_OMP_MIN && blas_omp_available() && !omp_in_parallel()) {
            std::ptrdiff_t nthreads = blas_omp_max_threads();
            #pragma omp parallel for schedule(static) num_threads(nthreads)
            for (std::ptrdiff_t i = 0; i < n; ++i) { T t = x[i]; x[i] = y[i]; y[i] = t; }
            return;
        }
#endif
        for (std::ptrdiff_t i = 0; i < n; ++i) { T t = x[i]; x[i] = y[i]; y[i] = t; }
    } else {
        std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) { T t = x[ix]; x[ix] = y[iy]; y[iy] = t; ix += incx; iy += incy; }
    }
}
