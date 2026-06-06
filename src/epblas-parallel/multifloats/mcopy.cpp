/*
 * mcopy — multifloats real DD: Y := X.
 * Memory-bandwidth bound; SIMD just widens the load/store width.
 */
#include <cstddef>
#include <cstring>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#define MCOPY_OMP_MIN 8192
#endif

namespace mf = multifloats;
using T = mf::float64x2;

extern "C" void mcopy_(const int *n_,
                       const T *x, const int *incx_,
                       T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        /* Memory-bandwidth bound; multiple cores can pull more aggregate
         * bandwidth, so split the memcpy into disjoint slices above the
         * crossover. */
        if (n > MCOPY_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {
            int nthreads = blas_omp_max_threads();
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                int lo = (int)((long long)n * tid / nth);
                int hi = (int)((long long)n * (tid + 1) / nth);
                if (lo < hi)
                    std::memcpy(y + lo, x + lo,
                                static_cast<std::size_t>(hi - lo) * sizeof(T));
            }
            return;
        }
#endif
        std::memcpy(y, x, static_cast<std::size_t>(n) * sizeof(T));
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}
