/*
 * wcopy — multifloats complex DD: Y := X.
 */
#include <cstddef>
#include <cstring>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
/* Threaded unit-stride copy engages only in the CACHE-RESIDENT window. The
 * 32-byte complex element copies fastest as a compiler-vectorized typed loop
 * (~100 GB/s, clean 32B AVX moves) — that beats glibc memcpy's medium-copy
 * `rep movsb` path when the slices live in L2/L3. Past ~L3 (n > MAX, 8 MB
 * byte-equivalent like kind10 ecopy's 524288·16B) the arrays stream from DRAM:
 * there a single-core memcpy's non-temporal stores saturate the bus and extra
 * threads only add fork/contention, so we fall back to the serial memcpy. */
#define WCOPY_OMP_MIN 8192
#define WCOPY_OMP_MAX 262144
#endif

namespace mf = multifloats;
using T = mf::complex64x2;

extern "C" void wcopy_(const int *n_,
                       const T *x, const int *incx_,
                       T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (n > WCOPY_OMP_MIN && n <= WCOPY_OMP_MAX &&
            blas_omp_max_threads() > 1 && !omp_in_parallel()) {
            int nthreads = blas_omp_max_threads();
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                int lo = (int)((long long)n * tid / nth);
                int hi = (int)((long long)n * (tid + 1) / nth);
                for (int i = lo; i < hi; ++i) y[i] = x[i];
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
