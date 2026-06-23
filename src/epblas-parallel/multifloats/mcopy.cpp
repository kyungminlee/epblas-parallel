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
#include "mf_omp.h"
/* Threaded unit-stride copy engages only in the CACHE-RESIDENT window. The
 * 16-byte DD element (two packed doubles) copies fastest as a compiler-
 * vectorized typed loop (clean AVX moves) — that beats glibc memcpy's
 * medium-copy `rep movsb` path when the slices live in L2/L3, and unlike
 * memcpy it actually scales with threads there (ob's element loop does;
 * par's threaded memcpy did not, losing N=65536). Past ~L3 (n > MAX, 4 MB
 * byte-equivalent) the arrays stream from DRAM: a single-core memcpy's
 * non-temporal stores already saturate the bus (par serial memcpy at N=1M
 * beats its own threaded memcpy), so extra threads only add contention and
 * we fall back to the serial memcpy. */
#define MCOPY_OMP_MIN 8192
#define MCOPY_OMP_MAX 262144
#endif
#include "../common/epblas_facade.h"

namespace mf = multifloats;
using TR = mf::float64x2;

static void mcopy_core(std::ptrdiff_t n,
                       const TR *x, std::ptrdiff_t incx,
                       TR *y, std::ptrdiff_t incy)
{
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (n > MCOPY_OMP_MIN && n <= MCOPY_OMP_MAX &&
            blas_omp_should_thread()) {
            std::ptrdiff_t nthreads = blas_omp_max_threads();
            #pragma omp parallel num_threads(nthreads)
            {
                std::ptrdiff_t tid = omp_get_thread_num();
                std::ptrdiff_t nth = omp_get_num_threads();
                std::ptrdiff_t lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
                for (std::ptrdiff_t i = lo; i < hi; ++i) y[i] = x[i];
            }
            return;
        }
#endif
        std::memcpy(y, x, static_cast<std::size_t>(n) * sizeof(TR));
    } else {
        std::ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        std::ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (std::ptrdiff_t i = 0; i < n; ++i) { y[iy] = x[ix]; ix += incx; iy += incy; }
    }
}

extern "C" { EPBLAS_FACADE_COPY(mcopy, TR) }
