/* mdot — multifloats real DD: Σ X·Y.
 *
 * SIMD Bailey-wide accumulator (V5). 4-wide SIMD with a 3-double-per-lane
 * wide accumulator (a0, a1, a2). Periodic renorm every K=64 iters keeps the
 * accumulator stable. Final horizontal reduce.
 *
 * Bench (vs scalar full-DD):
 *   N=64..1024 : 1.0–1.4× (small-N dispatch dominates)
 *   N=8K..1M   : ~5× faster, full DD precision (~30–32 digits)
 *
 * See V3 vs V5 study: Bailey-wide consistently beats SIMD full-DD on both
 * speed and precision; Kahan-defer was rejected because it collapses to
 * double precision.
 */
#include <cstddef>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include "mf_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;

#ifdef MBLAS_SIMD_DD
#include <immintrin.h>
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"

namespace {
/* canonical EFTs — mf_simd_fast.h (2a-5) */
using simd_fast::twoprod;
using simd_fast::fast2sum;
using simd_fast::twosum;
using simd_exact::load_dd4;
using simd_fast::horizontal_dd;  /* Bailey 2-limb finalizer — mf_simd_fast.h (#4) */
}
#endif

/* Σ X·Y over contiguous unit-stride ranges — the serial kernel, unchanged.
 * Carved out so the OpenMP partial-reduction can call it per sub-range. */
static T mdot_unit(int n, const T *x, const T *y)
{
#ifdef MBLAS_SIMD_DD
    __m256d a0 = _mm256_setzero_pd();
    __m256d a1 = _mm256_setzero_pd();
    __m256d a2 = _mm256_setzero_pd();
    constexpr int K = 64;
    int counter = K;
    const int n4 = n & ~3;
    for (int i = 0; i < n4; i += 4) {
        __m256d xh, xl, yh, yl;
        load_dd4(&x[i], xh, xl);
        load_dd4(&y[i], yh, yl);
        /* DD product via simplified twoprod: drop xl*yl (~2^-106 of xh*yh) */
        __m256d ph, pl;
        twoprod(xh, yh, ph, pl);
        pl = _mm256_add_pd(pl,
                _mm256_add_pd(_mm256_mul_pd(xh, yl), _mm256_mul_pd(xl, yh)));
        /* Wide-acc absorb (a0, a1, a2) += (ph, pl) */
        __m256d e0, e1, e2;
        twosum(a0, ph, a0, e0);
        twosum(a1, pl, a1, e1);
        twosum(a1, e0, a1, e2);
        a2 = _mm256_add_pd(a2, _mm256_add_pd(e1, e2));
        if (--counter == 0) {
            /* Renormalize 3→2 doubles */
            __m256d t, e;
            fast2sum(a1, a2, t, e);
            a1 = t; a2 = e;
            fast2sum(a0, a1, a0, a1);
            a1 = _mm256_add_pd(a1, a2);
            fast2sum(a0, a1, a0, a1);
            a2 = _mm256_setzero_pd();
            counter = K;
        }
    }
    __m256d t = _mm256_add_pd(a1, a2);
    T s = horizontal_dd(a0, t);
    for (int i = n4; i < n; ++i) s = s + x[i] * y[i];
    return s;
#else
    T s0{0.0, 0.0}, s1{0.0, 0.0}, s{0.0, 0.0};
    int i = 0;
    for (; i + 1 < n; i += 2) {
        s0 = s0 + x[i]     * y[i];
        s1 = s1 + x[i + 1] * y[i + 1];
    }
    s = s0 + s1;
    for (; i < n; ++i) s = s + x[i] * y[i];
    return s;
#endif
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride X·Y. Each thread dots its
 * contiguous slice with the serial kernel; partials combine in tid order.
 * Reduction order differs from serial → within fuzz tolerance (kind10 edot). */
#define MDOT_OMP_MIN 8192
#define MDOT_MAX_CPUS 64
__attribute__((noinline)) static int mdot_omp(int n, const T *x, const T *y, T *out)
{
    if (n <= MDOT_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > MDOT_MAX_CPUS) nthreads = MDOT_MAX_CPUS;
    T partial[MDOT_MAX_CPUS];
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo, hi; mf_omp::even_slice(n, tid, nth, lo, hi);
        partial[tid] = (lo < hi) ? mdot_unit(hi - lo, x + lo, y + lo) : T{0.0, 0.0};
    }
    T s{0.0, 0.0};
    for (int i = 0; i < nthreads; ++i) s = s + partial[i];
    *out = s;
    return 1;
}
#endif

extern "C" T mdot_(const int *n_,
                   const T *x, const int *incx_,
                   const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    T s{0.0, 0.0};
    if (n <= 0) return s;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (mdot_omp(n, x, y, &s)) return s;
#endif
        return mdot_unit(n, x, y);
    }

    /* Strided fallback */
    int ix = (incx < 0) ? (-n + 1) * incx : 0;
    int iy = (incy < 0) ? (-n + 1) * incy : 0;
    for (int i = 0; i < n; ++i) { s = s + x[ix] * y[iy]; ix += incx; iy += incy; }
    return s;
}
