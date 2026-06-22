/* qasum — kind16 real: returns Σ |X|. */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
/* fabsq via __builtin_fabsf128 — single `pand` instead of a libquadmath function call. */
#undef fabsq
#define fabsq(x) __builtin_fabsf128(x)
typedef __float128 T;

/* Σ|x| over a logical range; 2-accumulator unroll on the unit-stride path.
 * Carved out so the OMP partial-reduction can call it per chunk; serial
 * behaviour is identical to the pre-threading version. */
static T qasum_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx)
{
    T s0 = 0.0Q, s1 = 0.0Q;
    if (incx == 1) {
        ptrdiff_t i = 0;
        for (; i + 1 < n; i += 2) { s0 += fabsq(x[i]); s1 += fabsq(x[i + 1]); }
        if (i < n) s0 += fabsq(x[i]);
    } else {
        for (ptrdiff_t i = 0, ix = 0; i < n; ++i, ix += incx) s0 += fabsq(x[ix]);
    }
    return s0 + s1;
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride X. libquadmath makes each
 * |x| a heavy operation, so the epblas-openblas reference threads ΣX and pulls
 * ~4x ahead at large n; par mirrors it. noinline keeps the parallel-region
 * bookkeeping out of the serial kernel. Reduction order differs from serial
 * (not bit-identical), but within fuzz tolerance for a sum of magnitudes. */
#define QASUM_OMP_MIN 128
#define QASUM_MAX_CPUS 64
__attribute__((noinline)) static bool qasum_omp(ptrdiff_t n, const T *x, T *out)
{
    if (n <= QASUM_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > QASUM_MAX_CPUS) nthreads = QASUM_MAX_CPUS;
    T partial[QASUM_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) partial[tid] = qasum_kernel(hi - lo, x + lo, 1);
    }
    T s = 0.0Q;
    for (ptrdiff_t i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

static T qasum_core(ptrdiff_t n, const T *x, ptrdiff_t incx)
{
    if (n < 1 || incx < 1) return 0.0Q;
#ifdef _OPENMP
    if (incx == 1) {
        T s;
        if (qasum_omp(n, x, &s)) return s;
    }
#endif
    return qasum_kernel(n, x, incx);
}

EPBLAS_FACADE_ASUM(qasum, T, T)
