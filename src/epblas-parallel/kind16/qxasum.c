/* qxasum — kind16: Σ (|re(X)| + |im(X)|) for complex X. */
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
typedef __complex128 TC;
typedef __float128 R;

/* Σ(|re|+|im|) over a logical range. Carved out so the OMP partial-reduction
 * can call it per chunk; serial behaviour is unchanged. */
static R qxasum_kernel(ptrdiff_t n, const TC *x, ptrdiff_t incx)
{
    R s = 0.0Q;
    if (incx == 1) {
        for (ptrdiff_t i = 0; i < n; ++i) s += fabsq(__real__ x[i]) + fabsq(__imag__ x[i]);
    } else {
        for (ptrdiff_t i = 0, ix = 0; i < n; ++i, ix += incx)
            s += fabsq(__real__ x[ix]) + fabsq(__imag__ x[ix]);
    }
    return s;
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride X — see qasum.c. */
#define QXASUM_OMP_MIN 128
#define QXASUM_MAX_CPUS 64
__attribute__((noinline)) static bool qxasum_omp(ptrdiff_t n, const TC *x, R *out)
{
    if (n <= QXASUM_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > QXASUM_MAX_CPUS) nthreads = QXASUM_MAX_CPUS;
    R partial[QXASUM_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) partial[tid] = qxasum_kernel(hi - lo, x + lo, 1);
    }
    R s = 0.0Q;
    for (ptrdiff_t i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

static R qxasum_core(ptrdiff_t n, const TC *x, ptrdiff_t incx)
{
    if (n < 1 || incx < 1) return 0.0Q;
#ifdef _OPENMP
    if (incx == 1) {
        R s;
        if (qxasum_omp(n, x, &s)) return s;
    }
#endif
    return qxasum_kernel(n, x, incx);
}

EPBLAS_FACADE_ASUM(qxasum, R, TC)
