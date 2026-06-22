/* easum — kind10 real: returns Σ |X|. */
#include <math.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include <stddef.h>
#endif
#include "../common/epblas_facade.h"
typedef long double T;

/* Σ|x| over a contiguous logical range. 6-accumulator unroll matches NETLIB
 * DASUM and masks the ~3-cycle x87 fadd latency. */
static T easum_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx)
{
    T s0 = 0.0L, s1 = 0.0L;
    if (incx == 1) {
        T s2 = 0.0L, s3 = 0.0L, s4 = 0.0L, s5 = 0.0L;
        ptrdiff_t i = 0;
        for (; i + 5 < n; i += 6) {
            s0 += fabsl(x[i    ]);
            s1 += fabsl(x[i + 1]);
            s2 += fabsl(x[i + 2]);
            s3 += fabsl(x[i + 3]);
            s4 += fabsl(x[i + 4]);
            s5 += fabsl(x[i + 5]);
        }
        s0 += s1 + s2 + s3 + s4 + s5;
        s1 = 0.0L;
        for (; i < n; ++i) s0 += fabsl(x[i]);
    } else {
        for (ptrdiff_t i = 0, ix = 0; i < n; ++i, ix += incx) s0 += fabsl(x[ix]);
    }
    return s0 + s1;
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride X. ob threads ΣX at
 * n>10000; par mirrors it so it doesn't trail ob 2-3x at n≥16384. Carved
 * noinline so the parallel-region bookkeeping does not pressure the serial
 * kernel's x87 allocation. Reduction order differs from serial → not
 * bit-identical, but within fuzz tolerance for a sum of magnitudes. */
#define EASUM_OMP_MIN 10000
#define EASUM_MAX_CPUS 64
__attribute__((noinline)) static ptrdiff_t easum_omp(ptrdiff_t n, const T *x, T *out)
{
    if (n <= EASUM_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > EASUM_MAX_CPUS) nthreads = EASUM_MAX_CPUS;
    T partial[EASUM_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) partial[tid] = easum_kernel(hi - lo, x + lo, 1);
    }
    T s = 0.0L;
    for (ptrdiff_t i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

static T easum_core(ptrdiff_t n, const T *x, ptrdiff_t incx)
{
    if (n < 1 || incx < 1) return 0.0L;
#ifdef _OPENMP
    if (incx == 1) {
        T s;
        if (easum_omp(n, x, &s)) return s;
    }
#endif
    return easum_kernel(n, x, incx);
}

EPBLAS_FACADE_ASUM(easum, T, T)
