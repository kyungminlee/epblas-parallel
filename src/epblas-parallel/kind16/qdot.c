/* qdot — kind16 real: returns Σ X·Y. */
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __float128 T;

/* Σ x·y over a logical range; 5-accumulator unroll on the unit-stride path.
 * Carved out so the OMP partial-reduction can call it per chunk; serial
 * behaviour is unchanged. */
static T qdot_kernel(int n, const T *x, int incx, const T *y, int incy)
{
    T s0 = 0.0Q, s1 = 0.0Q;
    if (incx == 1 && incy == 1) {
        T s2 = 0.0Q, s3 = 0.0Q, s4 = 0.0Q;
        int i = 0;
        for (; i + 4 < n; i += 5) {
            s0 += x[i    ] * y[i    ];
            s1 += x[i + 1] * y[i + 1];
            s2 += x[i + 2] * y[i + 2];
            s3 += x[i + 3] * y[i + 3];
            s4 += x[i + 4] * y[i + 4];
        }
        s0 += s2 + s3 + s4;
        for (; i < n; ++i) s0 += x[i] * y[i];
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { s0 += x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s0 + s1;
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride X·Y — see qasum.c. */
#define QDOT_OMP_MIN 128
#define QDOT_MAX_CPUS 64
__attribute__((noinline)) static int qdot_omp(int n, const T *x, const T *y, T *out)
{
    if (n <= QDOT_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > QDOT_MAX_CPUS) nthreads = QDOT_MAX_CPUS;
    T partial[QDOT_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) partial[tid] = qdot_kernel(hi - lo, x + lo, 1, y + lo, 1);
    }
    T s = 0.0Q;
    for (int i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

T qdot_(const int *n_, const T *x, const int *incx_,
        const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    if (n <= 0) return 0.0Q;
#ifdef _OPENMP
    if (incx == 1 && incy == 1) {
        T s;
        if (qdot_omp(n, x, y, &s)) return s;
    }
#endif
    return qdot_kernel(n, x, incx, y, incy);
}
