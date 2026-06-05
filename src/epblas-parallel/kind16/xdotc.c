/* xdotc — kind16 complex: returns Σ conj(X)·Y. */
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
typedef __complex128 T;

/* Σ conj(x)·y over a logical range; 2-accumulator unroll on the unit-stride
 * path. Carved out so the OMP partial-reduction can call it per chunk; serial
 * behaviour is unchanged. */
static T xdotc_kernel(int n, const T *x, int incx, const T *y, int incy)
{
    T s = 0;
    if (incx == 1 && incy == 1) {
        T s0 = (T)0.0Q, s1 = (T)0.0Q;
        int i = 0;
        for (; i + 1 < n; i += 2) {
            s0 += ~x[i    ] * y[i    ];
            s1 += ~x[i + 1] * y[i + 1];
        }
        s += s0 + s1;
        for (; i < n; ++i) s += ~x[i] * y[i];
    } else {
        int ix = (incx < 0) ? (-n + 1) * incx : 0;
        int iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (int i = 0; i < n; ++i) { s += ~x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s;
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride conj(X)·Y. A manual complex
 * partial[] array is required — `#pragma omp reduction(+:)` does not accept
 * `__complex128`. See qasum.c for the threshold/noinline rationale. */
#define XDOTC_OMP_MIN 128
#define XDOTC_MAX_CPUS 64
__attribute__((noinline)) static int xdotc_omp(int n, const T *x, const T *y, T *out)
{
    if (n <= XDOTC_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    if (nthreads > XDOTC_MAX_CPUS) nthreads = XDOTC_MAX_CPUS;
    T partial[XDOTC_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
        int lo = (int)((long long)n * tid / nth);
        int hi = (int)((long long)n * (tid + 1) / nth);
        if (lo < hi) partial[tid] = xdotc_kernel(hi - lo, x + lo, 1, y + lo, 1);
    }
    T s = 0;
    for (int i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

T xdotc_(const int *n_, const T *x, const int *incx_,
         const T *y, const int *incy_)
{
    const int n = *n_, incx = *incx_, incy = *incy_;
    T s = 0;
    if (n <= 0) return s;
#ifdef _OPENMP
    if (incx == 1 && incy == 1) {
        if (xdotc_omp(n, x, y, &s)) return s;
    }
#endif
    return xdotc_kernel(n, x, incx, y, incy);
}
