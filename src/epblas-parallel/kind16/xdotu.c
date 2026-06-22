/* xdotu — kind16 complex: returns Σ X·Y. */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"
typedef __complex128 T;

/* Σ x·y over a logical range; 2-accumulator unroll on the unit-stride path.
 * Carved out so the OMP partial-reduction can call it per chunk; serial
 * behaviour is unchanged. */
static T xdotu_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx, const T *y, ptrdiff_t incy)
{
    T s = 0;
    if (incx == 1 && incy == 1) {
        T s0 = (T)0.0Q, s1 = (T)0.0Q;
        ptrdiff_t i = 0;
        for (; i + 1 < n; i += 2) {
            s0 += x[i    ] * y[i    ];
            s1 += x[i + 1] * y[i + 1];
        }
        s += s0 + s1;
        for (; i < n; ++i) s += x[i] * y[i];
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) { s += x[ix] * y[iy]; ix += incx; iy += incy; }
    }
    return s;
}

#ifdef _OPENMP
/* Threaded partial-reduction for large unit-stride X·Y. Manual complex
 * partial[] (omp reduction doesn't accept `__complex128`). See qasum.c. */
#define XDOTU_OMP_MIN 128
#define XDOTU_MAX_CPUS 64
__attribute__((noinline)) static bool xdotu_omp(ptrdiff_t n, const T *x, const T *y, T *out)
{
    if (n <= XDOTU_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > XDOTU_MAX_CPUS) nthreads = XDOTU_MAX_CPUS;
    T partial[XDOTU_MAX_CPUS] = {0};
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) partial[tid] = xdotu_kernel(hi - lo, x + lo, 1, y + lo, 1);
    }
    T s = 0;
    for (ptrdiff_t i = 0; i < nthreads; ++i) s += partial[i];
    *out = s;
    return 1;
}
#endif

static T xdotu_core(ptrdiff_t n, const T *x, ptrdiff_t incx,
                    const T *y, ptrdiff_t incy)
{
    T s = 0;
    if (n <= 0) return s;
#ifdef _OPENMP
    if (incx == 1 && incy == 1) {
        if (xdotu_omp(n, x, y, &s)) return s;
    }
#endif
    return xdotu_kernel(n, x, incx, y, incy);
}

EPBLAS_FACADE_DOT(xdotu, T, T)
