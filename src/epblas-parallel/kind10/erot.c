#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
/* erot — kind10 real Givens rotation. */
typedef long double T;

/* Unit-stride kernel, shared by the serial entry and the per-thread OMP slices.
 * Load both elements into locals before storing: x and y are distinct `T*` the
 * compiler can't prove non-aliasing for, so reading them inline across the
 * interleaved writes forces an 80-bit reload each iteration. Hoisting to locals
 * (the epblas-openblas shape) keeps them on the x87 register stack. Same ops in
 * the same order — bit-identical. */
static void erot_unit(ptrdiff_t n, T c, T s, T *x, T *y)
{
    for (ptrdiff_t i = 0; i < n; ++i) {
        T xi = x[i], yi = y[i];
        x[i] = c * xi + s * yi;
        y[i] = c * yi - s * xi;
    }
}

#ifdef _OPENMP
/* Threaded real Givens — same cache-bandwidth rationale as eaxpy_omp (see
 * eaxpy.c). Compute-bound (4 mul + 2 add per element); measured proto4/par1
 * ~1.11 at N=256, 0.78 at 384, <1.0 to 4M (~0.60), no upper bound. Break-even
 * ~N=330; 512 keeps margin. */
#define EROT_OMP_MIN 512
static int erot_omp(ptrdiff_t n, T c, T s, T *x, T *y)
{
    if (n <= EROT_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) erot_unit(hi - lo, c, s, x + lo, y + lo);
    }
    return 1;
}
#endif

void erot_(const int *n_, T *x, const int *incx_, T *y, const int *incy_,
           const T *c_, const T *s_)
{
    const ptrdiff_t n = *n_, incx = *incx_, incy = *incy_;
    const T c = *c_, s = *s_;
    if (n <= 0) return;
    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        if (erot_omp(n, c, s, x, y)) return;
#endif
        erot_unit(n, c, s, x, y);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        ptrdiff_t iy = (incy < 0) ? (-n + 1) * incy : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            T xi = x[ix], yi = y[iy];
            x[ix] = c * xi + s * yi;
            y[iy] = c * yi - s * xi;
            ix += incx; iy += incy;
        }
    }
}
