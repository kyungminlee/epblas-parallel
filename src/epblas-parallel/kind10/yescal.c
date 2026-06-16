#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
/* yescal — kind10: X := α·X with α real, X complex (CSSCAL/ZDSCAL analog).
 *
 * Treat the complex array as a 2N-long real array of (re, im) pairs.
 * The previous `... * 1.0iL` idiom triggered gcc's complex-multiplication
 * expansion (4 fmul + 2 fadd including products by zero); writing the
 * real and imag scales explicitly produces a tight 2-fmul-per-element
 * inner loop. */
typedef _Complex long double T;
typedef long double R;

/* Unit-stride kernel over the (re,im)-pair view, shared by serial + OMP slices. */
static void yescal_unit(ptrdiff_t n, R alpha, long double *p)
{
    long double *e = p + 2 * (long)n;
    for (; p < e; p += 2) {
        p[0] *= alpha;
        p[1] *= alpha;
    }
}

#ifdef _OPENMP
/* Threaded unit-stride real-alpha complex SCAL — cache-bandwidth rationale as
 * eaxpy_omp (see eaxpy.c); complex element = 2x bytes so the break-even count is
 * lower than the real scal. Threshold set by par4<=ob4 (ob keeps scal serial at
 * small N). Measured under iomp5: par4/ob4 1.23@1024, 1.05@2048, then 1.003@3072
 * and 0.92@4096 — break-even ~3072, stay serial through 2048. */
#define YESCAL_OMP_MIN 2048
static int yescal_omp(ptrdiff_t n, R alpha, long double *base)
{
    if (n <= YESCAL_OMP_MIN || blas_omp_max_threads() <= 1 || omp_in_parallel())
        return 0;
    int nthreads = blas_omp_max_threads();
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num(), nth = omp_get_num_threads();
        ptrdiff_t lo = (ptrdiff_t)((long long)n * tid / nth);
        ptrdiff_t hi = (ptrdiff_t)((long long)n * (tid + 1) / nth);
        if (lo < hi) yescal_unit(hi - lo, alpha, base + 2 * lo);
    }
    return 1;
}
#endif

void yescal_(const int *n_, const R *alpha_, T *x, const int *incx_)
{
    const ptrdiff_t n = *n_, incx = *incx_;
    const R alpha = *alpha_;
    if (n <= 0 || alpha == 1.0L) return;
    long double *base = (long double *)x;
    if (incx == 1) {
#ifdef _OPENMP
        if (yescal_omp(n, alpha, base)) return;
#endif
        yescal_unit(n, alpha, base);
    } else {
        ptrdiff_t ix = (incx < 0) ? (-n + 1) * incx : 0;
        for (ptrdiff_t i = 0; i < n; ++i) {
            base[2*ix]     *= alpha;
            base[2*ix + 1] *= alpha;
            ix += incx;
        }
    }
}
