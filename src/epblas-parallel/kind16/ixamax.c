/* ixamax — kind16 complex: 1-based argmax(|re|+|im|). */
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
typedef __complex128 T;
typedef __float128 R;

/* Scan a contiguous unit-stride range [0,n); return the 0-based index of the
 * first element with maximal |re|+|im| and store that magnitude in *bv_out.
 * Strictly-greater update keeps the lowest index on ties. */
static ptrdiff_t ixamax_kernel(ptrdiff_t n, const T *x, R *bv_out)
{
    ptrdiff_t best = 0;
    R bv = fabsq(__real__ x[0]) + fabsq(__imag__ x[0]);
    for (ptrdiff_t i = 1; i < n; ++i) {
        R v = fabsq(__real__ x[i]) + fabsq(__imag__ x[i]);
        if (v > bv) { bv = v; best = i; }
    }
    *bv_out = bv;
    return best;
}

#ifdef _OPENMP
/* Threaded argmax for large unit-stride X. Ascending-tid serial reduce with
 * strictly-greater keeps the lowest global index on ties, matching the serial
 * scan. See qasum.c for threshold/noinline rationale. */
#define IXAMAX_OMP_MIN 128
#define IXAMAX_MAX_CPUS 64
__attribute__((noinline)) static bool ixamax_omp(ptrdiff_t n, const T *x, ptrdiff_t *out)
{
    if (n <= IXAMAX_OMP_MIN || !blas_omp_should_thread())
        return 0;
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads > IXAMAX_MAX_CPUS) nthreads = IXAMAX_MAX_CPUS;
    R pval[IXAMAX_MAX_CPUS];
    ptrdiff_t pidx[IXAMAX_MAX_CPUS];   /* global 0-based index of each thread's local best */
    for (ptrdiff_t i = 0; i < nthreads; ++i) { pval[i] = -1.0Q; pidx[i] = -1; }
    #pragma omp parallel num_threads(nthreads)
    {
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
        ptrdiff_t lo = blas_part_bound(n, tid, nth);
        ptrdiff_t hi = blas_part_bound(n, tid + 1, nth);
        if (lo < hi) {
            R bv;
            ptrdiff_t li = ixamax_kernel(hi - lo, x + lo, &bv);
            pval[tid] = bv;
            pidx[tid] = lo + li;
        }
    }
    R bv = -1.0Q;
    ptrdiff_t best = 0;
    for (ptrdiff_t i = 0; i < nthreads; ++i) {
        if (pidx[i] >= 0 && pval[i] > bv) { bv = pval[i]; best = pidx[i]; }
    }
    *out = best + 1;   /* 1-based */
    return 1;
}
#endif

static ptrdiff_t ixamax_core(ptrdiff_t n, const T *x, ptrdiff_t incx)
{
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;
#ifdef _OPENMP
    if (incx == 1) {
        ptrdiff_t idx;
        if (ixamax_omp(n, x, &idx)) return idx;
    }
#endif
    ptrdiff_t best = 1;
    R bv = fabsq(__real__ x[0]) + fabsq(__imag__ x[0]);
    ptrdiff_t ix = incx;
    for (ptrdiff_t i = 2; i <= n; ++i) {
        R v = fabsq(__real__ x[ix]) + fabsq(__imag__ x[ix]);
        if (v > bv) { bv = v; best = i; }
        ix += incx;
    }
    return best;
}

EPBLAS_FACADE_IAMAX(ixamax, T)
