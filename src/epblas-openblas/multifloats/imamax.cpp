/*
 * imamax — kind10 port of OpenBLAS idamax.
 *
 * Returns 1-based index of first element with maximum absolute value.
 * Bails on n<1 or incx<=0 (returns 0). For n==1 returns 1.
 */
#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#include "mblas_tuning.h"
#define MULTI_THREAD_MINIMAL MBLAS_MT_MIN_L1

static inline T dd_abs(T a) { return multifloats::fabs(a); }

static void iamax_kernel(ptrdiff_t n, const T *x, ptrdiff_t incx,
                         ptrdiff_t *out_idx, T *out_max)
{
    /* 0-based index within this slice; caller converts to 1-based global */
    T m = dd_abs(x[0]);
    ptrdiff_t imax = 0;
    if (incx == 1) {
        for (ptrdiff_t i = 1; i < n; ++i) {
            T a = dd_abs(x[i]);
            if (a > m) { m = a; imax = i; }
        }
    } else {
        for (ptrdiff_t i = 1; i < n; ++i) {
            T a = dd_abs(x[i*incx]);
            if (a > m) { m = a; imax = i; }
        }
    }
    *out_idx = imax;
    *out_max = m;
}

extern "C" int imamax_(const int *N, const T *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);

    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;

#ifdef _OPENMP
    if (n > MULTI_THREAD_MINIMAL) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            if (nthreads > MBLAS_L1_MAX_THREADS) nthreads = MBLAS_L1_MAX_THREADS;
            ptrdiff_t pidx[MBLAS_L1_MAX_THREADS]; T pmax[MBLAS_L1_MAX_THREADS];
            for (int i = 0; i < MBLAS_L1_MAX_THREADS; ++i) { pidx[i] = -1; pmax[i] = 0.0; }
            #pragma omp parallel num_threads(nthreads)
            {
                int tid = omp_get_thread_num();
                int nth = omp_get_num_threads();
                ptrdiff_t chunk = (n + nth - 1) / nth;
                ptrdiff_t start = (ptrdiff_t)tid * chunk;
                ptrdiff_t end   = start + chunk;
                if (end > n) end = n;
                if (start < end) {
                    ptrdiff_t li; T lm;
                    iamax_kernel(end - start, x + start * incx, incx, &li, &lm);
                    pidx[tid] = start + li;
                    pmax[tid] = lm;
                }
            }
            /* First-wins on ties matches the Fortran reference (i>max). */
            ptrdiff_t gidx = 0;
            T gmax = -1.0;  /* will be overwritten by first valid */
            int first = 1;
            for (int i = 0; i < nthreads; ++i) {
                if (pidx[i] < 0) continue;
                if (first || pmax[i] > gmax) {
                    gmax = pmax[i]; gidx = pidx[i]; first = 0;
                }
            }
            return (int)(gidx + 1);
        }
    }
#endif
    ptrdiff_t li; T lm;
    iamax_kernel(n, x, incx, &li, &lm);
    return (int)(li + 1);
}
