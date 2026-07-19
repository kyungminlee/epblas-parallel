/*
 * ixamax — kind16 port of OpenBLAS izamax.
 *
 * Returns 1-based index of first element with maximum |Re|+|Im|.
 * Bails on n<1 or incx<=0 (returns 0). For n==1 returns 1.
 */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __complex128 C;
typedef __float128 T;

#include "qblas_tuning.h"
#define MULTI_THREAD_MINIMAL QBLAS_MT_MIN_L1

static inline T q_abs(T x) { return __builtin_fabsf128(x); }
static inline T cabs1(C z) {
    const T *p = (const T *)&z;
    return q_abs(p[0]) + q_abs(p[1]);
}

static void iamax_kernel(ptrdiff_t n, const C *x, ptrdiff_t incx,
                         ptrdiff_t *out_idx, T *out_max)
{
    T m = cabs1(x[0]);
    ptrdiff_t imax = 0;
    if (incx == 1) {
        for (ptrdiff_t i = 1; i < n; ++i) {
            T a = cabs1(x[i]);
            if (a > m) { m = a; imax = i; }
        }
    } else {
        for (ptrdiff_t i = 1; i < n; ++i) {
            T a = cabs1(x[i*incx]);
            if (a > m) { m = a; imax = i; }
        }
    }
    *out_idx = imax;
    *out_max = m;
}

int ixamax_(const int *N, const C *x, const int *INCX)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    if (n < 1 || incx <= 0) return 0;
    if (n == 1) return 1;

#ifdef _OPENMP
    if (n > MULTI_THREAD_MINIMAL) {
        int nthreads = omp_get_max_threads();
        if (nthreads > 1) {
            if (nthreads > QBLAS_L1_MAX_THREADS) nthreads = QBLAS_L1_MAX_THREADS;
            ptrdiff_t pidx[QBLAS_L1_MAX_THREADS]; T pmax[QBLAS_L1_MAX_THREADS];
            for (int i = 0; i < QBLAS_L1_MAX_THREADS; ++i) { pidx[i] = -1; pmax[i] = 0.0Q; }
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
                    iamax_kernel(end - start, x + start*incx, incx, &li, &lm);
                    pidx[tid] = start + li;
                    pmax[tid] = lm;
                }
            }
            ptrdiff_t gidx = 0;
            T gmax = 0.0Q;
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
