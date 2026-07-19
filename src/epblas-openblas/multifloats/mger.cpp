/*
 * mger — kind10 port of OpenBLAS dger.  Rank-1 update.
 *
 *   A := alpha * x * y^T + A      (A is M x N)
 *
 * Mirror of OpenBLAS interface/ger.c + kernel/generic/ger.c:
 *   - Quick return if (m==0 || n==0 || alpha==0).
 *   - For each column j of A:  a[:,j] += (alpha * y[j]) * x[:]
 *
 * OpenMP: each column update is independent (writes a disjoint slice of
 * A); parallelize the outer J loop.  Threshold mirrors OpenBLAS's m*n
 * cutoff (~2048 * threshold). Mirrors ger_thread.c's incx!=1 buffer-copy:
 * when threading and incx != 1, x is copied once to a unit-stride buffer
 * so each thread's inner loop runs at stride 1.
 *
 * Fortran ABI:
 *   subroutine mger(m, n, alpha, x, incx, y, incy, a, lda)
 */

#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#include "mblas_tuning.h"
#define MULTI_THREAD_MINIMAL MBLAS_MT_MIN_L2_MN

#define A_(i, j)  a[(size_t)(j) * (size_t)lda + (size_t)(i)]

extern "C" void mger_(const int *M, const int *N, const T *ALPHA,
           const T *x, const int *INCX,
           const T *y, const int *INCY,
           T *a, const int *LDA)
{
    ptrdiff_t m    = (ptrdiff_t)(*M);
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    ptrdiff_t lda  = (ptrdiff_t)(*LDA);
    T alpha = *ALPHA;

    if (m == 0 || n == 0 || alpha == 0.0) return;

    if (incx < 0) x -= (m - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

#ifdef _OPENMP
    int nthreads = 1;
    if (m * n > MULTI_THREAD_MINIMAL) nthreads = omp_get_max_threads();
#else
    int nthreads = 1;
#endif

    const T *xp = x;
    T *x_buf = NULL;
    if (incx != 1 && nthreads > 1) {
        x_buf = (T *)malloc((size_t)m * sizeof(T));
        if (x_buf) {
            ptrdiff_t ix = 0;
            for (ptrdiff_t i = 0; i < m; ++i) { x_buf[i] = x[ix]; ix += incx; }
            xp = x_buf;
        }
    }

    if (xp == x && incx != 1) {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) schedule(static) if(nthreads > 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            T t = alpha * y[j * incy];
            if (t == 0.0) continue;
            T *aj = &A_(0, j);
            ptrdiff_t ix = 0;
            for (ptrdiff_t i = 0; i < m; ++i) {
                aj[i] += t * x[ix];
                ix += incx;
            }
        }
    } else {
#ifdef _OPENMP
        #pragma omp parallel for num_threads(nthreads) schedule(static) if(nthreads > 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            T t = alpha * y[j * incy];
            if (t == 0.0) continue;
            T *aj = &A_(0, j);
            for (ptrdiff_t i = 0; i < m; ++i) aj[i] += t * xp[i];
        }
    }

    if (x_buf) free(x_buf);
}

#undef A_
