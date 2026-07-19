/*
 * eaxpy — kind10 (REAL(KIND=10) / 80-bit long double) port of OpenBLAS daxpy.
 *
 * Y := alpha * X + Y
 *
 * Faithful structure mirror of OpenBLAS:
 *   - Outer wrapper (NAME) mirrors interface/axpy.c — arg validation,
 *     n<=0 / alpha==0 shortcut, the (incx==0 && incy==0) special case,
 *     negative-stride pointer pre-shift, multi-thread cutoff dispatch.
 *   - Kernel (axpy_kernel) mirrors kernel/x86_64/daxpy.c — the unit-
 *     stride branch with 8-way unroll for the n&-8 head and a scalar
 *     tail, and the general 4-way-unrolled strided branch.
 *
 * Differences from OpenBLAS daxpy:
 *   - No SIMD. x86_64 has no AVX path for 80-bit long double.
 *   - No micro-arch dispatch: a single C kernel.
 *   - Multi-thread split uses OpenMP (when compiled with -fopenmp)
 *     instead of OpenBLAS's blas_level1_thread infrastructure.
 *
 * Fortran ABI:
 *   subroutine eaxpy(n, alpha, x, incx, y, incy)
 *   - All scalars are pointers (gfortran convention)
 *   - n, incx, incy: default integer == 4 bytes (gfortran default)
 *   - alpha, x, y: REAL(KIND=10)  ↔  long double
 */

#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#include "eblas_tuning.h"

static void axpy_kernel(ptrdiff_t n, T alpha, const T *x, ptrdiff_t incx,
                                              T *y,       ptrdiff_t incy)
{
    if (incx == 1 && incy == 1) {
        /* Mirror OpenBLAS's 8-way unrolled kernel head + scalar tail. */
        ptrdiff_t n1 = n & -8;
        ptrdiff_t i;
        for (i = 0; i < n1; i += 8) {
            y[i + 0] += alpha * x[i + 0];
            y[i + 1] += alpha * x[i + 1];
            y[i + 2] += alpha * x[i + 2];
            y[i + 3] += alpha * x[i + 3];
            y[i + 4] += alpha * x[i + 4];
            y[i + 5] += alpha * x[i + 5];
            y[i + 6] += alpha * x[i + 6];
            y[i + 7] += alpha * x[i + 7];
        }
        for (; i < n; ++i) y[i] += alpha * x[i];
        return;
    }

    /* General strided path — 4-way unroll head + scalar tail.
     * (Mirrors OpenBLAS's kernel/x86_64/daxpy.c lines 119–148.) */
    ptrdiff_t ix = 0, iy = 0;
    ptrdiff_t n1 = n & -4;
    ptrdiff_t i = 0;
    while (i < n1) {
        T m1 = alpha * x[ix];
        T m2 = alpha * x[ix + incx];
        T m3 = alpha * x[ix + 2 * incx];
        T m4 = alpha * x[ix + 3 * incx];
        y[iy]              += m1;
        y[iy + incy]       += m2;
        y[iy + 2 * incy]   += m3;
        y[iy + 3 * incy]   += m4;
        ix += incx * 4;
        iy += incy * 4;
        i += 4;
    }
    while (i < n) {
        y[iy] += alpha * x[ix];
        ix += incx;
        iy += incy;
        ++i;
    }
}

void eaxpy_(const int *N, const T *ALPHA,
            const T *x, const int *INCX,
            T       *y, const int *INCY)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    T alpha = *ALPHA;

    if (n <= 0) return;
    if (alpha == 0.0L) return;

    /* Both-zero-increment scalar shortcut — y[0] += n * alpha * x[0]. */
    if (incx == 0 && incy == 0) {
        *y += (T)n * alpha * (*x);
        return;
    }

    /* OpenBLAS pre-shifts the pointer for negative strides so the
     * kernel can index from 0 with positive walking. */
    if (incx < 0) x -= (n - 1) * incx;
    if (incy < 0) y -= (n - 1) * incy;

#ifdef _OPENMP
    int nthreads = 1;
    if (incx != 0 && incy != 0 && n > MULTI_THREAD_MINIMAL) {
        nthreads = omp_get_max_threads();
    }
    if (nthreads > 1) {
        /* Block-partition the index range — each thread runs an
         * independent axpy_kernel on its slice. */
        #pragma omp parallel num_threads(nthreads)
        {
            int tid  = omp_get_thread_num();
            int nth  = omp_get_num_threads();
            ptrdiff_t chunk = (n + nth - 1) / nth;
            ptrdiff_t start = (ptrdiff_t)tid * chunk;
            ptrdiff_t end   = start + chunk;
            if (end > n) end = n;
            if (start < end) {
                axpy_kernel(end - start, alpha,
                            x + start * incx, incx,
                            y + start * incy, incy);
            }
        }
        return;
    }
#endif
    axpy_kernel(n, alpha, x, incx, y, incy);
}
