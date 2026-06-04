/*
 * qrotm — kind16 port of OpenBLAS drotm.  Apply modified Givens rotation.
 *
 * dparam(1) = dflag selects which of three matrix layouts to apply:
 *    dflag = -2.0  -> identity (no-op)
 *    dflag = -1.0  -> H = [[h11 h12],[h21 h22]]
 *    dflag =  0.0  -> H = [[1 h12],[h21 1]]
 *    dflag = +1.0  -> H = [[h11 1],[-1 h22]]
 *
 * Reference: blas/src/qrotm.f.
 *
 * Loop shape: dflag is loop-invariant, so each branch keeps its own
 * for-loop (source-level unswitching).  Findings rule 7 — do NOT collapse
 * to a single inner loop with branches; gcc loses unswitching and emits
 * 3× more store insns.
 */
#include <stddef.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __float128 T;

#define MULTI_THREAD_MINIMAL 10000

static void rotm_neg(ptrdiff_t lo, ptrdiff_t hi, ptrdiff_t incx,
                     T *x, T *y, T h11, T h12, T h21, T h22)
{
    for (ptrdiff_t i = lo; i < hi; i += incx) {
        T w = x[i], z = y[i];
        x[i] = w*h11 + z*h12;
        y[i] = w*h21 + z*h22;
    }
}

static void rotm_zero(ptrdiff_t lo, ptrdiff_t hi, ptrdiff_t incx,
                      T *x, T *y, T h12, T h21)
{
    for (ptrdiff_t i = lo; i < hi; i += incx) {
        T w = x[i], z = y[i];
        x[i] = w + z*h12;
        y[i] = w*h21 + z;
    }
}

static void rotm_pos(ptrdiff_t lo, ptrdiff_t hi, ptrdiff_t incx,
                     T *x, T *y, T h11, T h22)
{
    for (ptrdiff_t i = lo; i < hi; i += incx) {
        T w = x[i], z = y[i];
        x[i] = w*h11 + z;
        y[i] = -w + h22*z;
    }
}

void qrotm_(const int *N, T *x, const int *INCX,
            T *y, const int *INCY, const T *DPARAM)
{
    ptrdiff_t n    = (ptrdiff_t)(*N);
    ptrdiff_t incx = (ptrdiff_t)(*INCX);
    ptrdiff_t incy = (ptrdiff_t)(*INCY);
    T dflag = DPARAM[0];

    if (n <= 0 || dflag + 2.0Q == 0.0Q) return;

    T h11, h12, h21, h22;

    if (incx == incy && incx > 0) {
        ptrdiff_t nsteps = n * incx;

#ifdef _OPENMP
        if (n > MULTI_THREAD_MINIMAL) {
            int nthreads = omp_get_max_threads();
            if (nthreads > 1) {
                if (dflag < 0.0Q) {
                    h11 = DPARAM[1]; h12 = DPARAM[3];
                    h21 = DPARAM[2]; h22 = DPARAM[4];
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int nth = omp_get_num_threads();
                        ptrdiff_t chunk = (n + nth - 1) / nth;
                        ptrdiff_t s = (ptrdiff_t)tid * chunk;
                        ptrdiff_t e = s + chunk;
                        if (e > n) e = n;
                        if (s < e)
                            rotm_neg(s * incx, e * incx, incx, x, y,
                                     h11, h12, h21, h22);
                    }
                } else if (dflag == 0.0Q) {
                    h12 = DPARAM[3]; h21 = DPARAM[2];
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int nth = omp_get_num_threads();
                        ptrdiff_t chunk = (n + nth - 1) / nth;
                        ptrdiff_t s = (ptrdiff_t)tid * chunk;
                        ptrdiff_t e = s + chunk;
                        if (e > n) e = n;
                        if (s < e)
                            rotm_zero(s * incx, e * incx, incx, x, y, h12, h21);
                    }
                } else {
                    h11 = DPARAM[1]; h22 = DPARAM[4];
                    #pragma omp parallel num_threads(nthreads)
                    {
                        int tid = omp_get_thread_num();
                        int nth = omp_get_num_threads();
                        ptrdiff_t chunk = (n + nth - 1) / nth;
                        ptrdiff_t s = (ptrdiff_t)tid * chunk;
                        ptrdiff_t e = s + chunk;
                        if (e > n) e = n;
                        if (s < e)
                            rotm_pos(s * incx, e * incx, incx, x, y, h11, h22);
                    }
                }
                return;
            }
        }
#endif

        if (dflag < 0.0Q) {
            h11 = DPARAM[1]; h12 = DPARAM[3];
            h21 = DPARAM[2]; h22 = DPARAM[4];
            rotm_neg(0, nsteps, incx, x, y, h11, h12, h21, h22);
        } else if (dflag == 0.0Q) {
            h12 = DPARAM[3]; h21 = DPARAM[2];
            rotm_zero(0, nsteps, incx, x, y, h12, h21);
        } else {
            h11 = DPARAM[1]; h22 = DPARAM[4];
            rotm_pos(0, nsteps, incx, x, y, h11, h22);
        }
    } else {
        ptrdiff_t kx = 0, ky = 0;
        if (incx < 0) kx = -(n - 1) * incx;
        if (incy < 0) ky = -(n - 1) * incy;
        if (dflag < 0.0Q) {
            h11 = DPARAM[1]; h12 = DPARAM[3];
            h21 = DPARAM[2]; h22 = DPARAM[4];
            for (ptrdiff_t i = 0; i < n; ++i) {
                T w = x[kx], z = y[ky];
                x[kx] = w*h11 + z*h12;
                y[ky] = w*h21 + z*h22;
                kx += incx; ky += incy;
            }
        } else if (dflag == 0.0Q) {
            h12 = DPARAM[3]; h21 = DPARAM[2];
            for (ptrdiff_t i = 0; i < n; ++i) {
                T w = x[kx], z = y[ky];
                x[kx] = w + z*h12;
                y[ky] = w*h21 + z;
                kx += incx; ky += incy;
            }
        } else {
            h11 = DPARAM[1]; h22 = DPARAM[4];
            for (ptrdiff_t i = 0; i < n; ++i) {
                T w = x[kx], z = y[ky];
                x[kx] = w*h11 + z;
                y[ky] = -w + h22*z;
                kx += incx; ky += incy;
            }
        }
    }
}
