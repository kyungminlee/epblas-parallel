/*
 * qspr2 — kind16 (__float128) symmetric packed rank-2 update.
 *   A := alpha*x*y^T + alpha*y*x^T + A
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define QSPR2_OMP_MIN 64

typedef __float128 T;

static inline char up(char c) {
    return (char)toupper((unsigned char)c);
}

void qspr2_core(
    char uplo,
    ptrdiff_t N,
    const T *alpha_,
    const T *restrict x, ptrdiff_t incx,
    const T *restrict y, ptrdiff_t incy,
    T *restrict ap)
{
    const T alpha = *alpha_;
    const T zero = 0.0Q;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const int use_omp = (N >= QSPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (ptrdiff_t j = 0; j < N; ++j) {
                if (x[j] != zero || y[j] != zero) {
                    const T t1 = alpha * y[j];
                    const T t2 = alpha * x[j];
                    const ptrdiff_t kk = (j * (j + 1)) / 2;
                    for (ptrdiff_t i = 0; i <= j; ++i) ap[kk + i] += x[i] * t1 + y[i] * t2;
                }
            }
        } else {
#ifdef _OPENMP
            const int use_omp = (N >= QSPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (ptrdiff_t j = 0; j < N; ++j) {
                if (x[j] != zero || y[j] != zero) {
                    const T t1 = alpha * y[j];
                    const T t2 = alpha * x[j];
                    const ptrdiff_t kk = j * N - (j * (j - 1)) / 2;
                    for (ptrdiff_t i = j; i < N; ++i) ap[kk + (i - j)] += x[i] * t1 + y[i] * t2;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        ptrdiff_t kk = 0;
        ptrdiff_t jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < N; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const T t1 = alpha * y[jy];
                    const T t2 = alpha * x[jx];
                    ptrdiff_t ix = kx, iy = ky;
                    for (ptrdiff_t k = kk; k < kk + j + 1; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            for (ptrdiff_t j = 0; j < N; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const T t1 = alpha * y[jy];
                    const T t2 = alpha * x[jx];
                    ptrdiff_t ix = jx, iy = jy;
                    for (ptrdiff_t k = kk; k < kk + N - j; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += N - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR2(qspr2, T)
