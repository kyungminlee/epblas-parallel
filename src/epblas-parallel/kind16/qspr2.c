/*
 * qspr2 — kind16 (__float128) symmetric packed rank-2 update.
 *   A := alpha*x*y^T + alpha*y*x^T + A
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define QSPR2_OMP_MIN 64

typedef __float128 TR;


void qspr2_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    const TR *restrict y, ptrdiff_t incy,
    TR *restrict ap)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0Q;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        if (UPLO == 'U') {
#ifdef _OPENMP
            const bool use_omp = (n >= QSPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[j] != zero || y[j] != zero) {
                    const TR t1 = alpha * y[j];
                    const TR t2 = alpha * x[j];
                    const ptrdiff_t kk = (j * (j + 1)) / 2;
                    for (ptrdiff_t i = 0; i <= j; ++i) ap[kk + i] += x[i] * t1 + y[i] * t2;
                }
            }
        } else {
#ifdef _OPENMP
            const bool use_omp = (n >= QSPR2_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[j] != zero || y[j] != zero) {
                    const TR t1 = alpha * y[j];
                    const TR t2 = alpha * x[j];
                    const ptrdiff_t kk = j * n - (j * (j - 1)) / 2;
                    for (ptrdiff_t i = j; i < n; ++i) ap[kk + (i - j)] += x[i] * t1 + y[i] * t2;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        ptrdiff_t kk = 0;
        ptrdiff_t jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TR t1 = alpha * y[jy];
                    const TR t2 = alpha * x[jx];
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
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TR t1 = alpha * y[jy];
                    const TR t2 = alpha * x[jx];
                    ptrdiff_t ix = jx, iy = jy;
                    for (ptrdiff_t k = kk; k < kk + n - j; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR2(qspr2, TR)
