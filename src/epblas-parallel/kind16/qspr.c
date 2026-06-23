/*
 * qspr — kind16 (__float128) symmetric packed rank-1 update.
 *   A := alpha*x*x^T + A
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

#define QSPR_OMP_MIN 64

typedef __float128 TR;


void qspr_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    TR *restrict ap)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0Q;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1) {
        /* Columns are independent in packed storage when accessed via kk(j). */
        if (UPLO == 'U') {
#ifdef _OPENMP
            const bool use_omp = (n >= QSPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[j] != zero) {
                    const TR tmp = alpha * x[j];
                    const ptrdiff_t kk = (j * (j + 1)) / 2;
                    for (ptrdiff_t i = 0; i <= j; ++i) ap[kk + i] += x[i] * tmp;
                }
            }
        } else {
#ifdef _OPENMP
            const bool use_omp = (n >= QSPR_OMP_MIN && blas_omp_max_threads() > 1);
            #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[j] != zero) {
                    const TR tmp = alpha * x[j];
                    /* kk0 = sum_{l=0}^{j-1}(N-l) = j*N - j*(j-1)/2 */
                    const ptrdiff_t kk = j * n - (j * (j - 1)) / 2;
                    for (ptrdiff_t i = j; i < n; ++i) ap[kk + (i - j)] += x[i] * tmp;
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t kk = 0;
        if (UPLO == 'U') {
            ptrdiff_t jx = kx;
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero) {
                    const TR tmp = alpha * x[jx];
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t k = kk; k < kk + j + 1; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += j + 1;
            }
        } else {
            ptrdiff_t jx = kx;
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero) {
                    const TR tmp = alpha * x[jx];
                    ptrdiff_t ix = jx;
                    for (ptrdiff_t k = kk; k < kk + n - j; ++k) {
                        ap[k] += x[ix] * tmp;
                        ix += incx;
                    }
                }
                jx += incx;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR(qspr, TR, TR)
