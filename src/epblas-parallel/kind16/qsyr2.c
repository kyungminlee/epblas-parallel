/*
 * qsyr2 — kind16 (__float128) symmetric rank-2 update.
 *   A := alpha · x · yᵀ + alpha · y · xᵀ + A
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

#define QSYR2_OMP_MIN 64

typedef __float128 TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void qsyr2_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    const TR *restrict y, ptrdiff_t incy,
    TR *restrict a, ptrdiff_t lda)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0Q;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (n >= QSYR2_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR xj = x[j], yj = y[j];
            if (xj != zero || yj != zero) {
                const TR tx = alpha * yj;
                const TR ty = alpha * xj;
                TR *aj = &A_(0, j);
                if (UPLO == 'L') for (ptrdiff_t i = j; i < n; ++i) aj[i] += x[i] * tx + y[i] * ty;
                else             for (ptrdiff_t i = 0; i <= j; ++i) aj[i] += x[i] * tx + y[i] * ty;
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR xj = x[kx + j * incx];
            const TR yj = y[ky + j * incy];
            if (xj != zero || yj != zero) {
                const TR tx = alpha * yj;
                const TR ty = alpha * xj;
                if (UPLO == 'L') {
                    for (ptrdiff_t i = j; i < n; ++i)
                        A_(i, j) += x[kx + i * incx] * tx + y[ky + i * incy] * ty;
                } else {
                    for (ptrdiff_t i = 0; i <= j; ++i)
                        A_(i, j) += x[kx + i * incx] * tx + y[ky + i * incy] * ty;
                }
            }
        }
    }
}


EPBLAS_FACADE_SYR2(qsyr2, TR)

#undef A_
