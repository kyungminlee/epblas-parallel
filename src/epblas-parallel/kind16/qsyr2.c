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

typedef __float128 T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void qsyr2_core(
    char uplo,
    ptrdiff_t N,
    const T *alpha_,
    const T *restrict x, ptrdiff_t incx,
    const T *restrict y, ptrdiff_t incy,
    T *restrict a, ptrdiff_t lda)
{
    const T alpha = *alpha_;
    const T zero = 0.0Q;
    const char UPLO = blas_up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (N >= QSYR2_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j) {
            const T xj = x[j], yj = y[j];
            if (xj != zero || yj != zero) {
                const T tx = alpha * yj;
                const T ty = alpha * xj;
                T *aj = &A_(0, j);
                if (UPLO == 'L') for (ptrdiff_t i = j; i < N; ++i) aj[i] += x[i] * tx + y[i] * ty;
                else             for (ptrdiff_t i = 0; i <= j; ++i) aj[i] += x[i] * tx + y[i] * ty;
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        for (ptrdiff_t j = 0; j < N; ++j) {
            const T xj = x[kx + j * incx];
            const T yj = y[ky + j * incy];
            if (xj != zero || yj != zero) {
                const T tx = alpha * yj;
                const T ty = alpha * xj;
                if (UPLO == 'L') {
                    for (ptrdiff_t i = j; i < N; ++i)
                        A_(i, j) += x[kx + i * incx] * tx + y[ky + i * incy] * ty;
                } else {
                    for (ptrdiff_t i = 0; i <= j; ++i)
                        A_(i, j) += x[kx + i * incx] * tx + y[ky + i * incy] * ty;
                }
            }
        }
    }
}


EPBLAS_FACADE_SYR2(qsyr2, T)

#undef A_
