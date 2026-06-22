/*
 * qsyr — kind16 (__float128) symmetric rank-1 update.
 *   A := alpha · x · xᵀ + A
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include <quadmath.h>
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../common/epblas_facade.h"

#define QSYR_OMP_MIN 64

typedef __float128 T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void qsyr_core(
    char uplo,
    ptrdiff_t N,
    const T *alpha_,
    const T *restrict x, ptrdiff_t incx,
    T *restrict a, ptrdiff_t lda)
{
    const T alpha = *alpha_;
    const T zero = 0.0Q;
    const char UPLO = blas_up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1) {
        const bool use_omp = (N >= QSYR_OMP_MIN && blas_omp_max_threads() > 1);
        /* Branch on use_omp at C source level — `#pragma omp parallel for
         * if(use_omp)` outlines unconditionally. */
#define QSYR_BODY                                                            \
        for (ptrdiff_t j = 0; j < N; ++j) {                                        \
            const T xj = x[j];                                               \
            if (xj != zero) {                                                \
                const T t = alpha * xj;                                      \
                T *aj = &A_(0, j);                                           \
                if (UPLO == 'L') {                                           \
                    for (ptrdiff_t i = j; i < N; ++i) aj[i] += t * x[i];           \
                } else {                                                     \
                    for (ptrdiff_t i = 0; i <= j; ++i) aj[i] += t * x[i];          \
                }                                                            \
            }                                                                \
        }
        if (use_omp) {
#ifdef _OPENMP
            /* schedule(static, 1): per-column work is linear in (N-j) (L) or
             * j (U). Round-robin balances heavy and light columns; full-storage
             * columns are lda-separated so there is no false sharing. */
            #pragma omp parallel for schedule(static, 1)
#endif
            QSYR_BODY
        } else {
            QSYR_BODY
        }
#undef QSYR_BODY
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        for (ptrdiff_t j = 0; j < N; ++j) {
            const T xj = x[kx + j * incx];
            if (xj != zero) {
                const T t = alpha * xj;
                if (UPLO == 'L') {
                    for (ptrdiff_t i = j; i < N; ++i) A_(i, j) += t * x[kx + i * incx];
                } else {
                    for (ptrdiff_t i = 0; i <= j; ++i) A_(i, j) += t * x[kx + i * incx];
                }
            }
        }
    }
}


EPBLAS_FACADE_SYR(qsyr, T, T)

#undef A_
