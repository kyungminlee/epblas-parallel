/*
 * xher — kind16 complex Hermitian rank-1 update.
 *   A := alpha · x · xᴴ + A    (alpha real, diag stays real)
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define XHER_OMP_MIN 64

typedef __complex128 TC;
typedef __float128   TR;

static inline char up(char c) {
    return (char)toupper((unsigned char)c);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xher_core(
    char uplo,
    ptrdiff_t N,
    const TR *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    TC *restrict a, ptrdiff_t lda)
{
    const TR alpha = *alpha_;
    const TR rzero = 0.0Q;
    const TC czero = 0.0Q + 0.0Qi;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == rzero) return;

    if (incx == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= XHER_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (ptrdiff_t j = 0; j < N; ++j) {
            const TC xj = x[j];
            if (xj != czero) {
                const TC t = alpha * conjq(xj);
                TC *aj = &A_(0, j);
                if (UPLO == 'L') {
                    for (ptrdiff_t i = j + 1; i < N; ++i) aj[i] += t * x[i];
                    aj[j] = crealq(aj[j]) + crealq(t * x[j]);
                } else {
                    for (ptrdiff_t i = 0; i < j; ++i) aj[i] += t * x[i];
                    aj[j] = crealq(aj[j]) + crealq(t * x[j]);
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        for (ptrdiff_t j = 0; j < N; ++j) {
            const TC xj = x[kx + (ptrdiff_t)j * incx];
            if (xj != czero) {
                const TC t = alpha * conjq(xj);
                if (UPLO == 'L') {
                    for (ptrdiff_t i = j + 1; i < N; ++i) A_(i, j) += t * x[kx + (ptrdiff_t)i * incx];
                    A_(j, j) = crealq(A_(j, j)) + crealq(t * x[kx + (ptrdiff_t)j * incx]);
                } else {
                    for (ptrdiff_t i = 0; i < j; ++i) A_(i, j) += t * x[kx + (ptrdiff_t)i * incx];
                    A_(j, j) = crealq(A_(j, j)) + crealq(t * x[kx + (ptrdiff_t)j * incx]);
                }
            }
        }
    }
}


EPBLAS_FACADE_SYR(xher, TR, TC)

#undef A_
