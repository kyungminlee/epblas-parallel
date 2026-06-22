/*
 * xher2 — kind16 complex Hermitian rank-2 update.
 *   A := alpha · x · yᴴ + conj(alpha) · y · xᴴ + A
 * alpha complex. Diagonal stays real.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

#define XHER2_OMP_MIN 64

typedef __complex128 T;

static inline char up(char c) {
    return (char)toupper((unsigned char)c);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xher2_core(
    char uplo,
    ptrdiff_t N,
    const T *alpha_,
    const T *restrict x, ptrdiff_t incx,
    const T *restrict y, ptrdiff_t incy,
    T *restrict a, ptrdiff_t lda)
{
    const T alpha = *alpha_;
    const T zero = 0.0Q + 0.0Qi;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= XHER2_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < N; ++j) {
            const T xj = x[j], yj = y[j];
            if (xj != zero || yj != zero) {
                const T temp1 = alpha * conjq(yj);
                const T temp2 = conjq(alpha * xj);
                T *aj = &A_(0, j);
                if (UPLO == 'L') {
                    for (ptrdiff_t i = j + 1; i < N; ++i) aj[i] += x[i] * temp1 + y[i] * temp2;
                    aj[j] = crealq(aj[j]) + crealq(x[j] * temp1 + y[j] * temp2);
                } else {
                    for (ptrdiff_t i = 0; i < j; ++i) aj[i] += x[i] * temp1 + y[i] * temp2;
                    aj[j] = crealq(aj[j]) + crealq(x[j] * temp1 + y[j] * temp2);
                }
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(N - 1) * incy : 0;
        for (ptrdiff_t j = 0; j < N; ++j) {
            const T xj = x[kx + (ptrdiff_t)j * incx];
            const T yj = y[ky + (ptrdiff_t)j * incy];
            if (xj != zero || yj != zero) {
                const T temp1 = alpha * conjq(yj);
                const T temp2 = conjq(alpha * xj);
                if (UPLO == 'L') {
                    for (ptrdiff_t i = j + 1; i < N; ++i)
                        A_(i, j) += x[kx + (ptrdiff_t)i * incx] * temp1 + y[ky + (ptrdiff_t)i * incy] * temp2;
                    A_(j, j) = crealq(A_(j, j)) + crealq(xj * temp1 + yj * temp2);
                } else {
                    for (ptrdiff_t i = 0; i < j; ++i)
                        A_(i, j) += x[kx + (ptrdiff_t)i * incx] * temp1 + y[ky + (ptrdiff_t)i * incy] * temp2;
                    A_(j, j) = crealq(A_(j, j)) + crealq(xj * temp1 + yj * temp2);
                }
            }
        }
    }
}


EPBLAS_FACADE_SYR2(xher2, T)

#undef A_
