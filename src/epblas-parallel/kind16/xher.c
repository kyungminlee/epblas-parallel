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

#define XHER_OMP_MIN 64

typedef __complex128 TC;
typedef __float128   TR;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void xher_(
    const char *uplo,
    const int *n_,
    const TR *alpha_,
    const TC *restrict x, const int *incx_,
    TC *restrict a, const int *lda_,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, lda = *lda_;
    const TR alpha = *alpha_;
    const TR rzero = 0.0Q;
    const TC czero = 0.0Q + 0.0Qi;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == rzero) return;

    if (incx == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= XHER_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const TC xj = x[j];
            if (xj != czero) {
                const TC t = alpha * conjq(xj);
                TC *aj = &A_(0, j);
                if (UPLO == 'L') {
                    for (int i = j + 1; i < N; ++i) aj[i] += t * x[i];
                    aj[j] = crealq(aj[j]) + crealq(t * x[j]);
                } else {
                    for (int i = 0; i < j; ++i) aj[i] += t * x[i];
                    aj[j] = crealq(aj[j]) + crealq(t * x[j]);
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        for (int j = 0; j < N; ++j) {
            const TC xj = x[kx + j * incx];
            if (xj != czero) {
                const TC t = alpha * conjq(xj);
                if (UPLO == 'L') {
                    for (int i = j + 1; i < N; ++i) A_(i, j) += t * x[kx + i * incx];
                    A_(j, j) = crealq(A_(j, j)) + crealq(t * x[kx + j * incx]);
                } else {
                    for (int i = 0; i < j; ++i) A_(i, j) += t * x[kx + i * incx];
                    A_(j, j) = crealq(A_(j, j)) + crealq(t * x[kx + j * incx]);
                }
            }
        }
    }
}

#undef A_
