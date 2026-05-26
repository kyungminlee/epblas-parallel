/*
 * yher2 — kind10 complex Hermitian rank-2 update.
 *   A := alpha · x · yᴴ + conj(alpha) · y · xᴴ + A
 * alpha is complex. Diagonal of A stays real on output.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YHER2_OMP_MIN 64

typedef _Complex long double T;
static const T ZERO = 0.0L + 0.0Li;
static inline T cconj(T z) { return ~z; }

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void yher2_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    const T *restrict y, const int *incy_,
    T *restrict a, const int *lda_,
    size_t uplo_len)
{
    (void)uplo_len;
    const int N = *n_;
    const int incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == ZERO) return;

    const T alpha_conj = cconj(alpha);

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= YHER2_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const T xj = x[j], yj = y[j];
            if (xj != ZERO || yj != ZERO) {
                /* Per Netlib ZHER2:
                 *   TEMP1 = ALPHA * conj(Y(J))
                 *   TEMP2 = conj(ALPHA * X(J))
                 *   A(i,j) += X(i)*TEMP1 + Y(i)*TEMP2
                 * Diagonal entry stays real. */
                const T temp1 = alpha * cconj(yj);
                const T temp2 = cconj(alpha * xj);
                T *aj = &A_(0, j);
                if (UPLO == 'L') {
                    for (int i = j + 1; i < N; ++i) aj[i] += x[i] * temp1 + y[i] * temp2;
                    aj[j] = __real__ aj[j] + __real__ (x[j] * temp1 + y[j] * temp2);
                } else {
                    for (int i = 0; i < j; ++i) aj[i] += x[i] * temp1 + y[i] * temp2;
                    aj[j] = __real__ aj[j] + __real__ (x[j] * temp1 + y[j] * temp2);
                }
            }
        }
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            const T xj = x[kx + j * incx];
            const T yj = y[ky + j * incy];
            if (xj != ZERO || yj != ZERO) {
                const T temp1 = alpha * cconj(yj);
                const T temp2 = cconj(alpha * xj);
                if (UPLO == 'L') {
                    for (int i = j + 1; i < N; ++i)
                        A_(i, j) += x[kx + i * incx] * temp1 + y[ky + i * incy] * temp2;
                    A_(j, j) = __real__ A_(j, j) + __real__ (xj * temp1 + yj * temp2);
                } else {
                    for (int i = 0; i < j; ++i)
                        A_(i, j) += x[kx + i * incx] * temp1 + y[ky + i * incy] * temp2;
                    A_(j, j) = __real__ A_(j, j) + __real__ (xj * temp1 + yj * temp2);
                }
            }
        }
        (void)alpha_conj;
    }
}

#undef A_
