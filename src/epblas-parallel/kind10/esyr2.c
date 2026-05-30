/*
 * esyr2 — kind10 (REAL(KIND=10)) symmetric rank-2 update.
 *   A := alpha · x · yᵀ + alpha · y · xᵀ + A    (only UPLO triangle touched)
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ESYR2_OMP_MIN 64

typedef long double T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void esyr2_(
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
    const T zero = 0.0L;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const int use_omp = (N >= ESYR2_OMP_MIN && blas_omp_max_threads() > 1);
#else
        const int use_omp = 0;
#endif
        /* Branch on use_omp at C source level — `#pragma omp parallel for
         * if(use_omp)` outlines unconditionally; at OMP=1 the GOMP_parallel
         * + omp_get_* overhead is a visible fraction of the per-call cost
         * for this small kernel. See Addendum 16. */
#define ESYR2_BODY                                                              \
        for (int j = 0; j < N; ++j) {                                           \
            const T xj = x[j], yj = y[j];                                       \
            if (xj != zero || yj != zero) {                                     \
                const T tx = alpha * yj;                                        \
                const T ty = alpha * xj;                                        \
                T *aj = &A_(0, j);                                              \
                if (UPLO == 'L') {                                              \
                    for (int i = j; i < N; ++i) aj[i] += x[i] * tx + y[i] * ty; \
                } else {                                                        \
                    for (int i = 0; i <= j; ++i) aj[i] += x[i] * tx + y[i] * ty;\
                }                                                               \
            }                                                                   \
        }
        if (use_omp) {
#ifdef _OPENMP
            /* schedule(static, 1): per-column work is linear in (N-j) (L)
             * or j (U). Round-robin distribution balances heavy and light
             * columns across threads — Rule 49. Plain schedule(static)
             * gives thread 0 the heaviest block (2-3× imbalance). */
            #pragma omp parallel for schedule(static, 1)
#endif
            ESYR2_BODY
        } else {
            ESYR2_BODY
        }
#undef ESYR2_BODY
    } else {
        int kx = (incx < 0) ? -(N - 1) * incx : 0;
        int ky = (incy < 0) ? -(N - 1) * incy : 0;
        for (int j = 0; j < N; ++j) {
            const T xj = x[kx + j * incx];
            const T yj = y[ky + j * incy];
            if (xj != zero || yj != zero) {
                const T tx = alpha * yj;
                const T ty = alpha * xj;
                if (UPLO == 'L') {
                    for (int i = j; i < N; ++i)
                        A_(i, j) += x[kx + i * incx] * tx + y[ky + i * incy] * ty;
                } else {
                    for (int i = 0; i <= j; ++i)
                        A_(i, j) += x[kx + i * incx] * tx + y[ky + i * incy] * ty;
                }
            }
        }
    }
}

#undef A_
