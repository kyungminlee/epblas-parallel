/*
 * esyr — kind10 (REAL(KIND=10)) symmetric rank-1 update.
 *   A := alpha · x · xᵀ + A     (only UPLO triangle of A touched)
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ESYR_OMP_MIN 64

typedef long double T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void esyr_(
    const char *uplo,
    const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    T *restrict a, const int *lda_,
    size_t uplo_len)
{
    (void)uplo_len;
    const ptrdiff_t N = *n_;
    const ptrdiff_t incx = *incx_, lda = *lda_;
    const T alpha = *alpha_;
    const T zero = 0.0L;
    const char UPLO = up(uplo);

    if (N == 0 || alpha == zero) return;

    if (incx == 1) {
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= ESYR_OMP_MIN && blas_omp_max_threads() > 1);
#else
        const ptrdiff_t use_omp = 0;
#endif
        /* Branch on use_omp at C source level — `#pragma omp parallel for
         * if(use_omp)` outlines unconditionally; at OMP=1 the GOMP_parallel
         * + omp_get_* overhead is a visible fraction of the per-call cost
         * for this small kernel. See Addendum 16. */
#define ESYR_BODY                                                            \
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
            /* schedule(static, 1): per-column work is linear in (N-j) (L)
             * or j (U). Round-robin distribution balances heavy and
             * light columns across threads — Rule 49. */
            #pragma omp parallel for schedule(static, 1)
#endif
            ESYR_BODY
        } else {
            ESYR_BODY
        }
#undef ESYR_BODY
    } else {
        /* General-stride fallback — hoist the matrix column to aj[i] and
         * walk the strided vector with a running index (Class-B fix,
         * memory project_ptrdiff_conversion_regressors). */
        const ptrdiff_t kx = (incx < 0) ? -(N - 1) * incx : 0;
        ptrdiff_t jx = kx;
        for (ptrdiff_t j = 0; j < N; ++j) {
            const T xj = x[jx];
            if (xj != zero) {
                const T t = alpha * xj;
                T *aj = &A_(0, j);
                if (UPLO == 'L') {
                    ptrdiff_t ix = jx;
                    for (ptrdiff_t i = j; i < N; ++i) { aj[i] += t * x[ix]; ix += incx; }
                } else {
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t i = 0; i <= j; ++i) { aj[i] += t * x[ix]; ix += incx; }
                }
            }
            jx += incx;
        }
    }
}

#undef A_
