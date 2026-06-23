/*
 * esyr — kind10 (REAL(KIND=10)) symmetric rank-1 update.
 *   A := alpha · x · xᵀ + A     (only UPLO triangle of A touched)
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define ESYR_OMP_MIN 64

typedef long double TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

static void esyr_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    TR *restrict a, ptrdiff_t lda)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0L;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1) {
        const bool use_omp = (n >= ESYR_OMP_MIN && blas_omp_max_threads() > 1);
        /* Branch on use_omp at C source level — `#pragma omp parallel for
         * if(use_omp)` outlines unconditionally; at OMP=1 the GOMP_parallel
         * + omp_get_* overhead is a visible fraction of the per-call cost
         * for this small kernel. See Addendum 16. */
#define ESYR_BODY                                                            \
        for (ptrdiff_t j = 0; j < n; ++j) {                                        \
            const TR xj = x[j];                                               \
            if (xj != zero) {                                                \
                const TR t = alpha * xj;                                      \
                TR *aj = &A_(0, j);                                           \
                if (UPLO == 'L') {                                           \
                    for (ptrdiff_t i = j; i < n; ++i) aj[i] += t * x[i];           \
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
         * walk the strided vector with a running index (Class-B fix). */
        const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t jx = kx;
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR xj = x[jx];
            if (xj != zero) {
                const TR t = alpha * xj;
                TR *aj = &A_(0, j);
                if (UPLO == 'L') {
                    ptrdiff_t ix = jx;
                    for (ptrdiff_t i = j; i < n; ++i) { aj[i] += t * x[ix]; ix += incx; }
                } else {
                    ptrdiff_t ix = kx;
                    for (ptrdiff_t i = 0; i <= j; ++i) { aj[i] += t * x[ix]; ix += incx; }
                }
            }
            jx += incx;
        }
    }
}

EPBLAS_FACADE_SYR(esyr, TR, TR)

#undef A_
