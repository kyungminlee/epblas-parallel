/*
 * esyr2 — kind10 (REAL(KIND=10)) symmetric rank-2 update.
 *   A := alpha · x · yᵀ + alpha · y · xᵀ + A    (only UPLO triangle touched)
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#endif

#define ESYR2_OMP_MIN 64

typedef long double TR;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

static void esyr2_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    const TR *restrict y, ptrdiff_t incy,
    TR *restrict a, ptrdiff_t lda)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0L;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        const bool use_omp = (n >= ESYR2_OMP_MIN && blas_omp_max_threads() > 1);
        /* Branch on use_omp at C source level — `#pragma omp parallel for
         * if(use_omp)` outlines unconditionally; at OMP=1 the GOMP_parallel
         * + omp_get_* overhead is a visible fraction of the per-call cost
         * for this small kernel. See Addendum 16. */
#define ESYR2_BODY                                                              \
        for (ptrdiff_t j = 0; j < n; ++j) {                                           \
            const TR xj = x[j], yj = y[j];                                       \
            if (xj != zero || yj != zero) {                                     \
                const TR tx = alpha * yj;                                        \
                const TR ty = alpha * xj;                                        \
                TR *aj = &A_(0, j);                                              \
                if (UPLO == 'L') {                                              \
                    for (ptrdiff_t i = j; i < n; ++i) aj[i] += x[i] * tx + y[i] * ty; \
                } else {                                                        \
                    for (ptrdiff_t i = 0; i <= j; ++i) aj[i] += x[i] * tx + y[i] * ty;\
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
        /* General-stride fallback — hoist the matrix column to aj[i] and
         * walk the strided vectors with running indices (Class-B fix). */
        const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        ptrdiff_t jx = kx, jy = ky;
        for (ptrdiff_t j = 0; j < n; ++j) {
            const TR xj = x[jx];
            const TR yj = y[jy];
            if (xj != zero || yj != zero) {
                const TR tx = alpha * yj;
                const TR ty = alpha * xj;
                TR *aj = &A_(0, j);
                if (UPLO == 'L') {
                    ptrdiff_t ix = jx, iy = jy;
                    for (ptrdiff_t i = j; i < n; ++i) {
                        aj[i] += x[ix] * tx + y[iy] * ty;
                        ix += incx; iy += incy;
                    }
                } else {
                    ptrdiff_t ix = kx, iy = ky;
                    for (ptrdiff_t i = 0; i <= j; ++i) {
                        aj[i] += x[ix] * tx + y[iy] * ty;
                        ix += incx; iy += incy;
                    }
                }
            }
            jx += incx; jy += incy;
        }
    }
}

EPBLAS_FACADE_SYR2(esyr2, TR)

#undef A_
