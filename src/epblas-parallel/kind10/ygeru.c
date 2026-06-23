/*
 * ygeru — kind10 complex unconjugated rank-1 update.
 *   A := alpha · x · yᵀ + A    where A is M×N
 */

#include <stddef.h>
#include "../common/epblas_facade.h"
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* RECALIBRATED 2026-06-07 (was 64): stale libgomp-era break-even; iomp5 hot-team
 * reuse lets this O(M*N) complex unconjugated rank-1 update thread from N=24.
 * Measured par4/par1 (taskset 0-3, min-of-10, M=N): N=24 0.45, N=32 0.37,
 * N=64 0.29, N=128 0.26. It actually wins from N=16 (0.77) but 24 keeps a
 * uniform floor with the Hermitian rank updates that lose below it.
 * Bit-exact (relerr 0). Uniform across the y* rank-update family. */
#define YGERU_OMP_MIN 24

typedef _Complex long double TC;
static const TC ZERO = 0.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

static void ygeru_core(
    ptrdiff_t m, ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    const TC *restrict y, ptrdiff_t incy,
    TC *restrict a, ptrdiff_t lda)
{
    const TC alpha = *alpha_;

    if (m == 0 || n == 0 || alpha == ZERO) return;

    if (incx == 1 && incy == 1) {
        const bool use_omp = (n >= YGERU_OMP_MIN && blas_omp_should_thread());
        /* C-source branch on use_omp (Add-16). */
#define YGERU_BODY                                                           \
        for (ptrdiff_t j = 0; j < n; ++j) {                                        \
            const TC yj = y[j];                                               \
            if (yj != ZERO) {                                                \
                const TC t = alpha * yj;                                      \
                TC *aj = &A_(0, j);                                           \
                for (ptrdiff_t i = 0; i < m; ++i) aj[i] += t * x[i];               \
            }                                                                \
        }
        if (use_omp) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            YGERU_BODY
        } else {
            YGERU_BODY
        }
#undef YGERU_BODY
    } else {
        /* Strided x/y. Columns of A are disjoint → OMP-over-j race-free and
         * bit-exact (jy recomputed as jy0 + j*incy, same as carried add). */
        const ptrdiff_t jy0 = (incy < 0) ? -(n - 1) * incy : 0;
        const ptrdiff_t ix0 = (incx < 0) ? -(m - 1) * incx : 0;
        const bool use_omp = (n >= YGERU_OMP_MIN && blas_omp_should_thread());
#define YGERU_STRIDED_BODY                                                   \
        for (ptrdiff_t j = 0; j < n; ++j) {                                  \
            const TC yj = y[jy0 + j * incy];                                  \
            if (yj != ZERO) {                                                \
                const TC t = alpha * yj;                                      \
                ptrdiff_t ix = ix0;                                          \
                TC *aj = &A_(0, j);                                           \
                for (ptrdiff_t i = 0; i < m; ++i) {                          \
                    aj[i] += t * x[ix];                                      \
                    ix += incx;                                              \
                }                                                            \
            }                                                                \
        }
        if (use_omp) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            YGERU_STRIDED_BODY
        } else {
            YGERU_STRIDED_BODY
        }
#undef YGERU_STRIDED_BODY
    }
}

EPBLAS_FACADE_GER(ygeru, TC)

#undef A_
