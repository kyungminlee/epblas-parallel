/*
 * ygerc — kind10 complex conjugated rank-1 update.
 *   A := alpha · x · yᴴ + A    where A is M×N
 */

#include <stddef.h>
#include "../common/epblas_facade.h"
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* RECALIBRATED 2026-06-07 (was 64): stale libgomp-era break-even; iomp5 hot-team
 * reuse lets this O(M*N) complex conjugated rank-1 update thread from N=24.
 * Measured par4/par1 (taskset 0-3, min-of-10, M=N): N=24 0.46, N=32 0.38,
 * N=64 0.28, N=128 0.26. It actually wins from N=16 (0.78) but 24 keeps a
 * uniform floor with the Hermitian rank updates that lose below it.
 * Bit-exact (relerr 0). Uniform across the y* rank-update family. */
#define YGERC_OMP_MIN 24

typedef _Complex long double TC;
static const TC ZERO = 0.0L + 0.0Li;
static inline TC cconj(TC z) { return ~z; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

static void ygerc_core(
    ptrdiff_t m, ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    const TC *restrict y, ptrdiff_t incy,
    TC *restrict a, ptrdiff_t lda)
{
    const TC alpha = *alpha_;

    if (m == 0 || n == 0 || alpha == ZERO) return;

    if (incx == 1 && incy == 1) {
        const bool use_omp = (n >= YGERC_OMP_MIN && blas_omp_should_thread());
        /* C-source branch on use_omp (Add-16). */
#define YGERC_BODY                                                           \
        for (ptrdiff_t j = 0; j < n; ++j) {                                        \
            const TC yj = cconj(y[j]);                                        \
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
            YGERC_BODY
        } else {
            YGERC_BODY
        }
#undef YGERC_BODY
    } else {
        /* Strided x/y. Columns of A are disjoint → OMP-over-j race-free and
         * bit-exact (jy recomputed as jy0 + j*incy, same as carried add). */
        const ptrdiff_t jy0 = (incy < 0) ? -(n - 1) * incy : 0;
        const ptrdiff_t ix0 = (incx < 0) ? -(m - 1) * incx : 0;
        const bool use_omp = (n >= YGERC_OMP_MIN && blas_omp_should_thread());
#define YGERC_STRIDED_BODY                                                   \
        for (ptrdiff_t j = 0; j < n; ++j) {                                  \
            const TC yj = cconj(y[jy0 + j * incy]);                           \
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
            YGERC_STRIDED_BODY
        } else {
            YGERC_STRIDED_BODY
        }
#undef YGERC_STRIDED_BODY
    }
}

EPBLAS_FACADE_GER(ygerc, TC)

#undef A_
