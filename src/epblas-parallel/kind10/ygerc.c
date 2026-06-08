/*
 * ygerc — kind10 complex conjugated rank-1 update.
 *   A := alpha · x · yᴴ + A    where A is M×N
 */

#include <stddef.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* RECALIBRATED 2026-06-07 (was 64): stale libgomp-era break-even; iomp5 hot-team
 * reuse lets this O(M*N) complex conjugated rank-1 update thread from N=24.
 * Measured par4/par1 (taskset 0-3, min-of-10, M=N): N=24 0.46, N=32 0.38,
 * N=64 0.28, N=128 0.26. It actually wins from N=16 (0.78) but 24 keeps a
 * uniform floor with the Hermitian rank updates that lose below it.
 * Bit-exact (relerr 0). Uniform across the y* rank-update family. */
#define YGERC_OMP_MIN 24

typedef _Complex long double T;
static const T ZERO = 0.0L + 0.0Li;
static inline T cconj(T z) { return ~z; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void ygerc_(
    const int *m_, const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    const T *restrict y, const int *incy_,
    T *restrict a, const int *lda_)
{
    const ptrdiff_t M = *m_, N = *n_;
    const ptrdiff_t incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;

    if (M == 0 || N == 0 || alpha == ZERO) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= YGERC_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
#else
        const ptrdiff_t use_omp = 0;
#endif
        /* C-source branch on use_omp (Add-16). */
#define YGERC_BODY                                                           \
        for (ptrdiff_t j = 0; j < N; ++j) {                                        \
            const T yj = cconj(y[j]);                                        \
            if (yj != ZERO) {                                                \
                const T t = alpha * yj;                                      \
                T *aj = &A_(0, j);                                           \
                for (ptrdiff_t i = 0; i < M; ++i) aj[i] += t * x[i];               \
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
        const ptrdiff_t jy0 = (incy < 0) ? -(N - 1) * incy : 0;
        const ptrdiff_t ix0 = (incx < 0) ? -(M - 1) * incx : 0;
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= YGERC_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
#else
        const ptrdiff_t use_omp = 0;
#endif
#define YGERC_STRIDED_BODY                                                   \
        for (ptrdiff_t j = 0; j < N; ++j) {                                  \
            const T yj = cconj(y[jy0 + j * incy]);                           \
            if (yj != ZERO) {                                                \
                const T t = alpha * yj;                                      \
                ptrdiff_t ix = ix0;                                          \
                T *aj = &A_(0, j);                                           \
                for (ptrdiff_t i = 0; i < M; ++i) {                          \
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

#undef A_
