/*
 * eger — kind10 (REAL(KIND=10)) general rank-1 update.
 *   A := alpha · x · yᵀ + A    where A is M×N
 *
 * Bandwidth-bound (~1 flop per A element). The structural payoff is
 * OMP-over-j (each column independent) + restrict-based alias info.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define EGER_OMP_MIN 64

typedef long double T;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

void eger_(
    const int *m_, const int *n_,
    const T *alpha_,
    const T *restrict x, const int *incx_,
    const T *restrict y, const int *incy_,
    T *restrict a, const int *lda_)
{
    const ptrdiff_t M = *m_, N = *n_;
    const ptrdiff_t incx = *incx_, incy = *incy_, lda = *lda_;
    const T alpha = *alpha_;
    const T zero = 0.0L;

    if (M == 0 || N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= EGER_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
#else
        const ptrdiff_t use_omp = 0;
#endif
        /* C-source branch on use_omp to dodge Add-16 outlining tax. */
#define EGER_BODY                                                            \
        for (ptrdiff_t j = 0; j < N; ++j) {                                        \
            const T yj = y[j];                                               \
            if (yj != zero) {                                                \
                const T t = alpha * yj;                                      \
                T *aj = &A_(0, j);                                           \
                for (ptrdiff_t i = 0; i < M; ++i) aj[i] += t * x[i];               \
            }                                                                \
        }
        if (use_omp) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            EGER_BODY
        } else {
            EGER_BODY
        }
#undef EGER_BODY
    } else {
        /* Strided x/y. Columns of A are disjoint, so OMP-over-j is race-free
         * and bit-exact (each column's inner loop runs in one thread, same
         * order as serial). jy is recomputed as jy0 + j*incy — arithmetically
         * identical to the carried `jy += incy` but parallelizable. */
        const ptrdiff_t jy0 = (incy < 0) ? -(N - 1) * incy : 0;
        const ptrdiff_t ix0 = (incx < 0) ? -(M - 1) * incx : 0;
#ifdef _OPENMP
        const ptrdiff_t use_omp = (N >= EGER_OMP_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
#else
        const ptrdiff_t use_omp = 0;
#endif
#define EGER_STRIDED_BODY                                                    \
        for (ptrdiff_t j = 0; j < N; ++j) {                                  \
            const T yj = y[jy0 + j * incy];                                  \
            if (yj != zero) {                                                \
                const T t = alpha * yj;                                      \
                T *aj = &A_(0, j);                                           \
                if (incx == 1) {                                             \
                    for (ptrdiff_t i = 0; i < M; ++i) aj[i] += t * x[i];     \
                } else {                                                     \
                    ptrdiff_t ix = ix0;                                      \
                    for (ptrdiff_t i = 0; i < M; ++i) {                      \
                        aj[i] += t * x[ix];                                  \
                        ix += incx;                                          \
                    }                                                        \
                }                                                            \
            }                                                                \
        }
        if (use_omp) {
#ifdef _OPENMP
            #pragma omp parallel for schedule(static)
#endif
            EGER_STRIDED_BODY
        } else {
            EGER_STRIDED_BODY
        }
#undef EGER_STRIDED_BODY
    }
}

#undef A_
