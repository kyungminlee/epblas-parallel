/*
 * eger — kind10 (REAL(KIND=10)) general rank-1 update.
 *   A := alpha · x · yᵀ + A    where A is M×N
 *
 * Bandwidth-bound (~1 flop per A element). The structural payoff is
 * OMP-over-j (each column independent) + restrict-based alias info.
 */

#include <stddef.h>
#include <stdlib.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define EGER_OMP_MIN 64

typedef long double T;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* Unit-stride column update  aj[i] += t · x[i]  (one A column, x contiguous).
 * Same shape as eaxpy_unit: per-element x87 op count is fixed at any unroll
 * factor (2 fldt, 1 fmul, 1 faddp, 1 fstpt — no SIMD for fp80), so the only
 * lever is loop-overhead amortization. gfortran's dger runs a 1-way loop yet
 * out-scheduled gcc's 1-way body by ~14% (par tied the openblas clone, both
 * behind netlib); the 8-way head (the epblas-openblas daxpy head) halves the
 * per-element increment/compare/branch cost and recovers it. restrict re-states
 * the aj/x no-alias fact the caller has but a plain helper would lose, so the
 * aj[i] store can't force an x[i] reload. Bit-exact: each aj[i] is independent. */
static void eger_col_unit(ptrdiff_t M, T t, const T *restrict x, T *restrict aj)
{
    const ptrdiff_t m = M % 8;
    for (ptrdiff_t i = 0; i < m; ++i) aj[i] += t * x[i];
    for (ptrdiff_t i = m; i < M; i += 8) {
        aj[i    ] += t * x[i    ];
        aj[i + 1] += t * x[i + 1];
        aj[i + 2] += t * x[i + 2];
        aj[i + 3] += t * x[i + 3];
        aj[i + 4] += t * x[i + 4];
        aj[i + 5] += t * x[i + 5];
        aj[i + 6] += t * x[i + 6];
        aj[i + 7] += t * x[i + 7];
    }
}

static void eger_core(
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *restrict x, ptrdiff_t incx,
    const T *restrict y, ptrdiff_t incy,
    T *restrict a, ptrdiff_t lda)
{
    const T alpha = *alpha_;
    const T zero = 0.0L;

    if (M == 0 || N == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
#ifdef _OPENMP
        const bool use_omp = (N >= EGER_OMP_MIN && blas_omp_should_thread());
#else
        const bool use_omp = 0;
#endif
        /* C-source branch on use_omp to dodge Add-16 outlining tax. */
#define EGER_BODY                                                            \
        for (ptrdiff_t j = 0; j < N; ++j) {                                  \
            const T yj = y[j];                                               \
            if (yj != zero) eger_col_unit(M, alpha * yj, x, &A_(0, j));      \
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
        const bool use_omp = (N >= EGER_OMP_MIN && blas_omp_should_thread());
#else
        const bool use_omp = 0;
#endif
        /* Threaded + strided x: copy x once to a unit-stride buffer so every
         * thread streams x[] at stride 1 (mirrors ob ger_thread.c). Bit-exact —
         * same values copied in order, same per-column accumulation. Serial keeps
         * the strided load (its inner loop isn't on a shared/contended path, and a
         * malloc per call there would only add overhead). */
        const T *xp = x;
        T *x_buf = NULL;
        if (use_omp && incx != 1) {
            x_buf = malloc((size_t)M * sizeof(T));
            if (x_buf) {
                ptrdiff_t ix = ix0;
                for (ptrdiff_t i = 0; i < M; ++i) { x_buf[i] = x[ix]; ix += incx; }
                xp = x_buf;
            }
        }
        const bool x_unit = (incx == 1) || (x_buf != NULL);
#define EGER_STRIDED_BODY                                                    \
        for (ptrdiff_t j = 0; j < N; ++j) {                                  \
            const T yj = y[jy0 + j * incy];                                  \
            if (yj != zero) {                                                \
                const T t = alpha * yj;                                      \
                T *aj = &A_(0, j);                                           \
                if (x_unit) {                                                \
                    eger_col_unit(M, t, xp, aj);                             \
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
        free(x_buf);
    }
}

EPBLAS_FACADE_GER(eger, T)

#undef A_
