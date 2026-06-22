/*
 * ygemv — kind10 complex (`_Complex long double`) general matrix-vector.
 *
 *   y := alpha · A · x + beta · y            (TRANS='N')   A is M×N
 *   y := alpha · Aᵀ · x + beta · y           (TRANS='T')   A is M×N, y is N
 *   y := alpha · Aᴴ · x + beta · y           (TRANS='C')   A is M×N, y is N
 *
 * Reference ZGEMV with J-axis unroll by 2 on the N path (halves y
 * memory traffic) and K-axis unroll by 2 on the T/C dot-product paths
 * (two independent fadd chains hide x87 latency on the cmul-accumulate
 * pattern).
 *
 * Partitioning strategy: thread owns a slice of the output vector y.
 * For TRANS='N' that's rows of A; for TRANS='T'/'C' that's columns of A.
 * All A accesses inside a thread's slice are column-major stride-1.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif
#include "../common/epblas_facade.h"

/* RECALIBRATED 2026-06-07 (was 64): stale libgomp-era break-even; iomp5 hot-team
 * reuse lets this dense complex fp80 matvec thread from N=32 (matching its sibling
 * yhemv). Measured par4/par1 (taskset 0-3, min-of-10, M=N): NoTrans (M-gated)
 * N=32 0.67, N=48 0.53; Trans/Conj (N-gated) N=32 0.66/0.48, N=48 0.37/0.35.
 * NoTrans is the binding path (N=24 only ties; Trans/Conj already win at 24), so
 * the shared floor is set to 32 where all three win solidly. Bit-exact (relerr 0). */
#define YGEMV_OMP_MIN 32

typedef _Complex long double T;


static inline T cconj(T z) { return ~z; }

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* External linkage: ytrsv calls ygemv_core directly (the §1.3 cross-call
 * retarget) so the trailing GEMV bypasses the by-ref facade. */
void ygemv_core(
    char trans,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    const char TR = blas_up(trans);

    if (M == 0 || N == 0) return;

    const ptrdiff_t leny = (TR == 'N') ? M : N;

    /* β-scale y. */
    if (beta != ONE) {
        if (incy == 1) {
            if (beta == ZERO) for (ptrdiff_t i = 0; i < leny; ++i) y[i] = ZERO;
            else              for (ptrdiff_t i = 0; i < leny; ++i) y[i] *= beta;
        } else {
            ptrdiff_t iy = (incy < 0) ? -(leny - 1) * incy : 0;
            for (ptrdiff_t i = 0; i < leny; ++i) {
                if (beta == ZERO) y[iy] = ZERO;
                else              y[iy] *= beta;
                iy += incy;
            }
        }
    }

    if (alpha == ZERO) return;

    if (TR == 'N') {
        if (incx == 1 && incy == 1) {
#ifdef _OPENMP
            const bool use_omp = (M >= YGEMV_OMP_MIN && blas_omp_should_thread());
#else
            const bool use_omp = 0;
#endif
            /* Branch on use_omp in C source — `if(use_omp)` pragma clause
             * still outlines the body into a `._omp_fn` function and pays
             * GOMP_parallel + omp_get_* overhead per call (Addendum 16).
             *
             * OMP=1 single-column inner: complex long-double cmadd expands
             * to ~4 fmuls + 4 fadds, consuming most of the x87 stack.
             * J-unroll-by-2 (2 cmuls per iter) hits stack pressure and
             * gcc spills column-pointer a1, *re-loading* it each fmul.
             * The single-column form is what gfortran emits internally
             * for kind10 complex and runs faster at OMP=1 (no spill).
             *
             * OMP path J-unrolls by 2 anyway: under parallel the
             * bottleneck is memory bandwidth (y RMW dominates), and
             * halving y traffic outweighs the spill (Add-34 measured
             * scaling 2.83x → 3.4x). */
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel
                {
                    const ptrdiff_t tid = omp_get_thread_num();
                    const ptrdiff_t nt  = omp_get_num_threads();
                    const ptrdiff_t i_lo = blas_part_bound(M, tid, nt);
                    const ptrdiff_t i_hi = blas_part_bound(M, tid + 1, nt);
                    ptrdiff_t j = 0;
                    for (; j + 1 < N; j += 2) {
                        const T t0 = alpha * x[j];
                        const T t1 = alpha * x[j + 1];
                        const T *a0 = &A_(0, j);
                        const T *a1 = &A_(0, j + 1);
                        for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                            y[i] = (y[i] + t0 * a0[i]) + t1 * a1[i];
                        }
                    }
                    for (; j < N; ++j) {
                        const T t = alpha * x[j];
                        const T *aj = &A_(0, j);
                        for (ptrdiff_t i = i_lo; i < i_hi; ++i) y[i] += t * aj[i];
                    }
                }
#endif
            } else {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    const T t = alpha * x[j];
                    const T *aj = &A_(0, j);
                    for (ptrdiff_t i = 0; i < M; ++i) y[i] += t * aj[i];
                }
            }
        } else if (incy == 1) {
            /* incx != 1, incy == 1: single-column inner — same reason
             * as fast path above. Complex cmul + J-unroll spills a1 on
             * x87 stack; single-column matches migrated. */
            ptrdiff_t jx = (incx < 0) ? -(N - 1) * incx : 0;
            for (ptrdiff_t j = 0; j < N; ++j) {
                const T t = alpha * x[jx];
                const T *aj = &A_(0, j);
                for (ptrdiff_t i = 0; i < M; ++i) y[i] += t * aj[i];
                jx += incx;
            }
        } else {
            /* incy != 1: strided y writes — replicate gfortran auto
             * J-unroll-by-2 to halve y traffic on the strided path.
             * Thread over disjoint output-row slices [i_lo,i_hi): each y[iy]
             * is written by one thread in the same j-order as serial →
             * race-free and bit-exact. jx is thread-local (recomputed). */
            const ptrdiff_t iy0 = (incy < 0) ? -(M - 1) * incy : 0;
#ifdef _OPENMP
            const bool use_omp = (M >= YGEMV_OMP_MIN && blas_omp_should_thread());
#else
            const bool use_omp = 0;
#endif
#define YGEMV_N_STRIDED_BODY(i_lo, i_hi) do {                                \
            ptrdiff_t jx = (incx < 0) ? -(N - 1) * incx : 0;                 \
            ptrdiff_t j = 0;                                                 \
            for (; j + 1 < N; j += 2) {                                      \
                const T t0 = alpha * x[jx];                                  \
                const T t1 = alpha * x[jx + incx];                          \
                /* Decompose the loop-invariant scalars into real/imag      \
                 * long-double locals and stride the complex A/y columns    \
                 * as 2n reals. _Complex t0/t1 otherwise force gfortran's    \
                 * J-unroll-by-2 constants onto the 8-slot x87 stack, which  \
                 * spills and reloads them every iteration (13 fp-loads vs   \
                 * gfortran's 10); the scalar form keeps them register-      \
                 * resident, restoring parity. Same association as the       \
                 * complex form — (y + t0·a0) + t1·a1 — so bit-identical. */  \
                const long double t0r = __real__ t0, t0i = __imag__ t0;      \
                const long double t1r = __real__ t1, t1i = __imag__ t1;      \
                const long double *A0 = (const long double *)&A_(0, j);      \
                const long double *A1 = (const long double *)&A_(0, j + 1);  \
                ptrdiff_t iy = iy0 + (i_lo) * incy;                          \
                for (ptrdiff_t i = (i_lo); i < (i_hi); ++i) {                \
                    const long double a0r = A0[2 * i], a0i = A0[2 * i + 1];  \
                    const long double a1r = A1[2 * i], a1i = A1[2 * i + 1];  \
                    long double *Y = (long double *)&y[iy];                  \
                    Y[0] = (Y[0] + (t0r * a0r - t0i * a0i))                  \
                                 + (t1r * a1r - t1i * a1i);                  \
                    Y[1] = (Y[1] + (t0r * a0i + t0i * a0r))                  \
                                 + (t1r * a1i + t1i * a1r);                  \
                    iy += incy;                                             \
                }                                                           \
                jx += 2 * incx;                                             \
            }                                                               \
            for (; j < N; ++j) {                                            \
                const T xj = x[jx];                                         \
                if (xj != ZERO) {                                          \
                    const T t = alpha * xj;                                 \
                    ptrdiff_t iy = iy0 + (i_lo) * incy;                     \
                    for (ptrdiff_t i = (i_lo); i < (i_hi); ++i) {           \
                        y[iy] += t * A_(i, j);                              \
                        iy += incy;                                         \
                    }                                                       \
                }                                                           \
                jx += incx;                                                 \
            }                                                               \
        } while (0)
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel
                {
                    const ptrdiff_t tid = omp_get_thread_num();
                    const ptrdiff_t nt  = omp_get_num_threads();
                    const ptrdiff_t i_lo = blas_part_bound(M, tid, nt);
                    const ptrdiff_t i_hi = blas_part_bound(M, tid + 1, nt);
                    YGEMV_N_STRIDED_BODY(i_lo, i_hi);
                }
#endif
            } else {
                YGEMV_N_STRIDED_BODY(0, M);
            }
#undef YGEMV_N_STRIDED_BODY
        }
    } else {
        /* TRANS='T' or 'C': y[j] += α · Σ_i A(i,j)[, conj] · x(i).
         * Single-acc dot product (complex inner-loop is x87-stack-heavy;
         * K-unroll with split accumulators regressed on similar paths
         * in ygemm — keep single accumulator). */
        const bool conj_a = (TR == 'C');
        if (incx == 1 && incy == 1) {
#ifdef _OPENMP
            const bool use_omp = (N >= YGEMV_OMP_MIN && blas_omp_should_thread());
#else
            const bool use_omp = 0;
#endif
            /* Branch on use_omp in C source — `if(use_omp)` pragma clause
             * still outlines (see Addendum 16). */
#define YGEMV_T_BODY                                                         \
            for (ptrdiff_t j = 0; j < N; ++j) {                                    \
                const T *aj = &A_(0, j);                                     \
                T s = ZERO;                                                  \
                if (conj_a) {                                                \
                    for (ptrdiff_t i = 0; i < M; ++i) s += cconj(aj[i]) * x[i];    \
                } else {                                                     \
                    for (ptrdiff_t i = 0; i < M; ++i) s += aj[i] * x[i];           \
                }                                                            \
                y[j] += alpha * s;                                           \
            }
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel for schedule(static)
#endif
                YGEMV_T_BODY
            } else {
                YGEMV_T_BODY
            }
#undef YGEMV_T_BODY
        } else {
            /* Strided x/y. Each output y[jy(j)] is disjoint across j → OMP-over-j
             * is race-free and bit-exact (jy recomputed as jy0 + j*incy). */
            const ptrdiff_t jy0 = (incy < 0) ? -(N - 1) * incy : 0;
            const ptrdiff_t ix0 = (incx < 0) ? -(M - 1) * incx : 0;
#ifdef _OPENMP
            const bool use_omp = (N >= YGEMV_OMP_MIN && blas_omp_should_thread());
#else
            const bool use_omp = 0;
#endif
#define YGEMV_T_STRIDED_BODY                                                  \
            for (ptrdiff_t j = 0; j < N; ++j) {                              \
                T s = ZERO;                                                  \
                ptrdiff_t ix = ix0;                                          \
                for (ptrdiff_t i = 0; i < M; ++i) {                          \
                    s += (conj_a ? cconj(A_(i, j)) : A_(i, j)) * x[ix];      \
                    ix += incx;                                              \
                }                                                            \
                y[jy0 + j * incy] += alpha * s;                              \
            }
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel for schedule(static)
#endif
                YGEMV_T_STRIDED_BODY
            } else {
                YGEMV_T_STRIDED_BODY
            }
#undef YGEMV_T_STRIDED_BODY
        }
    }
}

EPBLAS_FACADE_GEMV(ygemv, T)

#undef A_
