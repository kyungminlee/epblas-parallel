/*
 * egemv — kind10 (REAL(KIND=10) / `long double`) general matrix-vector.
 *
 *   y := alpha · A · x + beta · y          (TRANS='N')   A is M×N
 *   y := alpha · Aᵀ · x + beta · y         (TRANS='T'/'C') A is M×N, y is N
 *
 * Reference DGEMV algorithm. L2 is bandwidth-bound (~1 flop/element),
 * so structural optimization is limited; the overlay's value is
 * OpenMP parallelism (gfortran's reference DGEMV is serial) and
 * restrict-based alias elimination.
 *
 * Partitioning strategy: each thread owns a contiguous slice of the
 * OUTPUT vector y. For TRANS='N' that's rows of A (and y has length M).
 * For TRANS='T'/'C' that's columns of A (and y has length N). All A
 * accesses inside a thread's slice remain column-major stride-1.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#endif
#include "../common/epblas_facade.h"

#define EGEMV_OMP_MIN 64

typedef long double T;


#define A_(i, j)  a[(size_t)(j) * lda + (i)]

/* External linkage: etrsv/etbsv/etpsv call egemv_core directly (the §1.3
 * cross-call retarget) so the trailing GEMV bypasses the by-ref facade. */
void egemv_core(
    char trans,
    ptrdiff_t m, ptrdiff_t n,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict x, ptrdiff_t incx,
    const T *beta_,
    T *restrict y, ptrdiff_t incy)
{
    const T alpha = *alpha_, beta = *beta_;
    char TRANS = blas_up(trans);
    if (TRANS == 'C') TRANS = 'T';

    if (m == 0 || n == 0) return;

    const T zero = 0.0L, one = 1.0L;

    /* Output length: M for TRANS='N', N for TRANS='T'. */
    const ptrdiff_t leny = (TRANS == 'N') ? m : n;

    /* Beta-scale y (or zero) first — independent of A·x update. */
    if (beta != one) {
        if (incy == 1) {
            if (beta == zero) for (ptrdiff_t i = 0; i < leny; ++i) y[i] = zero;
            else              for (ptrdiff_t i = 0; i < leny; ++i) y[i] *= beta;
        } else {
            size_t iy0 = (incy < 0) ? (size_t)(-(leny - 1)) * (size_t)incy : 0;
            (void)iy0;
            ptrdiff_t iy = (incy < 0) ? -(leny - 1) * incy : 0;
            for (ptrdiff_t i = 0; i < leny; ++i) {
                if (beta == zero) y[iy] = zero;
                else              y[iy] *= beta;
                iy += incy;
            }
        }
    }

    if (alpha == zero) return;

    if (TRANS == 'N') {
        /* y[i] += alpha · sum_j A(i,j) · x(j).
         * Stride-1 contiguous case (most common: incx==incy==1).
         * Partition rows of A across threads so each thread updates
         * its own slice of y with no cross-thread writes. */
        if (incx == 1 && incy == 1) {
            /* J-axis unroll by 2: process two columns at a time so each
             * y[i] load+store services both column updates. Matches
             * gfortran's reference DGEMV codegen — halves y memory
             * traffic on the column-AXPY path.
             *
             * Body is a macro because we branch on use_omp in C source
             * to avoid the `#pragma omp parallel if(use_omp)` outlining
             * tax (Addendum 16: outlining costs ~1-3 µs per call even
             * when use_omp=false). Important when called from inside
             * etrsv_blocked's parallel region, where omp_in_parallel
             * forces use_omp=false and we pay the tax N/NB times. */
#define EGEMV_N_BODY(i_lo, i_hi) do {                                       \
                const ptrdiff_t span = (i_hi) - (i_lo);                           \
                ptrdiff_t j = 0;                                                  \
                for (; j + 1 < n; j += 2) {                                 \
                    const T t0 = alpha * x[j];                              \
                    const T t1 = alpha * x[j + 1];                          \
                    char *restrict yp = (char *)(y + (i_lo));               \
                    const char *restrict a0 = (const char *)&A_((i_lo), j); \
                    const char *restrict a1 = (const char *)&A_((i_lo), j + 1);\
                    const size_t end = (size_t)span * sizeof(T);            \
                    for (size_t k = 0; k < end; k += sizeof(T)) {           \
                        T *yk        = (T *)(yp + k);                       \
                        const T *a0k = (const T *)(a0 + k);                 \
                        const T *a1k = (const T *)(a1 + k);                 \
                        *yk = (*yk + t0 * *a0k) + t1 * *a1k;                \
                    }                                                       \
                }                                                           \
                for (; j < n; ++j) {                                        \
                    const T t = alpha * x[j];                               \
                    const T *aj = &A_(0, j);                                \
                    for (ptrdiff_t i = (i_lo); i < (i_hi); ++i) y[i] += t * aj[i];\
                }                                                           \
            } while (0)
            const bool use_omp = (m >= EGEMV_OMP_MIN && blas_omp_should_thread());
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel
                {
                    const ptrdiff_t tid = omp_get_thread_num();
                    const ptrdiff_t nth  = omp_get_num_threads();
                    const ptrdiff_t i_lo = blas_part_bound(m, tid, nth);
                    const ptrdiff_t i_hi = blas_part_bound(m, tid + 1, nth);
                    EGEMV_N_BODY(i_lo, i_hi);
                }
#endif
            } else {
                EGEMV_N_BODY(0, m);
            }
#undef EGEMV_N_BODY
        } else if (incy == 1) {
            /* incx != 1, incy == 1: y access is unit-stride.
             * J-unroll-by-2 — gfortran auto-J-unrolls its strided
             * INCY.EQ.1 branch the same way (see migrated egemv.f.o
             * inner at .380). Without the unroll this branch sits at
             * 0.65x of migrated; with it, parity.
             * Thread by partitioning OUTPUT rows [i_lo,i_hi): each thread
             * owns a disjoint y-slice and walks all columns, so it's
             * race-free and bit-exact. jx is thread-local (recomputed). */
#define EGEMV_N_INCY1_BODY(i_lo, i_hi) do {                                  \
                ptrdiff_t jx = (incx < 0) ? -(n - 1) * incx : 0;            \
                ptrdiff_t j = 0;                                            \
                for (; j + 1 < n; j += 2) {                                 \
                    const T t0 = alpha * x[jx];                             \
                    const T t1 = alpha * x[jx + incx];                      \
                    const T *a0 = &A_(0, j);                                \
                    const T *a1 = &A_(0, j + 1);                            \
                    for (ptrdiff_t i = (i_lo); i < (i_hi); ++i)             \
                        y[i] = (y[i] + t0 * a0[i]) + t1 * a1[i];            \
                    jx += 2 * incx;                                         \
                }                                                           \
                for (; j < n; ++j) {                                        \
                    const T t = alpha * x[jx];                              \
                    const T *aj = &A_(0, j);                                \
                    for (ptrdiff_t i = (i_lo); i < (i_hi); ++i) y[i] += t * aj[i]; \
                    jx += incx;                                             \
                }                                                           \
            } while (0)
            const bool use_omp = (m >= EGEMV_OMP_MIN && blas_omp_should_thread());
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel
                {
                    const ptrdiff_t tid = omp_get_thread_num();
                    const ptrdiff_t nth  = omp_get_num_threads();
                    const ptrdiff_t i_lo = blas_part_bound(m, tid, nth);
                    const ptrdiff_t i_hi = blas_part_bound(m, tid + 1, nth);
                    EGEMV_N_INCY1_BODY(i_lo, i_hi);
                }
#endif
            } else {
                EGEMV_N_INCY1_BODY(0, m);
            }
#undef EGEMV_N_INCY1_BODY
        } else {
            /* incy != 1: strided y. Same J-unroll-by-2 as Branch B —
             * gfortran's strided-y inner is also J-unrolled (migrated
             * .380 inner is 8 insns: 2 fmul + 2 fadd + 1 y load+store
             * + 1 y stride advance per pair of columns). Halves y
             * memory traffic, which is the dominant cost.
             * Thread by partitioning OUTPUT rows [i_lo,i_hi): each thread's
             * y-slice is disjoint, iy starts at iy0 + i_lo*incy. Bit-exact. */
            const ptrdiff_t iy0 = (incy < 0) ? -(m - 1) * incy : 0;
#define EGEMV_N_STRIDED_BODY(i_lo, i_hi) do {                                \
                ptrdiff_t jx = (incx < 0) ? -(n - 1) * incx : 0;            \
                ptrdiff_t j = 0;                                            \
                for (; j + 1 < n; j += 2) {                                 \
                    const T t0 = alpha * x[jx];                             \
                    const T t1 = alpha * x[jx + incx];                      \
                    const T *a0 = &A_(0, j);                                \
                    const T *a1 = &A_(0, j + 1);                            \
                    ptrdiff_t iy = iy0 + (i_lo) * incy;                     \
                    for (ptrdiff_t i = (i_lo); i < (i_hi); ++i) {           \
                        y[iy] = (y[iy] + t0 * a0[i]) + t1 * a1[i];          \
                        iy += incy;                                         \
                    }                                                       \
                    jx += 2 * incx;                                         \
                }                                                           \
                for (; j < n; ++j) {                                        \
                    const T xj = x[jx];                                     \
                    if (xj != zero) {                                       \
                        const T t = alpha * xj;                             \
                        ptrdiff_t iy = iy0 + (i_lo) * incy;                 \
                        for (ptrdiff_t i = (i_lo); i < (i_hi); ++i) {       \
                            y[iy] += t * A_(i, j);                          \
                            iy += incy;                                     \
                        }                                                   \
                    }                                                       \
                    jx += incx;                                             \
                }                                                           \
            } while (0)
            const bool use_omp = (m >= EGEMV_OMP_MIN && blas_omp_should_thread());
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel
                {
                    const ptrdiff_t tid = omp_get_thread_num();
                    const ptrdiff_t nth  = omp_get_num_threads();
                    const ptrdiff_t i_lo = blas_part_bound(m, tid, nth);
                    const ptrdiff_t i_hi = blas_part_bound(m, tid + 1, nth);
                    EGEMV_N_STRIDED_BODY(i_lo, i_hi);
                }
#endif
            } else {
                EGEMV_N_STRIDED_BODY(0, m);
            }
#undef EGEMV_N_STRIDED_BODY
        }
    } else {  /* TRANS = 'T' or 'C' (real: same): y[j] += alpha · sum_i A(i,j) · x(i) */
        if (incx == 1 && incy == 1) {
            /* C-source branch on use_omp to avoid the outlining tax
             * (Addendum 16). Body is a macro to share between the
             * parallel and serial paths. */
#define EGEMV_T_BODY do {                                                   \
                for (ptrdiff_t j = 0; j < n; ++j) {                               \
                    const T *aj = &A_(0, j);                                \
                    T s0 = zero, s1 = zero;                                 \
                    ptrdiff_t i = 0;                                              \
                    for (; i + 1 < m; i += 2) {                             \
                        s0 += aj[i]     * x[i];                             \
                        s1 += aj[i + 1] * x[i + 1];                         \
                    }                                                       \
                    T s = s0 + s1;                                          \
                    for (; i < m; ++i) s += aj[i] * x[i];                   \
                    y[j] += alpha * s;                                      \
                }                                                           \
            } while (0)
            const bool use_omp = (n >= EGEMV_OMP_MIN && blas_omp_should_thread());
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel for schedule(static)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    const T *aj = &A_(0, j);
                    T s0 = zero, s1 = zero;
                    ptrdiff_t i = 0;
                    for (; i + 1 < m; i += 2) {
                        s0 += aj[i]     * x[i];
                        s1 += aj[i + 1] * x[i + 1];
                    }
                    T s = s0 + s1;
                    for (; i < m; ++i) s += aj[i] * x[i];
                    y[j] += alpha * s;
                }
#endif
            } else {
                EGEMV_T_BODY;
            }
#undef EGEMV_T_BODY
        } else {
            /* Strided x/y. Each output y[jy(j)] is disjoint across j → OMP-over-j
             * is race-free and bit-exact (jy recomputed as jy0 + j*incy). */
            const ptrdiff_t jy0 = (incy < 0) ? -(n - 1) * incy : 0;
            const ptrdiff_t ix0 = (incx < 0) ? -(m - 1) * incx : 0;
            const bool use_omp = (n >= EGEMV_OMP_MIN && blas_omp_should_thread());
#define EGEMV_T_STRIDED_BODY                                                 \
            for (ptrdiff_t j = 0; j < n; ++j) {                              \
                T s = zero;                                                  \
                ptrdiff_t ix = ix0;                                          \
                for (ptrdiff_t i = 0; i < m; ++i) {                          \
                    s += A_(i, j) * x[ix];                                   \
                    ix += incx;                                              \
                }                                                            \
                y[jy0 + j * incy] += alpha * s;                              \
            }
            if (use_omp) {
#ifdef _OPENMP
                #pragma omp parallel for schedule(static)
#endif
                EGEMV_T_STRIDED_BODY
            } else {
                EGEMV_T_STRIDED_BODY
            }
#undef EGEMV_T_STRIDED_BODY
        }
    }
}

EPBLAS_FACADE_GEMV(egemv, T)

#undef A_
