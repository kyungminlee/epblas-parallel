/*
 * yhpr2 — kind10 complex Hermitian packed rank-2 update.
 *   A := alpha*x*y^H + conj(alpha)*y*x^H + A
 */

#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* RECALIBRATED 2026-06-07 (was 64): stale libgomp-era break-even; iomp5 hot-team
 * reuse lets this O(N^2) complex Hermitian packed rank-2 update thread from N=24.
 * Measured par4/par1 (taskset 0-3, min-of-10): N=24 0.58/0.55, N=32 0.50/0.48,
 * N=64 0.33, N=128 0.28. N=20 marginal (0.71-0.77), so 24 is the robust floor.
 * Bit-exact (relerr 0). Uniform across the y* rank-update family. */
#define YHPR2_OMP_MIN 24

typedef _Complex long double TC;
typedef long double TR;
static inline TC cconj(TC z) { return ~z; }


/* Shared x87 inline-asm rank-2 cores (yr2c_run / yr2c_run_strided) —
 * also used by yher2 (full-storage twin). See the header. */
#include "rank2c_x87_core.h"

/* Per-column rank-2 updates: off-diagonal run via the hand-tuned asm core, then
 * the Hermitian diagonal forced real (the run last would dirty the x87 stack, so
 * the single diagonal write follows). Upper run is [0,j) at the array bases;
 * lower run is [0,N-1-j) at the j+1 offsets. */
__attribute__((always_inline))
static inline void yhpr2_col_upper(ptrdiff_t j, TC t1, TC t2,
                            const TC *restrict x, const TC *restrict y, TC *restrict ap) {
    TC *restrict c = ap + (size_t)j * (j + 1) / 2;
    yr2c_run(j, t1, t2, x, y, c);
    c[j] = (TR)__real__ c[j] + (TR)__real__ (x[j] * t1 + y[j] * t2);
}

__attribute__((always_inline))
static inline void yhpr2_col_lower(ptrdiff_t j, ptrdiff_t n, TC t1, TC t2,
                            const TC *restrict x, const TC *restrict y, TC *restrict ap) {
    const ptrdiff_t mo = n - j - 1;
    TC *restrict c0 = ap + ((size_t)j * n - (size_t)j * (j - 1) / 2);
    yr2c_run(mo, t1, t2, x + j + 1, y + j + 1, c0 + 1);
    c0[0] = (TR)__real__ c0[0] + (TR)__real__ (x[j] * t1 + y[j] * t2);
}

/* Strided column twins: x/y are walked with incx/incy via yr2c_run_strided (no
 * gather). kx/ky are the negative-stride base offsets; jx/jy index the diagonal
 * vector element. Same packed-column layout and accumulation order as the unit
 * helpers, so bit-identical to the gathered path. */
__attribute__((always_inline))
static inline void yhpr2_col_upper_s(ptrdiff_t j, TC t1, TC t2,
                            const TC *restrict x, ptrdiff_t incx, ptrdiff_t kx,
                            const TC *restrict y, ptrdiff_t incy, ptrdiff_t ky,
                            TC *restrict ap) {
    TC *restrict c = ap + (size_t)j * (j + 1) / 2;
    const ptrdiff_t es = (ptrdiff_t)sizeof(TC);
    yr2c_run_strided(j, t1, t2, x + kx, y + ky, c, incx * es, incy * es);
    const ptrdiff_t jx = kx + j * incx, jy = ky + j * incy;
    c[j] = (TR)__real__ c[j] + (TR)__real__ (x[jx] * t1 + y[jy] * t2);
}

__attribute__((always_inline))
static inline void yhpr2_col_lower_s(ptrdiff_t j, ptrdiff_t n, TC t1, TC t2,
                            const TC *restrict x, ptrdiff_t incx, ptrdiff_t kx,
                            const TC *restrict y, ptrdiff_t incy, ptrdiff_t ky,
                            TC *restrict ap) {
    const ptrdiff_t mo = n - j - 1;
    TC *restrict c0 = ap + ((size_t)j * n - (size_t)j * (j - 1) / 2);
    const ptrdiff_t es = (ptrdiff_t)sizeof(TC);
    const ptrdiff_t jx = kx + j * incx, jy = ky + j * incy;
    yr2c_run_strided(mo, t1, t2, x + jx + incx, y + jy + incy, c0 + 1,
                      incx * es, incy * es);
    c0[0] = (TR)__real__ c0[0] + (TR)__real__ (x[jx] * t1 + y[jy] * t2);
}

/* Serial strided dispatch: walk columns in place via the strided asm core. Used
 * only when the run would NOT thread; the threaded strided path still gathers
 * into the contiguous core below (so the omp scaling is preserved). */
static void yhpr2_strided(char UPLO, ptrdiff_t n, TC alpha,
                          const TC *restrict x, ptrdiff_t incx,
                          const TC *restrict y, ptrdiff_t incy, TC *restrict ap)
{
    const TC zero = 0.0L + 0.0Li;
    const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
    const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
    ptrdiff_t jx = kx, jy = ky;
    if (UPLO == 'U') {
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[jx] != zero || y[jy] != zero)
                yhpr2_col_upper_s(j, alpha * cconj(y[jy]), cconj(alpha * x[jx]),
                                  x, incx, kx, y, incy, ky, ap);
            else {
                const size_t kk = (size_t)j * (j + 1) / 2;
                ap[kk + j] = (TR)__real__ ap[kk + j];
            }
            jx += incx; jy += incy;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[jx] != zero || y[jy] != zero)
                yhpr2_col_lower_s(j, n, alpha * cconj(y[jy]), cconj(alpha * x[jx]),
                                  x, incx, kx, y, incy, ky, ap);
            else {
                const size_t kk = (size_t)j * n - (size_t)j * (j - 1) / 2;
                ap[kk] = (TR)__real__ ap[kk];
            }
            jx += incx; jy += incy;
        }
    }
}

/* Unit-stride dispatch, shared by the contiguous fast path and the gathered
 * strided path. schedule(static,1): column j touches j (upper) or N-1-j (lower)
 * off-diagonal packed elements, so a contiguous static block hands one thread
 * the heavy triangle end and starves the rest (par caps at ~2x on 4 cores).
 * Cyclic static,1 interleaves short and long columns across the team, balancing
 * the skew symmetrically for both UPLO. The Hermitian diagonal is forced real
 * every column — including the skipped (x[j]==y[j]==0) ones — so the else branch
 * still writes it. */
static void yhpr2_contig(char UPLO, ptrdiff_t n, TC alpha,
                         const TC *restrict x, const TC *restrict y, TC *restrict ap)
{
    const TC zero = 0.0L + 0.0Li;
    if (UPLO == 'U') {
#ifdef _OPENMP
        const bool use_omp = (n >= YHPR2_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[j] != zero || y[j] != zero)
                yhpr2_col_upper(j, alpha * cconj(y[j]), cconj(alpha * x[j]), x, y, ap);
            else {
                const size_t kk = (size_t)j * (j + 1) / 2;
                ap[kk + j] = (TR)__real__ ap[kk + j];
            }
        }
    } else {
#ifdef _OPENMP
        const bool use_omp = (n >= YHPR2_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static, 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j) {
            if (x[j] != zero || y[j] != zero)
                yhpr2_col_lower(j, n, alpha * cconj(y[j]), cconj(alpha * x[j]), x, y, ap);
            else {
                const size_t kk = (size_t)j * n - (size_t)j * (j - 1) / 2;
                ap[kk] = (TR)__real__ ap[kk];
            }
        }
    }
}

static void yhpr2_core(
    char uplo,
    ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict x, ptrdiff_t incx,
    const TC *restrict y, ptrdiff_t incy,
    TC *restrict ap)
{
    const TC alpha = *alpha_;
    const TC zero = 0.0L + 0.0Li;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    if (incx == 1 && incy == 1) {
        yhpr2_contig(UPLO, n, alpha, x, y, ap);
        return;
    }

    /* Serial strided: walk the inputs in place via the strided asm core (no
     * gather), exactly matching gfortran's strided .L40. Only the threaded path
     * below gathers — there the O(N) gather is dwarfed by the O(N^2) threaded
     * work and buys the contiguous core's omp scaling; at serial small N the
     * gather is the whole ~2% gap to gfortran, so we skip it. The predicate
     * mirrors yhpr2_contig's internal use_omp so a would-thread run is never
     * sent down the serial path. */
#ifdef _OPENMP
    const bool would_thread = (n >= YHPR2_OMP_MIN && blas_omp_max_threads() > 1);
#else
    const bool would_thread = false;
#endif
    if (!would_thread) {
        yhpr2_strided(UPLO, n, alpha, x, incx, y, incy, ap);
        return;
    }

    /* General stride: gather x and y into contiguous scratch and run the
     * stride-1 core — which already beats both refs and threads. x and y are
     * read-only here (only ap is written), so unlike esymv there is no
     * scatter-back. The gather is O(N) against O(N^2) work, so it is free past
     * tiny N; the strided per-element walk below loses ~4-9% to the references
     * (placement-bound, see project_l2_strided_gather). Same column-order
     * accumulation as the direct walk, so bit-identical. Falls back to the
     * direct strided loop if the scratch allocation fails. */
    {
        const ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        const ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        /* Exact fit: the gather writes xc[0..n-1] and yc[0..n-1] with
         * yc = stackbuf + n (max offset 2n-1), so the threshold and the
         * array length must move together. */
        enum { YHPR2_STACK_N = 256 };
        TC stackbuf[2 * YHPR2_STACK_N];
        _Static_assert(2 * YHPR2_STACK_N * sizeof(TC) <= sizeof(stackbuf),
                       "yhpr2 stack-gather threshold exceeds stackbuf");
        TC *heap = NULL;
        TC *xc, *yc;
        if (n <= YHPR2_STACK_N) {
            xc = stackbuf; yc = stackbuf + n;
        } else {
            heap = (TC *)malloc((size_t)2 * n * sizeof(TC));
            xc = heap; yc = heap ? heap + n : NULL;
        }
        if (xc && yc) {
            ptrdiff_t ix = kx, iy = ky;
            for (ptrdiff_t k = 0; k < n; ++k) {
                xc[k] = x[ix]; yc[k] = y[iy];
                ix += incx; iy += incy;
            }
            yhpr2_contig(UPLO, n, alpha, xc, yc, ap);
            free(heap);
            return;
        }
        free(heap);
    }

    {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        ptrdiff_t kk = 0;
        ptrdiff_t jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TC t1 = alpha * cconj(y[jy]);
                    const TC t2 = cconj(alpha * x[jx]);
                    ptrdiff_t ix = kx, iy = ky;
                    for (ptrdiff_t k = kk; k < kk + j; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                    ap[kk + j] = (TR)__real__ ap[kk + j] + (TR)__real__ (x[jx] * t1 + y[jy] * t2);
                } else {
                    ap[kk + j] = (TR)__real__ ap[kk + j];
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            /* Direct strided lower walk — now only the malloc-failure fallback;
             * the common strided path gathers into the contiguous core above. */
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TC t1 = alpha * cconj(y[jy]);
                    const TC t2 = cconj(alpha * x[jx]);
                    ptrdiff_t ix = jx, iy = jy;
                    ap[kk] = (TR)__real__ ap[kk] + (TR)__real__ (x[jx] * t1 + y[jy] * t2);
                    for (ptrdiff_t k = kk + 1; k < kk + (n - j); ++k) {
                        ix += incx; iy += incy;
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                    }
                } else {
                    ap[kk] = (TR)__real__ ap[kk];
                }
                jx += incx; jy += incy;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR2(yhpr2, TC)
