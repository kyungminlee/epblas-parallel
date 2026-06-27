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


/* Per-column rank-2 updates, carved out as their own functions so the inner
 * loop compiles with clean x87 register allocation. Inlined into the
 * `omp parallel for` body, the upper-triangle loop loses ~10% (the outlined
 * region spills the kept-resident operands); keeping it a separate noinline
 * function restores parity with the reference and lets both the serial and
 * threaded paths share one tight loop. The Hermitian diagonal is forced real
 * here: the off-diagonal run plus the single real diagonal write. */
__attribute__((noinline))
static void yhpr2_col_upper(ptrdiff_t j, TC t1, TC t2,
                            const TC *restrict x, const TC *restrict y, TC *restrict ap) {
    TC *restrict c = ap + (size_t)j * (j + 1) / 2;
    for (ptrdiff_t i = 0; i < j; ++i) c[i] += x[i] * t1 + y[i] * t2;
    c[j] = (TR)__real__ c[j] + (TR)__real__ (x[j] * t1 + y[j] * t2);
}

__attribute__((noinline))
static void yhpr2_col_lower(ptrdiff_t j, ptrdiff_t n, TC t1, TC t2,
                            const TC *restrict x, const TC *restrict y, TC *restrict ap) {
    /* Pre-advance the off-diagonal bases so the loop runs 0-based over a single
     * induction variable indexing three pointers — the exact tight form gcc
     * picks for the upper helper. A loop that starts at i=1 (or indexes the
     * original arrays by the absolute j+1..N-1) instead makes gcc walk three
     * separate pointers with an extra increment per iteration (~7% on the
     * lower triangle). Diagonal last so the loop compiles on a clean x87 stack. */
    const ptrdiff_t mo = n - j - 1;
    TC *restrict c0 = ap + ((size_t)j * n - (size_t)j * (j - 1) / 2);
    TC *restrict c = c0 + 1;
    const TC *restrict xc = x + j + 1, *restrict yc = y + j + 1;
    for (ptrdiff_t i = 0; i < mo; ++i) c[i] += xc[i] * t1 + yc[i] * t2;
    c0[0] = (TR)__real__ c0[0] + (TR)__real__ (x[j] * t1 + y[j] * t2);
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
        TC stackbuf[2 * 256];
        TC *heap = NULL;
        TC *xc, *yc;
        if (n <= 256) {
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
