/*
 * espr2 — kind10 (long double) symmetric packed rank-2 update.
 *   A := alpha*x*y^T + alpha*y*x^T + A
 */

#include <stddef.h>
#include <stdlib.h>
#include "../common/blas_char.h"
#include <ctype.h>
#include "../common/epblas_facade.h"
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ESPR2_OMP_MIN 64

typedef long double TR;


/* Per-column rank-2 updates, carved out as their own functions so the inner
 * loop compiles with clean x87 register allocation. Inlined into the
 * `omp parallel for` body, the upper-triangle loop loses ~10% (the outlined
 * region spills the kept-resident operands); keeping it a separate noinline
 * function restores parity with the reference and lets both the serial and
 * threaded paths share one tight loop. */
__attribute__((noinline))
static void espr2_col_upper(ptrdiff_t j, TR t1, TR t2,
                            const TR *restrict x, const TR *restrict y, TR *restrict ap) {
    TR *restrict c = ap + (size_t)j * (j + 1) / 2;
    for (ptrdiff_t i = 0; i <= j; ++i) c[i] += x[i] * t1 + y[i] * t2;
}

__attribute__((noinline))
static void espr2_col_lower(ptrdiff_t j, ptrdiff_t n, TR t1, TR t2,
                            const TR *restrict x, const TR *restrict y, TR *restrict ap) {
    TR *restrict c = ap + ((size_t)j * n - (size_t)j * (j - 1) / 2) - j;
    for (ptrdiff_t i = j; i < n; ++i) c[i] += x[i] * t1 + y[i] * t2;
}

static void espr2_core(
    char uplo,
    ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict x, ptrdiff_t incx,
    const TR *restrict y, ptrdiff_t incy,
    TR *restrict ap)
{
    const TR alpha = *alpha_;
    const TR zero = 0.0L;
    const char UPLO = blas_up(uplo);

    if (n == 0 || alpha == zero) return;

    /* General stride: gather x and y into contiguous scratch and run the
     * contiguous paths. Inputs only — packed A is the output and already
     * contiguous, so there is no scatter-back. The strided walk reads x/y
     * O(n^2) times at stride inc{x,y} (the references' strided form costs
     * ~11% over their own contiguous form on both uplos, while par's
     * contiguous cores tie-or-beat the references' contiguous form), so the
     * O(n) gather nets ~9-10%. It also unlocks the balanced-schedule OMP
     * path, which lives in the contiguous branch only. malloc failure falls
     * through to the direct strided walk. */
    if (incx != 1 || incy != 1) {
        enum { ESPR2_STACK_N = 512 };
        TR stackbuf[2 * ESPR2_STACK_N];
        TR *heap = NULL;
        TR *xc = (n <= ESPR2_STACK_N)
            ? stackbuf
            : (heap = (TR *)malloc(2 * (size_t)n * sizeof(TR)));
        if (xc) {
            TR *restrict yc = xc + n;
            const ptrdiff_t kx0 = (incx < 0) ? -(n - 1) * incx : 0;
            const ptrdiff_t ky0 = (incy < 0) ? -(n - 1) * incy : 0;
            ptrdiff_t ix = kx0, iy = ky0;
            for (ptrdiff_t i = 0; i < n; ++i) {
                xc[i] = x[ix]; yc[i] = y[iy];
                ix += incx; iy += incy;
            }
            espr2_core(uplo, n, alpha_, xc, 1, yc, 1, ap);
            free(heap);
            return;
        }
    }

    if (incx == 1 && incy == 1) {
        /* schedule(static, 8): column j touches j+1 (upper) or N-j (lower)
         * packed elements, so a contiguous static block hands one thread the
         * heavy triangle end and starves the rest (par caps at ~2x on 4 cores).
         * A balanced schedule fixes that. Among balanced options a MODERATE
         * chunk beats cyclic chunk-1 for this real rank-2 body: adjacent packed
         * columns are contiguous in ap, so chunk-1 puts every neighbour pair on
         * different threads (false sharing). static,8 is ~1-2% faster than
         * static,1 here (measured 2026-06-02). NOTE the complex rank-2 twin
         * yhpr2 keeps static,1 — its heavier body fully hides the false sharing
         * and the finest balance wins there; the lighter real rank-1 espr also
         * uses static,8. */
        /* The per-column noinline helper exists for the OMP body (inlined into
         * the outlined region it spills the kept-resident operands, -~10%). But
         * the SERIAL path pays that helper's call overhead — two long-double args
         * marshaled through memory + call/ret per column, ~10ns/col — for nothing:
         * a plain serial loop is a clean register context, no outlining. At small
         * N that fixed per-column tax is the whole gap (the references inline their
         * column loop): par was flat ~18500ns at N=128 on BOTH uplos while netlib
         * hit 17145 on Upper. Inline the loop for serial, keep the helper for OMP.
         * Running base pointer (c += j+1 / base += n-j) also drops the per-column
         * j*(j+1)/2 packed-offset recompute. Bit-exact (same op order as helper). */
#ifdef _OPENMP
        const bool use_omp = (n >= ESPR2_OMP_MIN && blas_omp_max_threads() > 1);
#else
        const bool use_omp = false;
#endif
        if (UPLO == 'U') {
#ifdef _OPENMP
            if (use_omp) {
                #pragma omp parallel for schedule(static, 8)
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero || y[j] != zero)
                        espr2_col_upper(j, alpha * y[j], alpha * x[j], x, y, ap);
                }
            } else
#endif
            {
                TR *restrict c = ap;
                for (ptrdiff_t j = 0; j < n; ++j) {
                    if (x[j] != zero || y[j] != zero) {
                        const TR t1 = alpha * y[j], t2 = alpha * x[j];
                        for (ptrdiff_t i = 0; i <= j; ++i) c[i] += x[i] * t1 + y[i] * t2;
                    }
                    c += j + 1;
                }
            }
        } else {
            /* Lower keeps the noinline helper for BOTH serial and OMP: unlike
             * Upper, inlining the Lower column loop into the j-loop regressed
             * large-N ~9% (the isolated helper compile keeps the inner-loop
             * operands x87-resident; inlined, the outer induction evicts them —
             * the clean-regalloc benefit the helper was carved out for). Lower
             * was never the flagged shape — it already beats netlib ~9% here. */
#ifdef _OPENMP
            #pragma omp parallel for if(use_omp) schedule(static, 8)
#endif
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[j] != zero || y[j] != zero)
                    espr2_col_lower(j, n, alpha * y[j], alpha * x[j], x, y, ap);
            }
        }
    } else {
        ptrdiff_t kx = (incx < 0) ? -(n - 1) * incx : 0;
        ptrdiff_t ky = (incy < 0) ? -(n - 1) * incy : 0;
        ptrdiff_t kk = 0;
        ptrdiff_t jx = kx, jy = ky;
        if (UPLO == 'U') {
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TR t1 = alpha * y[jy];
                    const TR t2 = alpha * x[jx];
                    ptrdiff_t ix = kx, iy = ky;
                    for (ptrdiff_t k = kk; k < kk + j + 1; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += j + 1;
            }
        } else {
            for (ptrdiff_t j = 0; j < n; ++j) {
                if (x[jx] != zero || y[jy] != zero) {
                    const TR t1 = alpha * y[jy];
                    const TR t2 = alpha * x[jx];
                    ptrdiff_t ix = jx, iy = jy;
                    for (ptrdiff_t k = kk; k < kk + n - j; ++k) {
                        ap[k] += x[ix] * t1 + y[iy] * t2;
                        ix += incx; iy += incy;
                    }
                }
                jx += incx; jy += incy;
                kk += n - j;
            }
        }
    }
}

EPBLAS_FACADE_SPR2(espr2, TR)
