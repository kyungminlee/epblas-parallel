/*
 * egemmtr_ — kind10 real (long double) triangular GEMM-update, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * egemmtr_serial.c (packers, micro-kernels, macro kernels, beta-scale,
 * block policy), shared through egemmtr_kernel.h.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only the UPLO triangle of C)
 *
 * Threading: outer `omp parallel`, Bp shared and packed once per (jc, pc)
 * via `omp single` (implicit barrier), Ap private per thread, `omp for`
 * over the ic loop with `schedule(static, 1)` (interleaves ic chunks so the
 * triangular load — early-ic threads get fewer skipped tiles for LOWER,
 * later-ic threads for UPPER — balances).
 *
 * Nesting guard: when called from inside another routine's parallel region,
 * delegates to egemmtr_serial and opens no team of its own (the libgomp
 * barrier wedge guard, project-etrsm-omp4-wedge).
 */

#include "egemmtr_kernel.h"
#include "../common/blas_char.h"
#include "../common/epblas_facade.h"
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#include <stddef.h>
#endif

typedef egemmtr_TR TR;

#define MR EGEMMTR_MR
#define NR EGEMMTR_NR
#define EGEMMTR_OMP_MIN 32

static inline ptrdiff_t imin(ptrdiff_t a, ptrdiff_t b) { return a < b ? a : b; }

#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static void egemmtr_core(char uplo, char transa, char transb,
                         ptrdiff_t n, ptrdiff_t k,
                         const TR *alpha_,
                         const TR *restrict a, ptrdiff_t lda,
                         const TR *restrict b, ptrdiff_t ldb,
                         const TR *beta_,
                         TR *restrict c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Already inside a team → run serially in this thread, no nested
     * region (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        egemmtr_serial(uplo, transa, transb, n, k, alpha_, a, lda, b, ldb,
                       beta_, c, ldc);
        return;
    }
#endif
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const char ta = egemmtr_trans_code(&transa);
    const char tb = egemmtr_trans_code(&transb);

    if (n <= 0) return;
    const TR zero = 0.0L, one = 1.0L;

    if (alpha == zero || k == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const bool use_omp0 = (n >= EGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp0) schedule(static, 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j)
            egemmtr_beta_scale(j, j + 1, n, UPLO, beta, c, ldc);
        return;
    }

    /* Beta-scale the UPLO triangle up front so the packed kernel can
     * always assume beta=1. */
    if (beta != one) {
#ifdef _OPENMP
        const bool use_omp_beta = (n >= EGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp_beta) schedule(static, 1)
#endif
        for (ptrdiff_t j = 0; j < n; ++j)
            egemmtr_beta_scale(j, j + 1, n, UPLO, beta, c, ldc);
    }

    ptrdiff_t MC, KC, NC;
    egemmtr_block_sizes(&MC, &KC, &NC);
    if (NC > n) NC = n;
    if (NC < NR) NC = NR;

#ifdef _OPENMP
    /* The `omp for` below partitions the ic (output-row-block) loop across the
     * team. With the default MC=64, small/moderate N yields only N/MC ic-blocks
     * — too few for the team — and the triangular work-skew (later ic-blocks
     * keep more of the stored triangle, so they are heavier) leaves the
     * lightest-block threads idle under schedule(static, 1). Cap MC locally so
     * the ic loop yields >=~3 blocks per thread, giving static,1 enough chunks
     * to balance the smooth row-work ramp. This is a ROWS-only retiling: it does
     * not reorder any K-reduction, so the output is bit-identical. The cap is
     * LOCAL to this threaded entry — egemmtr_block_sizes and egemmtr_serial keep
     * MC=64, and it is a no-op once N/MC_default already gives enough blocks
     * (large N) since we only ever lower MC. */
    const ptrdiff_t nthr = blas_omp_max_threads();
    if (n >= EGEMMTR_OMP_MIN && nthr > 1) {
        ptrdiff_t cap = egemmtr_round_up((n + 3 * nthr - 1) / (3 * nthr), MR);
        if (cap < 32) cap = 32;        /* keep the register kernel amortized */
        if (MC > cap) MC = cap;
    }
#endif

    const ptrdiff_t sa_rows = egemmtr_round_up(MC, MR);
    const ptrdiff_t sb_cols = egemmtr_round_up(NC, NR);
    const size_t ap_bytes = (size_t)sa_rows * KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * sb_cols * sizeof(TR);

    TR *Bp = (TR *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) {
        egemmtr_scalar_fallback(n, k, UPLO, ta, tb, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

#ifdef _OPENMP
    const bool use_omp = (n >= EGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel if(use_omp)
#endif
    {
        TR *Ap = (TR *)aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (Ap) {
            for (ptrdiff_t jc = 0; jc < n; jc += NC) {
                const ptrdiff_t jb = imin(NC, n - jc);
                for (ptrdiff_t pc = 0; pc < k; pc += KC) {
                    const ptrdiff_t pb = imin(KC, k - pc);

#ifdef _OPENMP
                    #pragma omp single
#endif
                    egemmtr_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                    /* implicit barrier at end of `single` makes Bp safe. */

#ifdef _OPENMP
                    #pragma omp for schedule(static, 1)
#endif
                    for (ptrdiff_t ic = 0; ic < n; ic += MC) {
                        const ptrdiff_t ib = imin(MC, n - ic);

                        ptrdiff_t tile_class;
                        if (UPLO == 'L') {
                            if (ic + ib <= jc)        tile_class = 0;
                            else if (ic >= jc + jb)   tile_class = 2;
                            else                      tile_class = 1;
                        } else {
                            if (ic >= jc + jb)        tile_class = 0;
                            else if (ic + ib <= jc)   tile_class = 2;
                            else                      tile_class = 1;
                        }
                        if (tile_class == 0) continue;

                        egemmtr_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);

                        if (tile_class == 1)
                            egemmtr_macro_kernel_tri(ib, jb, pb, alpha, Ap, Bp,
                                                     &C_(ic, jc), ldc,
                                                     ic, jc, UPLO);
                        else
                            egemmtr_macro_kernel_rect(ib, jb, pb, alpha, Ap, Bp,
                                                      &C_(ic, jc), ldc);
                    }
                    /* implicit barrier at end of `for` keeps Bp stable. */
                }
            }
        }
        free(Ap);
    }

    free(Bp);
}

EPBLAS_FACADE_GEMMTR(egemmtr, TR)

#undef C_
