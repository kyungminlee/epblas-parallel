/*
 * egemmtr_ — kind10 real (long double) triangular GEMM-update, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * egemmtr_serial.c (packers, micro-kernels, macro kernels, beta-scale,
 * block policy), shared through egemmtr_kernel.h.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only the UPLO triangle of C)
 *
 * Threading (COLUMN-PANEL, the ygemmtr par4 lesson — 2026-06-24): one
 * `omp parallel` whose threads each own DISJOINT output column panels via an
 * `omp for schedule(static,1) nowait` over the jc loop. Each thread keeps a
 * PRIVATE Ap *and* Bp and packs its own B panel — there is no shared Bp, no
 * `omp single`, and no barrier inside the region. This kills the ~6% serial
 * fraction that the old shared-Bp/`omp single` row-threading paid (B-packing is
 * now parallel), lifting OMP=4 scaling from ~2.65x toward the column-threaded
 * frontier — which flips the UTN/LTN cells (where OpenBLAS has its fast native
 * A^T·B kernel and scales ~3.2x) from a ~1.04 par/ob4 loss to a win.
 *
 * NC is capped so the jc loop yields ~3 panels per thread; schedule(dynamic,1)
 * then balances the LINEAR triangular column-work ramp (UPPER: work grows with
 * jc; LOWER: shrinks) by greedy grab. static,1 round-robin left a ~1.5x LOWER
 * imbalance with so few chunks (LTN/512 par/ob 1.046->1.064); dynamic fixed it
 * (->0.922) and lifted every TN cell to a win. Rows stay serial within a thread
 * at the default MC=64 (the register kernel stays amortized).
 * Bit-identical to the serial path — only the loop carving differs, no
 * K-reduction is reordered.
 *
 * Nesting guard: when called from inside another routine's parallel region,
 * delegates to egemmtr_serial and opens no team of its own (the libgomp
 * barrier wedge guard, project-etrsm-omp4-wedge).
 */

#include "egemmtr_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
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
    const char ta = blas_trans_real(transa);
    const char tb = blas_trans_real(transb);

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

#ifdef _OPENMP
    const ptrdiff_t nthr = blas_omp_max_threads();
    const bool use_omp = (n >= EGEMMTR_OMP_MIN && nthr > 1);
    /* Column-panel threading: cap NC so the (threaded) jc loop yields ~3 panels
     * per thread, giving static,1 enough chunks to balance the linear triangular
     * column-work ramp. Columns are the threaded axis now, so this REPLACES the
     * old MC (rows) cap; rows stay serial within a thread at MC=64 so the
     * register kernel stays amortized. Cap is local to this threaded entry —
     * egemmtr_serial keeps NC=512 — and only ever lowers NC (no-op at large N
     * where N/NC_default already gives enough panels). Cols-only retiling: no
     * K-reduction is reordered, so the output is bit-identical. */
    if (use_omp) {
        ptrdiff_t cap = blas_round_up((n + 3 * nthr - 1) / (3 * nthr), NR);
        if (cap < NR) cap = NR;
        if (NC > cap) NC = cap;
    }
#else
    const bool use_omp = false;
#endif
    if (NC > n) NC = n;
    if (NC < NR) NC = NR;

    const ptrdiff_t sa_rows = blas_round_up(MC, MR);
    const ptrdiff_t sb_cols = blas_round_up(NC, NR);
    const size_t ap_bytes = (size_t)sa_rows * KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * sb_cols * sizeof(TR);

#ifdef _OPENMP
    #pragma omp parallel if(use_omp)
#endif
    {
        /* Per-thread scratch: PRIVATE Ap and Bp — no shared B, no omp single. */
        TR *Bp = (TR *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
        TR *Ap = Bp ? (TR *)aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63) : NULL;
        const bool have_buf = (Ap && Bp);

        /* EVERY team thread reaches this worksharing `omp for` (the alloc check
         * must NOT gate it). A thread whose tiny buffers failed to allocate
         * still owns its static-1 share of jc panels, so it computes them with
         * the buffer-free scalar path below — disjoint columns, still correct,
         * no double-compute. */
#ifdef _OPENMP
        /* dynamic,1: the per-column triangular work ramp (UPPER grows with jc,
         * LOWER shrinks) leaves a ~1.5x load imbalance under static,1 when the NC
         * cap yields only ~3 panels/thread — too few chunks for round-robin to
         * smooth the steep ramp. Greedy dynamic grab balances it without an
         * area-partition; columns are disjoint so order doesn't affect the result
         * (bit-identical). Scheduling overhead is negligible at ~12 panels. */
        #pragma omp for schedule(dynamic, 1) nowait
#endif
        for (ptrdiff_t jc = 0; jc < n; jc += NC) {
            const ptrdiff_t jb = blas_imin(NC, n - jc);

            if (!have_buf) {
                egemmtr_scalar_fallback_cols(jc, jc + jb, n, k, UPLO, ta, tb,
                                             alpha, a, lda, b, ldb, c, ldc);
                continue;
            }

            for (ptrdiff_t pc = 0; pc < k; pc += KC) {
                const ptrdiff_t pb = blas_imin(KC, k - pc);

                egemmtr_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);

                for (ptrdiff_t ic = 0; ic < n; ic += MC) {
                    const ptrdiff_t ib = blas_imin(MC, n - ic);

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
            }
        }
        free(Ap);
        free(Bp);
    }
}

EPBLAS_FACADE_GEMMTR(egemmtr, TR)

#undef C_
