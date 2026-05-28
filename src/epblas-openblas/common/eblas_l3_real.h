/*
 * eblas_l3_real.h — shared L3 kernel + packers for the kind10 real path.
 *
 * Port source: OpenBLAS.
 *   - kernel/generic/gemmkernel_2x2.c   → eblas_egemm_kernel
 *   - kernel/generic/gemm_ncopy_2.c     → eblas_egemm_ncopy
 *   - kernel/generic/gemm_tcopy_2.c     → eblas_egemm_tcopy
 *   - kernel/generic/gemm_beta.c        → eblas_egemm_beta
 *
 * Shared across the L3 real ports (egemm, esymm, esyrk, esyr2k, etrmm,
 * etrsm, egemmtr). Each routine still owns its own dispatch + level3
 * driver — the shared bits are only the per-tile microkernel, the
 * column-stride packers, and the beta pre-pass.
 *
 * Register-tile dims MR=2, NR=2 match OpenBLAS's reference scalar
 * kernel (`generic/gemmkernel_2x2.c`). x86_64 has no AVX path for
 * 80-bit long double, so the scalar generic kernel is the right
 * starting point and `gcc -O3` keeps the four accumulators on the
 * 8-deep x87 register stack across the K-loop.
 */
#ifndef EBLAS_L3_REAL_H
#define EBLAS_L3_REAL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ── Register-tile dimensions (compile-time constants) ───────────── */
#define EBLAS_EGEMM_MR 2
#define EBLAS_EGEMM_NR 2

/* ── Cache-block defaults (overridable via getenv at routine entry) ─
 *
 * P = MC (row panel of A held in L2)
 * Q = KC (K-panel)
 * R = NC (column band of B held above)
 *
 * Defaults sized so MC*KC of `long double` (16 B each on x86-64)
 * fits inside ~256 KB of L2:  64 * 256 * 16 B  = 256 KB.
 * KC*NC for B is 256 * 512 * 16 B = 2 MB (lives in L3 across the
 * KC-walk; one ncopy per (jc, pc) tile).
 *
 * These values are deliberate cache-arithmetic choices for the
 * 80-bit-no-SIMD case — they don't trace back to any specific
 * other port. */
#define EBLAS_EGEMM_GEMM_P  64
#define EBLAS_EGEMM_GEMM_Q  256
#define EBLAS_EGEMM_GEMM_R  512

/* ── Microkernel ─────────────────────────────────────────────────────
 *
 * Compute C += alpha * Ap * Bp for one (bm, bn) tile of C, where
 * bm <= some MC-aligned bound, bn <= some NC-aligned bound, and
 * Ap / Bp are panel-packed:
 *
 *   Ap: 2-row panels.  For panel q in 0..ceil(bm/MR)-1,
 *       Ap[(q*bk + p)*MR + i] = A[i_panel_top + i, k_panel_left + p]
 *       for i in 0..MR-1, p in 0..bk-1, zero-padded when MR doesn't
 *       divide bm.
 *   Bp: 2-col panels.  Similar shape but stride-NR per K-row.
 *
 * Faithful translation of OpenBLAS gemmkernel_2x2.c — same MR=2
 * NR=2 outer-product over k, same edge handling for the bm&1 and
 * bn&1 tails. Difference: applies alpha as `C += alpha*acc`
 * directly (OpenBLAS's macro `TO_OUTPUT(C_TO_F32(C) + acc)` is a
 * bf16-conversion no-op for non-BFLOAT16 builds).
 */
void eblas_egemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        long double alpha,
                        const long double *Ap,
                        const long double *Bp,
                        long double *C, ptrdiff_t ldc);

/* ── A-panel inner-copy: source matrix in N (no-trans) layout ─────────
 *
 * Read A[0..m, 0..n] (column-major, lda between cols) and write the
 * NR=2-row panel-packed form into `b`. NR=2 columns at a time:
 *
 *   b[0] = a[0, 0]   b[1] = a[0, 1]
 *   b[2] = a[1, 0]   b[3] = a[1, 1]   ...
 *
 * Used as ICOPY for normal A and OCOPY for normal B (the same
 * physical layout for both — OpenBLAS reuses `gemm_ncopy_2.c` for
 * both roles by exchanging the (m,n) argument convention).
 */
void eblas_egemm_ncopy(ptrdiff_t m, ptrdiff_t n,
                       const long double *a, ptrdiff_t lda,
                       long double *b);

/* ── A-panel inner-copy: source matrix in T (trans) layout ────────────
 *
 * Read A[0..m, 0..n] (column-major, lda between cols) and write the
 * transposed NR=2-row panel-packed form into `b`. Symmetric counterpart
 * of eblas_egemm_ncopy — handles the trans cases of A and B.
 */
void eblas_egemm_tcopy(ptrdiff_t m, ptrdiff_t n,
                       const long double *a, ptrdiff_t lda,
                       long double *b);

/* ── Beta pre-pass ───────────────────────────────────────────────────
 *
 * Apply C[i, j] := beta * C[i, j] for i in [0, m), j in [0, n).
 * Handles the beta == 0 case as a zero-fill (matches Fortran
 * BLAS spec — out-of-range NaN in C must not propagate when
 * beta == 0). beta == 1 returns immediately.
 *
 * Mirrors OpenBLAS gemm_beta.c.
 */
void eblas_egemm_beta(ptrdiff_t m, ptrdiff_t n,
                      long double beta,
                      long double *c, ptrdiff_t ldc);

/* ── Env-var block-size overrides ────────────────────────────────────
 *
 * Read once into static cache. Names match the epblas-parallel overlay's
 * pattern (EBLAS_MC, EBLAS_KC, EBLAS_NC) so the same tuning env
 * controls both overlays — these are runtime block-size knobs, not
 * portage of the epblas-parallel implementation.
 *
 * Returns the effective (MC, KC, NC) values for this process.
 */
void eblas_egemm_blocks(int *mc, int *kc, int *nc);

/* ── SYMM-aware packers (real) ───────────────────────────────────────
 *
 * Port source: OpenBLAS kernel/generic/symm_ucopy_2.c (esymm_ucopy)
 *              and symm_lcopy_2.c (esymm_lcopy).
 *
 * Read an (m × n) slab from a symmetric matrix `a` (col-major, leading
 * dimension lda) and emit it in the MR=2 / NR=2 strip-packed format
 * the shared GEMM microkernel expects.
 *
 * - `posX` is the source-A coordinate the function increments by 2 per
 *   outer pair (an M-row coord for ICOPY in SIDE=L, or an N-col coord
 *   for OCOPY in SIDE=R).
 * - `posY` is the source-A coordinate the inner i-loop walks (a K-col
 *   coord for ICOPY, K-row coord for OCOPY — they're interchangeable
 *   because A is symmetric).
 *
 * Both functions reconstruct the cross-diagonal entries by mirroring
 * the stored triangle (upper for `ucopy`, lower for `lcopy`). The
 * crossing point falls on one iteration where offset == 0; at that
 * point both addressing formulas point to the same element on the
 * diagonal, so a single read suffices and the advance switches modes.
 */
void eblas_esymm_ucopy(ptrdiff_t m, ptrdiff_t n,
                       const long double *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       long double *b);

void eblas_esymm_lcopy(ptrdiff_t m, ptrdiff_t n,
                       const long double *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       long double *b);

/* ── Triangular beta pre-pass (real) ─────────────────────────────────
 *
 * Apply C[i, j] := beta * C[i, j] for the UPLO triangle only — for SYRK
 * the off-UPLO triangle is supposed to be left untouched, so we cannot
 * use eblas_egemm_beta (which touches the full M×N rectangle).
 *
 * Mirrors OpenBLAS's `syrk_beta` (driver/level3/syrk_k.c lines 52–99).
 */
void eblas_esyrk_beta_u(ptrdiff_t n,
                        long double beta,
                        long double *c, ptrdiff_t ldc);
void eblas_esyrk_beta_l(ptrdiff_t n,
                        long double beta,
                        long double *c, ptrdiff_t ldc);

/* ── SYRK kernel: GEMM with diagonal-aware writeback (real) ──────────
 *
 * Port source: OpenBLAS driver/level3/syrk_kernel.c (LOWER and !LOWER
 * variants — selected via the {u,l} suffix here).
 *
 * Computes one (m × n) tile of C += alpha * Ap * Bp, but writes only
 * the UPLO triangle of the global C. `offset` is the position of the
 * tile's TL corner relative to the C diagonal (offset = X - Y, where
 * X is the C-row of the tile's TL and Y is the C-col of the tile's TL).
 *
 * Ap / Bp are already in MR=2 / NR=2 strip-packed format; ldc is in
 * elements. The kernel classifies the tile:
 *   - Entirely in the kept triangle → full eblas_egemm_kernel call.
 *   - Entirely in the off-UPLO triangle → no-op.
 *   - Diagonal-crossing → split into pre-diagonal full-write strip(s),
 *     a square diagonal block (GEMM'd into a small subbuffer then
 *     mask-merged into the right triangle of C), and a post-diagonal
 *     full-write strip(s).
 */
void eblas_esyrk_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          long double alpha,
                          const long double *Ap,
                          const long double *Bp,
                          long double *c, ptrdiff_t ldc,
                          ptrdiff_t offset);
void eblas_esyrk_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          long double alpha,
                          const long double *Ap,
                          const long double *Bp,
                          long double *c, ptrdiff_t ldc,
                          ptrdiff_t offset);

/* ── SYR2K kernel: two-pass diagonal-aware GEMM (real) ───────────────
 *
 * Port source: OpenBLAS driver/level3/syr2k_kernel.c (LOWER and !LOWER
 * variants — selected via the {u,l} suffix). Same offset/strip semantics
 * as the SYRK kernel; the difference is the diagonal NR×NR subbuffer
 * merge writes `subbuf[i,j] + subbuf[j,i]` (symmetric mirror), so a
 * single first-pass call covers both A*B^T and B*A^T contributions on
 * the diagonal block.
 *
 * `flag` controls the diagonal subbuf merge:
 *   - flag != 0: do the merge (used for pass 1 with Ap=A, Bp=B).
 *   - flag == 0: skip the merge (used for pass 2 with Ap=B, Bp=A —
 *     pass 1 already wrote both contributions to the diagonal block).
 *
 * The pre/post-diagonal strip writes are not flag-gated — both passes
 * write their strip contributions (A*B^T from pass 1, B*A^T from pass 2)
 * into the same kept-triangle locations.
 */
void eblas_esyr2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           long double alpha,
                           const long double *Ap,
                           const long double *Bp,
                           long double *c, ptrdiff_t ldc,
                           ptrdiff_t offset, int flag);
void eblas_esyr2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           long double alpha,
                           const long double *Ap,
                           const long double *Bp,
                           long double *c, ptrdiff_t ldc,
                           ptrdiff_t offset, int flag);

/* ── TRMM A-side triangular packers (real) ───────────────────────────
 *
 * Faithful ports of OpenBLAS kernel/generic/trmm_{ut,un,lt,ln}copy_2.c.
 * Each function packs an (m × n) slab from a triangular A (col-major,
 * leading dimension lda) into the MR=2/NR=2 strip-packed format the
 * shared GEMM/TRMM microkernels expect.
 *
 * The same source file serves both ICOPY (SIDE=L, the sa target) and
 * OCOPY (SIDE=R, the sb target) roles in OpenBLAS — distinguished by
 * the (m, n) shape and (posX, posY) origin at the call site.
 *
 * - (posX, posY) name the A-source coordinate of the slab's TL corner.
 *   posX walks the 2-step axis (the "is" / register-strip axis); posY
 *   walks the K axis (the "ls" / panel axis). The packer reconstructs
 *   the diagonal-crossing pattern from (posX - posY).
 * - `unit` is the runtime equivalent of OpenBLAS's compile-time UNIT
 *   macro: if non-zero, the on-diagonal element is forced to 1.0L and
 *   the off-diagonal-but-stored-as-zero positions are written ZERO.
 *
 * Per-iteration the four logical positions (above, below, on-diagonal,
 * strict-zero) map to four write patterns — for UNCOPY/UTCOPY (upper)
 * vs LNCOPY/LTCOPY (lower) the "above" and "below" labels are swapped.
 * "T" (transposed) packers walk the K axis along the 2nd-element-of-
 * pair dimension; "N" (normal) packers walk it along the leading dim.
 */
void eblas_etrmm_iutcopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         long double *b, int unit);
void eblas_etrmm_iuncopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         long double *b, int unit);
void eblas_etrmm_iltcopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         long double *b, int unit);
void eblas_etrmm_ilncopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY,
                         long double *b, int unit);

/* ── TRMM diagonal-aware microkernel (real) ──────────────────────────
 *
 * Faithful port of OpenBLAS kernel/generic/trmmkernel_2x2.c, with the
 * compile-time LEFT and TRANSA macros converted to runtime `left` and
 * `trans` flags. The kernel implements 4 distinct (LEFT, TRANSA)
 * variants — the offset/temp formulas and the ptrba/ptrbb advances
 * differ per variant, exactly as in the OpenBLAS source. TRMMKERNEL
 * is always on for us.
 *
 * Semantics: C := alpha * Ap * Bp on the (bm × bn) tile, with the
 * triangular cut governed by `offset` (the TL-corner offset of this
 * tile relative to the A diagonal — same convention as the SYRK
 * kernels). Note this is `=` (overwrite), NOT `+=` like
 * eblas_egemm_kernel — TRMM's L3 nest fully overwrites each B tile
 * exactly once, matching the OpenBLAS GEMM_KERNEL(beta=0) convention.
 */
void eblas_etrmm_kernel(int left, int trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        long double alpha,
                        const long double *ba,
                        const long double *bb,
                        long double *C, ptrdiff_t ldc,
                        ptrdiff_t offset);

/* ── GEMM kernel, overwrite variant ──────────────────────────────────
 *
 * Same as eblas_egemm_kernel but with C := alpha * Ap * Bp (no
 * accumulate). Mirrors the OpenBLAS GEMM_KERNEL(beta=0) calling
 * convention used inside the TRMM L3 driver for off-diagonal tiles.
 * Implemented as: zero the tile then call eblas_egemm_kernel.
 */
void eblas_egemm_kernel_store(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                              long double alpha,
                              const long double *Ap,
                              const long double *Bp,
                              long double *C, ptrdiff_t ldc);

/* ── TRSM A-side triangular packers (real) ───────────────────────────
 *
 * Faithful ports of OpenBLAS kernel/generic/trsm_{ut,un,lt,ln}copy_2.c
 * with compile-time UNIT replaced by runtime `unit`. The (m, n, a,
 * lda, offset, b) shape matches OpenBLAS exactly — `offset` is the
 * diagonal-alignment offset that lets the packer reconstruct the
 * diagonal-crossing pattern. unit==1 forces diag to 1.0L (no inversion).
 *
 * **Critical structural difference vs the TRMM packers**: TRSM packers
 * write `1.0L / a[diag]` (or 1.0L when unit) at every on-diagonal
 * position. They write strict-stored-triangle data at the off-diagonal
 * stored positions and LEAVE THE OTHER TRIANGLE UNTOUCHED — the
 * downstream TRSM kernel never reads those positions (the solve walks
 * only the kept-triangle data; the internal trailing GEMM reads the
 * already-solved B-slab).
 *
 * Used in 2 roles by the L3 driver:
 *   - I-side (sa) for SIDE=L: trsm_L.c lines 127/129, etc.
 *   - O-side (sb) for SIDE=R: trsm_R.c lines 170/172/290/293, etc.
 * Same source file serves both — distinguished by (m, n, posX-arg)
 * shape at call site.
 */
void eblas_etrsm_iutcopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t offset, long double *b, int unit);
void eblas_etrsm_iuncopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t offset, long double *b, int unit);
void eblas_etrsm_iltcopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t offset, long double *b, int unit);
void eblas_etrsm_ilncopy(ptrdiff_t m, ptrdiff_t n,
                         const long double *a, ptrdiff_t lda,
                         ptrdiff_t offset, long double *b, int unit);

/* ── TRSM diagonal-aware microkernel (real) ──────────────────────────
 *
 * Faithful port of OpenBLAS kernel/generic/trsm_kernel_{LN,LT,RN,RT}.c —
 * the 4 distinct algorithms (different solve direction + different
 * internal GEMM trailing-update structure) collapsed into one function
 * dispatching on runtime `left` and `trans` flags. The kernel walks
 * (m, n) tiles of register-tile sub-blocks; for each register tile
 *   1) does a trailing-update GEMM with dm1 = -1 against the already-
 *      solved B slab to the left/right (via the shared
 *      eblas_egemm_kernel — note its += accumulate semantics line up
 *      since the trsm_kernel calls it with dm1 = -1, which subtracts),
 *   2) calls the per-variant `solve()` that scales B by inv(diag(A))
 *      and propagates that into B's same-tile rows/cols via the
 *      packed triangular A entries.
 *
 * Semantics: C is the (m × n) sub-tile of B; alpha here is always
 * dm1 = -1 (matches OpenBLAS — TRSM applies the user's alpha as a
 * B *= alpha pre-pass, then runs the kernel with -1 scaling). `ba`
 * is the packed triangular A (with inv(diag) baked in by the packer),
 * `bb` is the packed B slab (for SIDE=L it's GEMM_ONCOPY of a tile of
 * B; for SIDE=R it's GEMM_ITCOPY of a tile of B); `offset` is the
 * TL-corner offset of this tile relative to the A diagonal.
 *
 * The 4 variants differ in:
 *   - which direction `solve()` walks (i=m-1 down → LN; i=0 up → LT;
 *     i=0 up over n → RN; i=n-1 down over n → RT)
 *   - where the trailing-GEMM trapezoid lives relative to the tile
 *     (above for LN/RT; below for LT/RN) — encoded via kk = m+offset
 *     vs kk = offset vs kk = -offset, etc.
 */
void eblas_etrsm_kernel(int left, int trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        const long double *ba,
                        const long double *bb,
                        long double *C, ptrdiff_t ldc,
                        ptrdiff_t offset);

#ifdef __cplusplus
}
#endif

#endif /* EBLAS_L3_REAL_H */
