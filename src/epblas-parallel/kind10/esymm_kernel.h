/*
 * esymm_kernel.h — internal shared surface for the kind10 real
 * (REAL(KIND=10) / long double) symmetric matrix-multiply overlay, split
 * across two translation units:
 *
 *   esymm_serial.c   — the symm-aware packers + the pure single-thread
 *                      fused driver (`esymm_serial`). No `#pragma omp`.
 *   esymm_parallel.c — the public Fortran entry `esymm_`: same fused
 *                      driver fanned across an OpenMP team (M-axis split,
 *                      shared Bp), with an `omp_in_parallel()` guard that
 *                      delegates to `esymm_serial` when called from inside
 *                      another routine's parallel region.
 *
 * Structure mirrors the egemm overlay: esymm owns NO GEMM math of its own.
 * It composes the shared egemm kernel primitives (block policy, packers,
 * beta pre-pass, MR×NR macro-kernel — see egemm_kernel.h) and adds only the
 * SYMM-aware packers below, which read the symmetric operand into egemm's
 * packed layout while mirroring the UPLO triangle across the diagonal. The
 * same microkernel then streams diagonal and off-diagonal tiles alike — no
 * scalar diagonal special-case, no per-tile re-dispatch into egemm.
 */
#ifndef EPBLAS_PARALLEL_KIND10_ESYMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ESYMM_KERNEL_H

#include <stddef.h>

typedef long double esymm_TR;

/* ── SYMM-aware packers (real) ───────────────────────────────────────
 *
 * Pack a block of the symmetric operand `a` (col-major, leading dim lda,
 * only the UPLO triangle populated — the other triangle is never read)
 * into the egemm packed layout. The logical element (row, col) is
 * reconstructed by mirroring: A[row,col] == A[col,row], so a position in
 * the unstored triangle is fetched from its transpose in the stored one.
 *
 *   esymm_pack_a_sym → egemm_pack_A('N') layout: rows [ic, ic+ib) over the
 *                      MR-row panels, cols [pc, pc+pb) over the K depth.
 *                      Used for the SIDE='L' A-operand.
 *   esymm_pack_b_sym → egemm_pack_B('N') layout: rows [pc, pc+pb) over the
 *                      K depth, cols [jc, jc+jb) over the NR-col panels.
 *                      Used for the SIDE='R' B-operand.
 *
 * `uplo` is 'U' or 'L' (already upper-cased by the caller).
 */
void esymm_pack_a_sym(const esymm_TR *a, ptrdiff_t lda,
                      ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb,
                      char uplo, esymm_TR *Ap);
void esymm_pack_b_sym(const esymm_TR *a, ptrdiff_t lda,
                      ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb,
                      char uplo, esymm_TR *Bp);

/* Pure-serial by-value core (no OpenMP). Same math as esymm_. */
void esymm_serial(
    char side, char uplo,
    ptrdiff_t m, ptrdiff_t n,
    const esymm_TR *alpha_,
    const esymm_TR *a, ptrdiff_t lda,
    const esymm_TR *b, ptrdiff_t ldb,
    const esymm_TR *beta_,
    esymm_TR *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_ESYMM_KERNEL_H */
