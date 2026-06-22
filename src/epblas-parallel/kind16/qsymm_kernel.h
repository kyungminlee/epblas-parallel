/*
 * qsymm_kernel.h — internal shared surface for the kind16 real
 * (REAL(KIND=16) / __float128) symmetric matrix-multiply overlay, split
 * across two translation units:
 *
 *   qsymm_serial.c   — the symm-aware packers + the pure single-thread
 *                      fused driver (`qsymm_serial_`). No `#pragma omp`.
 *   qsymm_parallel.c — the public Fortran entry `qsymm_`: same fused
 *                      driver fanned across an OpenMP team (M-axis split,
 *                      shared Bp), with an `omp_in_parallel()` guard that
 *                      delegates to `qsymm_serial_` when called from inside
 *                      another routine's parallel region.
 *
 * Structure mirrors the qgemm overlay: qsymm owns NO GEMM math of its own.
 * It composes the shared qgemm kernel primitives (block policy, packers,
 * beta pre-pass, MR×NR macro-kernel — see qgemm_kernel.h) and adds only the
 * SYMM-aware packers below, which read the symmetric operand into qgemm's
 * packed layout while mirroring the UPLO triangle across the diagonal. The
 * same microkernel then streams diagonal and off-diagonal tiles alike — no
 * scalar diagonal special-case, no per-tile re-dispatch into qgemm.
 */
#ifndef EPBLAS_PARALLEL_KIND16_QSYMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QSYMM_KERNEL_H

#include <stddef.h>

typedef __float128 qsymm_T;

/* Normalize a Fortran uplo/side char to its uppercase code. */
char qsymm_uplo(const char *p);

/* ── SYMM-aware packers (real) ───────────────────────────────────────
 *
 * Pack a block of the symmetric operand `a` (col-major, leading dim lda,
 * only the UPLO triangle populated — the other triangle is never read)
 * into the qgemm packed layout, mirroring A[row,col] == A[col,row].
 *
 *   qsymm_pack_a_sym → qgemm_pack_A('N') layout (SIDE='L' A-operand).
 *   qsymm_pack_b_sym → qgemm_pack_B('N') layout (SIDE='R' B-operand).
 *
 * `uplo` is 'U' or 'L' (already upper-cased by the caller).
 */
void qsymm_pack_a_sym(const qsymm_T *a, ptrdiff_t lda,
                      ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb,
                      char uplo, qsymm_T *Ap);
void qsymm_pack_b_sym(const qsymm_T *a, ptrdiff_t lda,
                      ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb,
                      char uplo, qsymm_T *Bp);

/* Pure-serial by-value entry (no OpenMP); shares the ptrdiff_t core ABI. */
void qsymm_serial(
    char side, char uplo,
    ptrdiff_t M, ptrdiff_t N,
    const qsymm_T *alpha_,
    const qsymm_T *a, ptrdiff_t lda,
    const qsymm_T *b, ptrdiff_t ldb,
    const qsymm_T *beta_,
    qsymm_T *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_QSYMM_KERNEL_H */
