/*
 * etrsm_kernel.h — internal shared declarations for the kind10 etrsm
 * (REAL(KIND=10) / long double) triangular-solve overlay.
 *
 * The overlay is a faithful L3 pack-and-conquer port of OpenBLAS DTRSM
 * (sibling of the openblas overlay's etrsm.c), split across three TUs:
 *
 *   etrsm_pack.c     — the four diagonal-inverting A-packers
 *                      (etrsm_i{ut,un,lt,ln}copy).
 *   etrsm_kernel.c   — the four solve() directions + the diagonal-aware TRSM
 *                      micro-kernel (etrsm_solve_kernel), paired with the
 *                      shared ob-convention GEMM substrate from etri_kernel.c.
 *   etrsm_serial.c   — the SIDE='L'/'R' band drivers and the pure-serial
 *                      Fortran-ABI entry `etrsm_serial`.
 *   etrsm_parallel.c — the public Fortran entry `etrsm_`: threading
 *                      orchestration only (per-thread Ap/Bp scratch,
 *                      contiguous slice of the free axis), with an
 *                      `omp_in_parallel()` guard that delegates to
 *                      `etrsm_serial` when called from inside another
 *                      routine's parallel region.
 *
 * Why the shared substrate (etri_kernel.c) rather than par's egemm
 * primitives: the TRSM solve and its trailing GEMM share one packed
 * diagonal-block buffer at MR/NR granularity, and OpenBLAS's
 * contiguous-odd-tail packing convention is baked into the packer ↔
 * solve ↔ kernel triad. par's egemm packs odd tails zero-padded at
 * stride MR instead, so its kernel reads those bytes differently (proven
 * mismatch on every odd m/n/k). The triangular routines therefore share a
 * self-consistent ob-convention kernel/packer substrate (etri_kernel.c).
 * The layout-AGNOSTIC helpers ARE reused from the egemm overlay:
 * egemm_choose_blocks (incl. its L2-detected adaptive MC),
 * egemm_beta_prepass, egemm_round_up, egemm_trans_code.
 *
 * Nested calls must run serial: opening a nested OpenMP region trips the
 * libgomp barrier wedge. The entry
 * guard in etrsm_parallel.c is the single nesting gate.
 */
#ifndef EPBLAS_PARALLEL_KIND10_ETRSM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ETRSM_KERNEL_H

#include <stddef.h>

typedef long double etrsm_T;

/* ── Diagonal-inverting A-packers (etrsm_pack.c) ─────────────────────
 * Pack an m×n slab of triangular A into the packed buffer `b`, storing
 * 1/diag on the diagonal register-block and reading only the relevant
 * triangle. `offset` positions the diagonal; `unit` selects unit-diag. */
void etrsm_ilncopy(ptrdiff_t m, ptrdiff_t n, const etrsm_T *a, ptrdiff_t lda,
                   ptrdiff_t offset, etrsm_T *b, ptrdiff_t unit);
void etrsm_iltcopy(ptrdiff_t m, ptrdiff_t n, const etrsm_T *a, ptrdiff_t lda,
                   ptrdiff_t offset, etrsm_T *b, ptrdiff_t unit);
void etrsm_iuncopy(ptrdiff_t m, ptrdiff_t n, const etrsm_T *a, ptrdiff_t lda,
                   ptrdiff_t offset, etrsm_T *b, ptrdiff_t unit);
void etrsm_iutcopy(ptrdiff_t m, ptrdiff_t n, const etrsm_T *a, ptrdiff_t lda,
                   ptrdiff_t offset, etrsm_T *b, ptrdiff_t unit);

/* ── Diagonal-aware TRSM micro-kernel (etrsm_kernel.c) ───────────────
 * Pairs the shared ob-convention GEMM substrate (etri_kernel.h) with the
 * four diagonal solve directions. Self-consistent with the
 * diagonal-inverting packers above (OpenBLAS contiguous-odd-tail
 * convention); see the header note on why the substrate is NOT par's
 * egemm. */
void etrsm_solve_kernel(ptrdiff_t left, ptrdiff_t trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        const etrsm_T *ba, const etrsm_T *bb,
                        etrsm_T *C, ptrdiff_t ldc, ptrdiff_t offset);

/* ── Band drivers (etrsm_serial.c) ───────────────────────────────────
 * Run the full L3 nest for one slice of the partition axis: a column
 * band [js0, js1) of B for SIDE='L', or a row band [m_lo, m_hi) for
 * SIDE='R'. Ap/Bp are caller-owned per-thread scratch. */
void etrsm_L_band(ptrdiff_t upper, ptrdiff_t trans, ptrdiff_t unit,
                  ptrdiff_t M, ptrdiff_t js0, ptrdiff_t js1,
                  ptrdiff_t MC, ptrdiff_t KC, ptrdiff_t NC,
                  const etrsm_T *a, ptrdiff_t lda, etrsm_T *b, ptrdiff_t ldb,
                  etrsm_T *Ap, etrsm_T *Bp);
void etrsm_R_band(ptrdiff_t upper, ptrdiff_t trans, ptrdiff_t unit,
                  ptrdiff_t N, ptrdiff_t m_lo, ptrdiff_t m_hi,
                  ptrdiff_t MC, ptrdiff_t KC, ptrdiff_t NC,
                  const etrsm_T *a, ptrdiff_t lda, etrsm_T *b, ptrdiff_t ldb,
                  etrsm_T *Ap, etrsm_T *Bp);

/* Pure-serial by-value core (no OpenMP). Shares the ptrdiff_t signature
 * with etrsm_core in etrsm_parallel.c. */
void etrsm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t M, ptrdiff_t N,
    const etrsm_T *alpha_,
    const etrsm_T *a, ptrdiff_t lda,
    etrsm_T *b, ptrdiff_t ldb);

#endif /* EPBLAS_PARALLEL_KIND10_ETRSM_KERNEL_H */
