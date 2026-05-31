/*
 * etrsm_kernel.h — internal shared declarations for the kind10 etrsm
 * (REAL(KIND=10) / long double) triangular-solve overlay.
 *
 * The overlay is a faithful L3 pack-and-conquer port of OpenBLAS DTRSM
 * (sibling of the openblas overlay's etrsm.c), split across three TUs:
 *
 *   etrsm_pack.c     — the four diagonal-inverting A-packers
 *                      (etrsm_i{ut,un,lt,ln}copy).
 *   etrsm_serial.c   — the ob-convention L3 substrate (a private MR=NR=2
 *                      GEMM micro-kernel + ncopy/tcopy + the four solve()
 *                      directions + the diagonal-aware TRSM micro-kernel,
 *                      all static), the SIDE='L'/'R' band drivers, and the
 *                      pure-serial Fortran-ABI entry `etrsm_serial`.
 *   etrsm_parallel.c — the public Fortran entry `etrsm_`: threading
 *                      orchestration only (per-thread Ap/Bp scratch,
 *                      contiguous slice of the free axis), with an
 *                      `omp_in_parallel()` guard that delegates to
 *                      `etrsm_serial` when called from inside another
 *                      routine's parallel region.
 *
 * Why a private GEMM micro-kernel rather than reusing par's egemm
 * primitives: the TRSM solve and its trailing GEMM share one packed
 * diagonal-block buffer at MR/NR granularity, and OpenBLAS's
 * contiguous-odd-tail packing convention is baked into the packer ↔
 * solve ↔ kernel triad. par's egemm packs odd tails zero-padded at
 * stride MR instead, so its kernel reads those bytes differently (proven
 * mismatch on every odd m/n/k). The substrate therefore carries its own
 * self-consistent ob-convention kernel/packers. The layout-AGNOSTIC
 * helpers ARE reused from the egemm overlay: egemm_choose_blocks (incl.
 * its L2-detected adaptive MC), egemm_beta_prepass, egemm_round_up,
 * egemm_trans_code.
 *
 * Nested calls must run serial: opening a nested OpenMP region trips the
 * libgomp barrier wedge (see memory project-etrsm-omp4-wedge). The entry
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
                   ptrdiff_t offset, etrsm_T *b, int unit);
void etrsm_iltcopy(ptrdiff_t m, ptrdiff_t n, const etrsm_T *a, ptrdiff_t lda,
                   ptrdiff_t offset, etrsm_T *b, int unit);
void etrsm_iuncopy(ptrdiff_t m, ptrdiff_t n, const etrsm_T *a, ptrdiff_t lda,
                   ptrdiff_t offset, etrsm_T *b, int unit);
void etrsm_iutcopy(ptrdiff_t m, ptrdiff_t n, const etrsm_T *a, ptrdiff_t lda,
                   ptrdiff_t offset, etrsm_T *b, int unit);

/* ── ob-convention L3 substrate (etrsm_kernel.c) ─────────────────────
 * A private MR=NR=2 GEMM micro-kernel and its matching ncopy/tcopy
 * packers, plus the diagonal-aware TRSM micro-kernel. Self-consistent
 * with the diagonal-inverting packers above (OpenBLAS contiguous-odd-tail
 * convention); see the header note on why these are NOT par's egemm
 * primitives. `etrsm_gemm_kernel` computes C += alpha·Ap·Bp over one
 * packed (bm,bn,bk) tile. */
void etrsm_gemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk, etrsm_T alpha,
                       const etrsm_T *Ap, const etrsm_T *Bp,
                       etrsm_T *C, ptrdiff_t ldc);
void etrsm_ncopy(ptrdiff_t m, ptrdiff_t n, const etrsm_T *a, ptrdiff_t lda,
                 etrsm_T *b);
void etrsm_tcopy(ptrdiff_t m, ptrdiff_t n, const etrsm_T *a, ptrdiff_t lda,
                 etrsm_T *b);
void etrsm_solve_kernel(int left, int trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        const etrsm_T *ba, const etrsm_T *bb,
                        etrsm_T *C, ptrdiff_t ldc, ptrdiff_t offset);

/* ── Band drivers (etrsm_serial.c) ───────────────────────────────────
 * Run the full L3 nest for one slice of the partition axis: a column
 * band [js0, js1) of B for SIDE='L', or a row band [m_lo, m_hi) for
 * SIDE='R'. Ap/Bp are caller-owned per-thread scratch. */
void etrsm_L_band(int upper, int trans, int unit,
                  int M, int js0, int js1,
                  int MC, int KC, int NC,
                  const etrsm_T *a, int lda, etrsm_T *b, int ldb,
                  etrsm_T *Ap, etrsm_T *Bp);
void etrsm_R_band(int upper, int trans, int unit,
                  int N, int m_lo, int m_hi,
                  int MC, int KC, int NC,
                  const etrsm_T *a, int lda, etrsm_T *b, int ldb,
                  etrsm_T *Ap, etrsm_T *Bp);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as etrsm_. */
void etrsm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const etrsm_T *alpha_,
    const etrsm_T *a, const int *lda_,
    etrsm_T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len);

#endif /* EPBLAS_PARALLEL_KIND10_ETRSM_KERNEL_H */
