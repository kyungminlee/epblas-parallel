/*
 * etrmm_kernel.h — internal shared declarations for the kind10 etrmm
 * (REAL(KIND=10) / long double) triangular matrix-multiply overlay.
 *
 *   B := alpha · op(A) · B   (SIDE='L')   or   B := alpha · B · op(A) (SIDE='R')
 *
 * The overlay is a faithful L3 pack-and-conquer port of OpenBLAS DTRMM
 * (sibling of the openblas overlay's etrmm.c), split across four TUs:
 *
 *   etrmm_pack.c     — the four TRMM A-packers (etrmm_i{ut,un,lt,ln}copy).
 *   etrmm_kernel.c   — the diagonal-aware TRMM micro-kernel (etrmm_kernel),
 *                      paired with the shared ob-convention GEMM substrate
 *                      from etri_kernel.c.
 *   etrmm_serial.c   — the SIDE='L'/'R' band drivers and the pure-serial
 *                      Fortran-ABI entry `etrmm_serial`.
 *   etrmm_parallel.c — the public Fortran entry `etrmm_`: threading
 *                      orchestration only (per-thread Ap/Bp scratch,
 *                      contiguous slice of the free axis), with an
 *                      `omp_in_parallel()` guard that delegates to
 *                      `etrmm_serial` when called from inside another
 *                      routine's parallel region (libgomp barrier-wedge
 *                      guard, memory project-etrsm-omp4-wedge).
 *
 * TRMM as alpha-prescale + overwrite nest: alpha pre-scales B in place
 * (egemm_beta_prepass), then the L3 nest runs with kernel-alpha = 1.0L,
 * overwriting B tile by tile. Since the spec is linear in B, the result is
 * alpha · op(A) · B_old.
 *
 * Shares the ob-convention GEMM substrate (etri_kernel.h) with etrsm — see
 * that header for why the triangular routines carry a private substrate
 * rather than reusing par's egemm primitives (contiguous-odd-tail packing).
 */
#ifndef EPBLAS_PARALLEL_KIND10_ETRMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ETRMM_KERNEL_H

#include <stddef.h>

typedef long double etrmm_T;

/* ── TRMM A-packers (etrmm_pack.c) ───────────────────────────────────
 * Pack the relevant triangle of A (plus the diagonal) into the packed
 * buffer `b` in the ob contiguous-odd-tail convention. `posX`/`posY`
 * position the diagonal; `unit` selects unit-diagonal. */
void etrmm_iutcopy(ptrdiff_t m, ptrdiff_t n, const etrmm_T *a, ptrdiff_t lda,
                   ptrdiff_t posX, ptrdiff_t posY, etrmm_T *b, int unit);
void etrmm_iuncopy(ptrdiff_t m, ptrdiff_t n, const etrmm_T *a, ptrdiff_t lda,
                   ptrdiff_t posX, ptrdiff_t posY, etrmm_T *b, int unit);
void etrmm_iltcopy(ptrdiff_t m, ptrdiff_t n, const etrmm_T *a, ptrdiff_t lda,
                   ptrdiff_t posX, ptrdiff_t posY, etrmm_T *b, int unit);
void etrmm_ilncopy(ptrdiff_t m, ptrdiff_t n, const etrmm_T *a, ptrdiff_t lda,
                   ptrdiff_t posX, ptrdiff_t posY, etrmm_T *b, int unit);

/* ── Diagonal-aware TRMM micro-kernel (etrmm_kernel.c) ───────────────
 * C := alpha · ba · bb (overwrite) over one packed (bm,bn,bk) tile; the
 * (left, trans) flags select TRMM_KERNEL_{LN,LT,RN,RT}. `offset` positions
 * the diagonal within the tile. Self-consistent with the etrmm_i*copy
 * packers and the etri_kernel.c substrate (contiguous-odd-tail). */
void etrmm_kernel(int left, int trans,
                  ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk, etrmm_T alpha,
                  const etrmm_T *ba, const etrmm_T *bb,
                  etrmm_T *C, ptrdiff_t ldc, ptrdiff_t offset);

/* ── Band drivers (etrmm_serial.c) ───────────────────────────────────
 * Run the full L3 nest for one slice of the partition axis: a column band
 * [js0, js1) of B for SIDE='L', or a row band [m_lo, m_hi) for SIDE='R'.
 * Ap/Bp are caller-owned per-thread scratch. */
void etrmm_L_band(int upper, int trans, int unit,
                  int M, int js0, int js1,
                  int MC, int KC, int NC,
                  const etrmm_T *a, int lda, etrmm_T *b, int ldb,
                  etrmm_T *Ap, etrmm_T *Bp);
void etrmm_R_band(int upper, int trans, int unit,
                  int N, int m_lo, int m_hi,
                  int MC, int KC, int NC,
                  const etrmm_T *a, int lda, etrmm_T *b, int ldb,
                  etrmm_T *Ap, etrmm_T *Bp);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as etrmm_. */
void etrmm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const etrmm_T *alpha_,
    const etrmm_T *a, const int *lda_,
    etrmm_T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len);

#endif /* EPBLAS_PARALLEL_KIND10_ETRMM_KERNEL_H */
