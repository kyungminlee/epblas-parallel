/*
 * xsyrk_kernel.h — internal shared declarations for the kind16 complex
 * (COMPLEX(KIND=16) / __complex128) symmetric rank-k overlay, split across
 * two translation units:
 *
 *   xsyrk_serial.c   — all the math: the per-diagonal-block worker
 *                      (xsyrk_block: beta pre-scale + scalar diagonal add +
 *                      xgemm_serial transpose trailing update), the
 *                      beta-only column scaler, and the pure-serial by-value
 *                      entry `xsyrk_serial`. No `#pragma omp`.
 *   xsyrk_parallel.c — the public Fortran entries (`xsyrk_` LP64 + `xsyrk_64_`
 *                      ILP64) over one `ptrdiff_t` core: threading only (one
 *                      `omp parallel for schedule(dynamic,1)` over the diagonal
 *                      blocks), with an `omp_in_parallel()` guard that delegates
 *                      to `xsyrk_serial` when called from inside another
 *                      routine's parallel region.
 *
 * C is N×N symmetric (NOT Hermitian — no conjugate; see xherk). alpha and
 * beta are complex. The work is partitioned by diagonal block (jc);
 * triangular work is uneven so the parallel driver uses dynamic scheduling.
 * xsyrk_block runs its trailing update through xgemm_serial (NOT xgemm_) so
 * a panel worker never opens a region inside the team xsyrk_parallel.c
 * opened. Mirrors the kind10 ysyrk overlay.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H

#include <stddef.h>
#include <quadmath.h>

typedef __complex128 xsyrk_T;

/* Decode a uplo char to upper-cased 'U'/'L'. */
char xsyrk_uplo(char c);

/* Decode a trans char to upper-cased 'N'/'T'. */
char xsyrk_trans(char c);

/* Env-tunable block size (XSYRK_NB; otherwise 32). */
ptrdiff_t xsyrk_nb(void);

/* One diagonal block [jc, jc+jb): beta pre-scale of the block's columns, the
 * scalar symmetric rank-k diagonal add, and the trailing xgemm_serial
 * transpose update against the rest of the panel. */
void xsyrk_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t N, ptrdiff_t K,
                 xsyrk_T alpha, xsyrk_T beta,
                 const xsyrk_T *a, ptrdiff_t lda, xsyrk_T *c, ptrdiff_t ldc,
                 char UPLO, char TR);

/* C := beta*C over the columns [j_start, j_end) — the alpha==0 / K==0 quick
 * path (and the per-block pre-scale). */
void xsyrk_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t N, xsyrk_T beta,
                      xsyrk_T *c, ptrdiff_t ldc, char UPLO);

/* Pure-serial by-value entry (no OpenMP). Shares the ptrdiff_t core ABI so
 * callers already inside a parallel region can swap the symbol name only. */
void xsyrk_serial(
    char uplo, char trans,
    ptrdiff_t N, ptrdiff_t K,
    const xsyrk_T *alpha_,
    const xsyrk_T *a, ptrdiff_t lda,
    const xsyrk_T *beta_,
    xsyrk_T *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H */
