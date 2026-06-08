/*
 * xsyrk_kernel.h — internal shared declarations for the kind16 complex
 * (COMPLEX(KIND=16) / __complex128) symmetric rank-k overlay, split across
 * two translation units:
 *
 *   xsyrk_serial.c   — all the math: the per-diagonal-block worker
 *                      (xsyrk_block: beta pre-scale + scalar diagonal add +
 *                      xgemm_serial_ transpose trailing update), the
 *                      beta-only column scaler, and the pure-serial
 *                      Fortran-ABI entry `xsyrk_serial_`. No `#pragma omp`.
 *   xsyrk_parallel.c — the public Fortran entry `xsyrk_`: threading only
 *                      (one `omp parallel for schedule(dynamic,1)` over the
 *                      diagonal blocks), with an `omp_in_parallel()` guard
 *                      that delegates to `xsyrk_serial_` when called from
 *                      inside another routine's parallel region.
 *
 * C is N×N symmetric (NOT Hermitian — no conjugate; see xherk). alpha and
 * beta are complex. The work is partitioned by diagonal block (jc);
 * triangular work is uneven so the parallel driver uses dynamic scheduling.
 * xsyrk_block runs its trailing update through xgemm_serial_ (NOT xgemm_) so
 * a panel worker never opens a region inside the team xsyrk_parallel.c
 * opened. Mirrors the kind10 ysyrk overlay.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H

#include <stddef.h>
#include <quadmath.h>

typedef __complex128 xsyrk_T;

/* Decode a Fortran uplo char to upper-cased 'U'/'L'. */
char xsyrk_uplo(const char *p);

/* Decode a Fortran trans char to upper-cased 'N'/'T'. */
char xsyrk_trans(const char *p);

/* Env-tunable block size (XSYRK_NB; otherwise 32). */
ptrdiff_t xsyrk_nb(void);

/* One diagonal block [jc, jc+jb): beta pre-scale of the block's columns, the
 * scalar symmetric rank-k diagonal add, and the trailing xgemm_serial_
 * transpose update against the rest of the panel. */
void xsyrk_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t N, ptrdiff_t K,
                 xsyrk_T alpha, xsyrk_T beta,
                 const xsyrk_T *a, ptrdiff_t lda, xsyrk_T *c, ptrdiff_t ldc,
                 char UPLO, char TR);

/* C := beta*C over the columns [j_start, j_end) — the alpha==0 / K==0 quick
 * path (and the per-block pre-scale). */
void xsyrk_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t N, xsyrk_T beta,
                      xsyrk_T *c, ptrdiff_t ldc, char UPLO);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same int signature as xsyrk_. */
void xsyrk_serial_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const xsyrk_T *alpha_,
    const xsyrk_T *a, const int *lda_,
    const xsyrk_T *beta_,
    xsyrk_T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len);

#endif /* EPBLAS_PARALLEL_KIND16_XSYRK_KERNEL_H */
