/*
 * ysyrk_kernel.h — internal shared declarations for the kind10 complex
 * (COMPLEX(KIND=10) / _Complex long double) symmetric rank-k overlay,
 * split across two translation units:
 *
 *   ysyrk_serial.c   — all the math: the per-diagonal-block worker
 *                      (ysyrk_block: beta pre-scale + scalar diagonal +
 *                      ygemm_serial trailing update), the beta-only column
 *                      scaler, and the pure-serial Fortran-ABI entry
 *                      `ysyrk_serial`. No `#pragma omp`.
 *   ysyrk_parallel.c — the public Fortran entry `ysyrk_`: threading only
 *                      (one `omp parallel for schedule(dynamic,1)` over the
 *                      diagonal blocks), with an `omp_in_parallel()` guard
 *                      that delegates to `ysyrk_serial` when called from
 *                      inside another routine's parallel region.
 *
 * The work is partitioned by diagonal block (jc): the serial entry walks
 * the blocks in a plain loop; the parallel driver hands the same per-block
 * worker to a dynamic-scheduled team (triangular work is uneven, so
 * dynamic balances better than static). ysyrk_block runs its trailing
 * update through ygemm_serial — opening a nested ygemm team would trip the
 * libgomp barrier wedge (see memory project-etrsm-omp4-wedge).
 */
#ifndef EPBLAS_PARALLEL_KIND10_YSYRK_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YSYRK_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double ysyrk_T;

/* Env-tunable block size (YSYRK_NB). */
int ysyrk_nb(void);

/* One diagonal block [jc, jc+jb): beta pre-scale of the block's triangular
 * columns, the scalar diagonal rank-k add, and the trailing ygemm_serial
 * update against the rest of the panel. */
void ysyrk_block(int jc, int jb, int N, int K, ysyrk_T alpha, ysyrk_T beta,
                 const ysyrk_T *a, int lda, ysyrk_T *c, int ldc,
                 char UPLO, char TR);

/* C := beta*C over the triangular columns [j_start, j_end) — the
 * alpha==0 / K==0 quick path. */
void ysyrk_beta_scale(int j_start, int j_end, int N, ysyrk_T beta,
                      ysyrk_T *c, int ldc, char UPLO);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as ysyrk_. */
void ysyrk_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const ysyrk_T *alpha_,
    const ysyrk_T *a, const int *lda_,
    const ysyrk_T *beta_,
    ysyrk_T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len);

#endif /* EPBLAS_PARALLEL_KIND10_YSYRK_KERNEL_H */
