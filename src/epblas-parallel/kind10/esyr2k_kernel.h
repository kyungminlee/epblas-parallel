/*
 * esyr2k_kernel.h — internal shared declarations for the kind10 real
 * (REAL(KIND=10) / long double) symmetric rank-2k overlay, split across two
 * translation units:
 *
 *   esyr2k_serial.c   — all the math: the per-diagonal-block worker
 *                       (esyr2k_block: beta pre-scale + scalar rank-2k
 *                       diagonal + two egemm_serial trailing updates), the
 *                       beta-only column scaler, and the pure-serial
 *                       Fortran-ABI entry `esyr2k_serial`. No `#pragma omp`.
 *   esyr2k_parallel.c — the public Fortran entry `esyr2k_`: threading only
 *                       (one `omp parallel for schedule(dynamic,1)` over the
 *                       diagonal blocks), with an `omp_in_parallel()` guard
 *                       that delegates to `esyr2k_serial` when called from
 *                       inside another routine's parallel region.
 *
 * C is N×N symmetric (only the UPLO triangle is touched). Work is partitioned
 * by diagonal block (jc): the serial entry walks the blocks in a plain loop;
 * the parallel driver hands the same per-block worker to a dynamic-scheduled
 * team. esyr2k_block runs its trailing updates through egemm_serial — opening
 * a nested egemm team would trip the libgomp barrier wedge (see memory
 * project-etrsm-omp4-wedge). The block size is TR-aware (the TR='T' diagonal
 * kernel already saturates the x87 2-stream fadd ceiling, so it runs as a
 * single full-N block with no trailing egemm); both entries compute it the
 * same way.
 */
#ifndef EPBLAS_PARALLEL_KIND10_ESYR2K_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ESYR2K_KERNEL_H

#include <stddef.h>

typedef long double esyr2k_T;

/* Env-tunable block size (ESYR2K_NB, default 64). TR-aware nb is the caller's
 * job: pass N for TR='T', esyr2k_nb() otherwise. */
int esyr2k_nb(void);

/* One diagonal block [jc, jc+jb): beta pre-scale of the block's triangular
 * columns, the scalar rank-2k diagonal add, and the two trailing egemm_serial
 * updates against the rest of the panel. */
void esyr2k_block(int jc, int jb, int N, int K, esyr2k_T alpha, esyr2k_T beta,
                  const esyr2k_T *a, int lda, const esyr2k_T *b, int ldb,
                  esyr2k_T *c, int ldc, char UPLO, char TR);

/* C := beta*C over the triangular columns [j_start, j_end) — the
 * alpha==0 / K==0 quick path. */
void esyr2k_beta_scale(int j_start, int j_end, int N, esyr2k_T beta,
                       esyr2k_T *c, int ldc, char UPLO);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as esyr2k_. */
void esyr2k_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const esyr2k_T *alpha_,
    const esyr2k_T *a, const int *lda_,
    const esyr2k_T *b, const int *ldb_,
    const esyr2k_T *beta_,
    esyr2k_T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len);

#endif /* EPBLAS_PARALLEL_KIND10_ESYR2K_KERNEL_H */
