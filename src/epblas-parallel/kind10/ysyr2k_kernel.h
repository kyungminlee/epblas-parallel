/*
 * ysyr2k_kernel.h — internal shared declarations for the kind10 complex
 * (COMPLEX(KIND=10) / _Complex long double) symmetric rank-2k overlay,
 * split across two translation units:
 *
 *   ysyr2k_serial.c   — all the math: the per-diagonal-block worker
 *                       (ysyr2k_block: beta pre-scale + scalar rank-2k
 *                       diagonal + two ygemm_serial trailing updates), the
 *                       beta-only column scaler, and the pure-serial
 *                       Fortran-ABI entry `ysyr2k_serial`. No `#pragma omp`.
 *   ysyr2k_parallel.c — the public Fortran entry `ysyr2k_`: threading only
 *                       (one `omp parallel for schedule(dynamic,1)` over the
 *                       diagonal blocks), with an `omp_in_parallel()` guard
 *                       that delegates to `ysyr2k_serial` when called from
 *                       inside another routine's parallel region.
 *
 * C is complex SYMMETRIC (not Hermitian — see yher2k). Work is partitioned
 * by diagonal block (jc): the serial entry walks the blocks in a plain loop;
 * the parallel driver hands the same per-block worker to a dynamic-scheduled
 * team. ysyr2k_block runs its trailing updates through ygemm_serial —
 * opening a nested ygemm team would trip the libgomp barrier wedge.
 */
#ifndef EPBLAS_PARALLEL_KIND10_YSYR2K_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YSYR2K_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double ysyr2k_T;

/* Env-tunable block size (YSYR2K_NB). */
ptrdiff_t ysyr2k_nb(void);

/* One diagonal block [jc, jc+jb): beta pre-scale of the block's triangular
 * columns, the scalar rank-2k diagonal add, and the two trailing
 * ygemm_serial updates against the rest of the panel. */
void ysyr2k_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t N, ptrdiff_t K, ysyr2k_T alpha, ysyr2k_T beta,
                  const ysyr2k_T *a, ptrdiff_t lda, const ysyr2k_T *b, ptrdiff_t ldb,
                  ysyr2k_T *c, ptrdiff_t ldc, char UPLO, char TR);

/* C := beta*C over the triangular columns [j_start, j_end) — the
 * alpha==0 / K==0 quick path. */
void ysyr2k_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t N, ysyr2k_T beta,
                       ysyr2k_T *c, ptrdiff_t ldc, char UPLO);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as ysyr2k_. */
void ysyr2k_serial(
    const char *uplo, const char *trans,
    const ptrdiff_t *n_, const ptrdiff_t *k_,
    const ysyr2k_T *alpha_,
    const ysyr2k_T *a, const ptrdiff_t *lda_,
    const ysyr2k_T *b, const ptrdiff_t *ldb_,
    const ysyr2k_T *beta_,
    ysyr2k_T *c, const ptrdiff_t *ldc_,
    size_t uplo_len, size_t trans_len);

#endif /* EPBLAS_PARALLEL_KIND10_YSYR2K_KERNEL_H */
