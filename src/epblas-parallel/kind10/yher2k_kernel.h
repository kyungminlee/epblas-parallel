/*
 * yher2k_kernel.h — internal shared declarations for the kind10 complex
 * (COMPLEX(KIND=10) / _Complex long double) Hermitian rank-2k overlay,
 * split across two translation units:
 *
 *   yher2k_serial.c   — all the math: the per-diagonal-block worker
 *                       (yher2k_block: beta pre-scale keeping the diagonal
 *                       real + scalar Hermitian rank-2 diagonal add + two
 *                       ygemm_serial conjugate-transpose trailing updates),
 *                       the beta-only column scaler, and the pure-serial
 *                       Fortran-ABI entry `yher2k_serial`. No `#pragma omp`.
 *   yher2k_parallel.c — the public Fortran entry `yher2k_`: threading only
 *                       (one `omp parallel for schedule(dynamic,1)` over the
 *                       diagonal blocks), with an `omp_in_parallel()` guard
 *                       that delegates to `yher2k_serial` when called from
 *                       inside another routine's parallel region.
 *
 * alpha is COMPLEX, beta is REAL; the diagonal of C stays real on output.
 * Work is partitioned by diagonal block (jc): the serial entry walks the
 * blocks in a plain loop; the parallel driver hands the same per-block
 * worker to a dynamic-scheduled team. yher2k_block runs its trailing updates
 * through ygemm_serial — opening a nested ygemm team would trip the libgomp
 * barrier wedge.
 */
#ifndef EPBLAS_PARALLEL_KIND10_YHER2K_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YHER2K_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double yher2k_TC;
typedef long double          yher2k_TR;

/* Env-tunable block size (YHER2K_NB). */
ptrdiff_t yher2k_nb(void);

/* One diagonal block [jc, jc+jb): beta pre-scale of the block's columns
 * (diagonal kept real), the scalar Hermitian rank-2 diagonal add, and the
 * two trailing ygemm_serial conjugate-transpose updates. */
void yher2k_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t n, ptrdiff_t k, yher2k_TC alpha, yher2k_TR beta,
                  const yher2k_TC *a, ptrdiff_t lda, const yher2k_TC *b, ptrdiff_t ldb,
                  yher2k_TC *c, ptrdiff_t ldc, char UPLO, char TRANS);

/* C := beta*C over the columns [j_start, j_end) keeping the diagonal real —
 * the alpha==0 / K==0 quick path (and the per-block pre-scale). beta==1
 * realifies only the diagonal entry. */
void yher2k_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, yher2k_TR beta,
                       yher2k_TC *c, ptrdiff_t ldc, char UPLO);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as yher2k_. */
void yher2k_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const yher2k_TC *alpha_,
    const yher2k_TC *a, ptrdiff_t lda,
    const yher2k_TC *b, ptrdiff_t ldb,
    const yher2k_TR *beta_,
    yher2k_TC *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_YHER2K_KERNEL_H */
