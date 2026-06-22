/*
 * yherk_kernel.h — internal shared declarations for the kind10 complex
 * (COMPLEX(KIND=10) / _Complex long double) Hermitian rank-k overlay,
 * split across two translation units:
 *
 *   yherk_serial.c   — all the math: the per-diagonal-block worker
 *                      (yherk_block: beta pre-scale keeping the diagonal
 *                      real + scalar Hermitian diagonal add + ygemm_serial
 *                      conjugate-transpose trailing update), the beta-only
 *                      column scaler, and the pure-serial Fortran-ABI entry
 *                      `yherk_serial`. No `#pragma omp`.
 *   yherk_parallel.c — the public Fortran entry `yherk_`: threading only
 *                      (one `omp parallel for schedule(dynamic,1)` over the
 *                      diagonal blocks), with an `omp_in_parallel()` guard
 *                      that delegates to `yherk_serial` when called from
 *                      inside another routine's parallel region.
 *
 * alpha and beta are REAL; the diagonal of C stays real on output. The work
 * is partitioned by diagonal block (jc): the serial entry walks the blocks
 * in a plain loop; the parallel driver hands the same per-block worker to a
 * dynamic-scheduled team (triangular work is uneven, so dynamic balances
 * better than static). yherk_block runs its trailing update through
 * ygemm_serial — opening a nested ygemm team would trip the libgomp barrier
 * wedge.
 */
#ifndef EPBLAS_PARALLEL_KIND10_YHERK_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YHERK_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double yherk_TC;
typedef long double          yherk_TR;

/* Env-tunable block size (YHERK_NB). */
ptrdiff_t yherk_nb(void);

/* One diagonal block [jc, jc+jb): beta pre-scale of the block's columns
 * (diagonal kept real), the scalar Hermitian diagonal rank-k add, and the
 * trailing ygemm_serial conjugate-transpose update against the rest of the
 * panel. */
void yherk_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t n, ptrdiff_t k, yherk_TR alpha, yherk_TR beta,
                 const yherk_TC *a, ptrdiff_t lda, yherk_TC *c, ptrdiff_t ldc,
                 char UPLO, char TR_c);

/* C := beta*C over the columns [j_start, j_end) keeping the diagonal real —
 * the alpha==0 / K==0 quick path (and the per-block pre-scale). beta==1
 * realifies only the diagonal entry. */
void yherk_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, yherk_TR beta,
                      yherk_TC *c, ptrdiff_t ldc, char UPLO);

/* Pure-serial by-value core (no OpenMP). */
void yherk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const yherk_TR *alpha_,
    const yherk_TC *a, ptrdiff_t lda,
    const yherk_TR *beta_,
    yherk_TC *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_YHERK_KERNEL_H */
