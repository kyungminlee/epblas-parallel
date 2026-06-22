/*
 * xherk_kernel.h — internal shared declarations for the kind16 complex
 * (COMPLEX(KIND=16) / __complex128) Hermitian rank-k overlay, split across
 * two translation units:
 *
 *   xherk_serial.c   — all the math: the per-diagonal-block worker
 *                      (xherk_block: beta pre-scale keeping the diagonal
 *                      real + scalar Hermitian diagonal add + xgemm_serial
 *                      conjugate-transpose trailing update), the beta-only
 *                      column scaler, and the pure-serial by-value core
 *                      `xherk_serial`. No `#pragma omp`.
 *   xherk_parallel.c — the public Fortran entries `xherk_`/`xherk_64_`
 *                      (LP64/ILP64 facades over a shared ptrdiff_t core):
 *                      threading only (one `omp parallel for
 *                      schedule(dynamic,1)` over the diagonal blocks), with an
 *                      `omp_in_parallel()` guard that delegates to
 *                      `xherk_serial` when called from inside another
 *                      routine's parallel region.
 *
 * alpha and beta are REAL; the diagonal of C stays real on output. The work
 * is partitioned by diagonal block (jc); triangular work is uneven so the
 * parallel driver uses dynamic scheduling. xherk_block runs its trailing
 * update through xgemm_serial_ (NOT xgemm_) so a panel worker never opens a
 * region inside the team xherk_parallel.c opened. Mirrors the kind10 yherk
 * overlay (__complex128 lowers to libquadmath soft-float calls).
 */
#ifndef EPBLAS_PARALLEL_KIND16_XHERK_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XHERK_KERNEL_H

#include <stddef.h>
#include <quadmath.h>

typedef __complex128 xherk_TC;   /* matrices A, C */
typedef __float128   xherk_TR;   /* scalars alpha, beta */

/* Decode an uplo char to upper-cased 'U'/'L'. */
char xherk_uplo(char c);

/* Decode a trans char to upper-cased 'N'/'C'. */
char xherk_trans(char c);

/* Env-tunable block size (XHERK_NB; otherwise 32). */
ptrdiff_t xherk_nb(void);

/* One diagonal block [jc, jc+jb): beta pre-scale of the block's columns
 * (diagonal kept real), the scalar Hermitian diagonal rank-k add, and the
 * trailing xgemm_serial conjugate-transpose update against the rest of the
 * panel. */
void xherk_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t n, ptrdiff_t k,
                 xherk_TR alpha, xherk_TR beta,
                 const xherk_TC *a, ptrdiff_t lda, xherk_TC *c, ptrdiff_t ldc,
                 char UPLO, char TR_c);

/* C := beta*C over the columns [j_start, j_end) keeping the diagonal real —
 * the alpha==0 / K==0 quick path (and the per-block pre-scale). beta==1
 * realifies only the diagonal entry. */
void xherk_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, xherk_TR beta,
                      xherk_TC *c, ptrdiff_t ldc, char UPLO);

/* Pure-serial by-value core (no OpenMP). */
void xherk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const xherk_TR *alpha_,
    const xherk_TC *a, ptrdiff_t lda,
    const xherk_TR *beta_,
    xherk_TC *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_XHERK_KERNEL_H */
