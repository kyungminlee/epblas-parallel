/*
 * yhemm_kernel.h — internal shared declarations for the kind10 complex
 * (COMPLEX(KIND=10) / _Complex long double) Hermitian matrix-multiply
 * overlay, split across two translation units:
 *
 *   yhemm_serial.c   — all the math: the alpha==0 beta-only column scaler,
 *                      the SIDE='L' single-diagonal-block fast path (per
 *                      column), the SIDE='L' column-panel worker and the
 *                      SIDE='R' row-panel worker (both with ygemm_serial
 *                      "read A_IK once, use it twice" trailing updates, the
 *                      reflection leg using 'C'), and the pure-serial
 *                      Fortran-ABI entry `yhemm_serial`. No `#pragma omp`.
 *   yhemm_parallel.c — the public Fortran entry `yhemm_`: threading only
 *                      (one `omp parallel for schedule(static)` over the
 *                      outer panel axis), with an `omp_in_parallel()` guard
 *                      that delegates to `yhemm_serial` when called from
 *                      inside another routine's parallel region.
 *
 * Same blocked structure as ysymm, but Hermitian: the reflection gemm uses
 * 'C' and the scalar diagonal block keeps the diagonal real. The trailing
 * updates run through ygemm_serial — opening a nested ygemm team would trip
 * the libgomp barrier wedge.
 */
#ifndef EPBLAS_PARALLEL_KIND10_YHEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YHEMM_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double yhemm_TC;

/* Block/panel size (env YHEMM_NB; otherwise 32). */
ptrdiff_t yhemm_nb(void);

/* alpha==0 quick path: C := beta*C over columns [j_start, j_end), rows
 * [0, M). */
void yhemm_beta_only(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, yhemm_TC beta,
                     yhemm_TC *c, ptrdiff_t ldc);

/* SIDE='L', M <= nb single-block fast path, one column range [j_start,
 * j_end): inlined scalar ZHEMM with beta folded into the diagonal write. */
void yhemm_L_singleblock(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m,
                         yhemm_TC alpha, yhemm_TC beta,
                         const yhemm_TC *a, ptrdiff_t lda,
                         const yhemm_TC *b, ptrdiff_t ldb,
                         yhemm_TC *c, ptrdiff_t ldc, char UPLO);

/* SIDE='L' general path, one column panel [jc, jc+jb). */
void yhemm_L_panel(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t m, yhemm_TC alpha, yhemm_TC beta,
                   const yhemm_TC *a, ptrdiff_t lda, const yhemm_TC *b, ptrdiff_t ldb,
                   yhemm_TC *c, ptrdiff_t ldc, char UPLO, ptrdiff_t nb);

/* SIDE='R' general path, one row panel [ic, ic+ib). */
void yhemm_R_panel(ptrdiff_t ic, ptrdiff_t ib, ptrdiff_t n, yhemm_TC alpha, yhemm_TC beta,
                   const yhemm_TC *a, ptrdiff_t lda, const yhemm_TC *b, ptrdiff_t ldb,
                   yhemm_TC *c, ptrdiff_t ldc, char UPLO, ptrdiff_t nb);

/* Pure-serial by-value core (no OpenMP). */
void yhemm_serial(
    char side, char uplo,
    ptrdiff_t m, ptrdiff_t n,
    const yhemm_TC *alpha_,
    const yhemm_TC *a, ptrdiff_t lda,
    const yhemm_TC *b, ptrdiff_t ldb,
    const yhemm_TC *beta_,
    yhemm_TC *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_YHEMM_KERNEL_H */
