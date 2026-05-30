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
 * the libgomp barrier wedge (see memory project-etrsm-omp4-wedge).
 */
#ifndef EPBLAS_PARALLEL_KIND10_YHEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YHEMM_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double yhemm_T;

/* Block/panel size (env YHEMM_NB; otherwise 32). */
int yhemm_nb(void);

/* alpha==0 quick path: C := beta*C over columns [j_start, j_end), rows
 * [0, M). */
void yhemm_beta_only(int j_start, int j_end, int M, yhemm_T beta,
                     yhemm_T *c, int ldc);

/* SIDE='L', M <= nb single-block fast path, one column range [j_start,
 * j_end): inlined scalar ZHEMM with beta folded into the diagonal write. */
void yhemm_L_singleblock(int j_start, int j_end, int M,
                         yhemm_T alpha, yhemm_T beta,
                         const yhemm_T *a, int lda,
                         const yhemm_T *b, int ldb,
                         yhemm_T *c, int ldc, char UPLO);

/* SIDE='L' general path, one column panel [jc, jc+jb). */
void yhemm_L_panel(int jc, int jb, int M, yhemm_T alpha, yhemm_T beta,
                   const yhemm_T *a, int lda, const yhemm_T *b, int ldb,
                   yhemm_T *c, int ldc, char UPLO, int nb);

/* SIDE='R' general path, one row panel [ic, ic+ib). */
void yhemm_R_panel(int ic, int ib, int N, yhemm_T alpha, yhemm_T beta,
                   const yhemm_T *a, int lda, const yhemm_T *b, int ldb,
                   yhemm_T *c, int ldc, char UPLO, int nb);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as yhemm_. */
void yhemm_serial(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const yhemm_T *alpha_,
    const yhemm_T *a, const int *lda_,
    const yhemm_T *b, const int *ldb_,
    const yhemm_T *beta_,
    yhemm_T *c, const int *ldc_,
    size_t side_len, size_t uplo_len);

#endif /* EPBLAS_PARALLEL_KIND10_YHEMM_KERNEL_H */
