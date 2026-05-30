/*
 * ysymm_kernel.h — internal shared declarations for the kind10 complex
 * (COMPLEX(KIND=10) / _Complex long double) symmetric matrix-multiply
 * overlay, split across two translation units:
 *
 *   ysymm_serial.c   — all the math: the alpha==0 beta-only column scaler,
 *                      the SIDE='L' single-diagonal-block fast path (per
 *                      column), the SIDE='L' column-panel worker and the
 *                      SIDE='R' row-panel worker (both with ygemm_serial
 *                      "read A_IK once, use it twice" trailing updates),
 *                      and the pure-serial Fortran-ABI entry `ysymm_serial`.
 *                      No `#pragma omp`.
 *   ysymm_parallel.c — the public Fortran entry `ysymm_`: threading only
 *                      (one `omp parallel for schedule(static)` over the
 *                      outer panel axis — J column panels for SIDE='L', I
 *                      row panels for SIDE='R'), with an `omp_in_parallel()`
 *                      guard that delegates to `ysymm_serial` when called
 *                      from inside another routine's parallel region.
 *
 * Each thread owns a disjoint slice of C's columns (L) or rows (R), so the
 * inner I/K loops run serial inside the worker with no cross-thread races.
 * The trailing updates run through ygemm_serial — opening a nested ygemm
 * team would trip the libgomp barrier wedge (see project-etrsm-omp4-wedge).
 */
#ifndef EPBLAS_PARALLEL_KIND10_YSYMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_YSYMM_KERNEL_H

#include <stddef.h>
#include <complex.h>

typedef _Complex long double ysymm_T;

/* Block/panel size (env YSYMM_NB; otherwise 32). */
int ysymm_nb(void);

/* alpha==0 quick path: C := beta*C over columns [j_start, j_end), rows
 * [0, M). */
void ysymm_beta_only(int j_start, int j_end, int M, ysymm_T beta,
                     ysymm_T *c, int ldc);

/* SIDE='L', M <= nb single-block fast path, one column range [j_start,
 * j_end): inlined scalar ZSYMM with beta folded into the diagonal write. */
void ysymm_L_singleblock(int j_start, int j_end, int M,
                         ysymm_T alpha, ysymm_T beta,
                         const ysymm_T *a, int lda,
                         const ysymm_T *b, int ldb,
                         ysymm_T *c, int ldc, char UPLO);

/* SIDE='L' general path, one column panel [jc, jc+jb): beta pre-scale +
 * I/K block loops (ygemm_serial trailing) + scalar diagonal block. */
void ysymm_L_panel(int jc, int jb, int M, ysymm_T alpha, ysymm_T beta,
                   const ysymm_T *a, int lda, const ysymm_T *b, int ldb,
                   ysymm_T *c, int ldc, char UPLO, int nb);

/* SIDE='R' general path, one row panel [ic, ic+ib): beta pre-scale +
 * J/K block loops (ygemm_serial trailing) + scalar diagonal block. */
void ysymm_R_panel(int ic, int ib, int N, ysymm_T alpha, ysymm_T beta,
                   const ysymm_T *a, int lda, const ysymm_T *b, int ldb,
                   ysymm_T *c, int ldc, char UPLO, int nb);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as ysymm_. */
void ysymm_serial(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const ysymm_T *alpha_,
    const ysymm_T *a, const int *lda_,
    const ysymm_T *b, const int *ldb_,
    const ysymm_T *beta_,
    ysymm_T *c, const int *ldc_,
    size_t side_len, size_t uplo_len);

#endif /* EPBLAS_PARALLEL_KIND10_YSYMM_KERNEL_H */
