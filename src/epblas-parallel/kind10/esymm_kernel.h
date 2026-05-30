/*
 * esymm_kernel.h — internal shared declarations for the kind10 real
 * (REAL(KIND=10) / long double) symmetric matrix-multiply overlay, split
 * across two translation units:
 *
 *   esymm_serial.c   — all the math: the alpha==0 beta-only column scaler,
 *                      the SIDE='L' single-diagonal-block fast path (per
 *                      column), the SIDE='L' column-panel worker and the
 *                      SIDE='R' row-panel worker (both with egemm_serial
 *                      "read A_IK once, use it twice" trailing updates),
 *                      and the pure-serial Fortran-ABI entry `esymm_serial`.
 *                      No `#pragma omp`.
 *   esymm_parallel.c — the public Fortran entry `esymm_`: threading only
 *                      (one `omp parallel for schedule(static)` over the
 *                      outer panel axis — J column panels for SIDE='L', I
 *                      row panels for SIDE='R'), with an `omp_in_parallel()`
 *                      guard that delegates to `esymm_serial` when called
 *                      from inside another routine's parallel region.
 *
 * Each thread owns a disjoint slice of C's columns (L) or rows (R), so the
 * inner I/K loops run serial inside the worker with no cross-thread races.
 * The trailing updates run through egemm_serial — opening a nested egemm
 * team would trip the libgomp barrier wedge (see project-etrsm-omp4-wedge).
 */
#ifndef EPBLAS_PARALLEL_KIND10_ESYMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ESYMM_KERNEL_H

#include <stddef.h>

typedef long double esymm_T;

/* Block/panel size (env ESYMM_NB; otherwise 64). */
int esymm_nb(void);

/* alpha==0 quick path: C := beta*C over columns [j_start, j_end), rows
 * [0, M). */
void esymm_beta_only(int j_start, int j_end, int M, esymm_T beta,
                     esymm_T *c, int ldc);

/* SIDE='L', M <= nb single-block fast path, one column range [j_start,
 * j_end): inlined scalar DSYMM with beta folded into the diagonal write. */
void esymm_L_singleblock(int j_start, int j_end, int M,
                         esymm_T alpha, esymm_T beta,
                         const esymm_T *a, int lda,
                         const esymm_T *b, int ldb,
                         esymm_T *c, int ldc, char UPLO);

/* SIDE='L' general path, one column panel [jc, jc+jb): beta pre-scale +
 * I/K block loops (egemm_serial trailing) + scalar diagonal block. */
void esymm_L_panel(int jc, int jb, int M, esymm_T alpha, esymm_T beta,
                   const esymm_T *a, int lda, const esymm_T *b, int ldb,
                   esymm_T *c, int ldc, char UPLO, int nb);

/* SIDE='R' general path, one row panel [ic, ic+ib): beta pre-scale +
 * J/K block loops (egemm_serial trailing) + scalar diagonal block. */
void esymm_R_panel(int ic, int ib, int N, esymm_T alpha, esymm_T beta,
                   const esymm_T *a, int lda, const esymm_T *b, int ldb,
                   esymm_T *c, int ldc, char UPLO, int nb);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as esymm_. */
void esymm_serial(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const esymm_T *alpha_,
    const esymm_T *a, const int *lda_,
    const esymm_T *b, const int *ldb_,
    const esymm_T *beta_,
    esymm_T *c, const int *ldc_,
    size_t side_len, size_t uplo_len);

#endif /* EPBLAS_PARALLEL_KIND10_ESYMM_KERNEL_H */
