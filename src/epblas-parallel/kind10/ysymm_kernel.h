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
ptrdiff_t ysymm_nb(void);

/* alpha==0 quick path: C := beta*C over columns [j_start, j_end), rows
 * [0, M). */
void ysymm_beta_only(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m, ysymm_T beta,
                     ysymm_T *c, ptrdiff_t ldc);

/* SIDE='L', M <= nb single-block fast path, one column range [j_start,
 * j_end): inlined scalar ZSYMM with beta folded into the diagonal write. */
void ysymm_L_singleblock(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t m,
                         ysymm_T alpha, ysymm_T beta,
                         const ysymm_T *a, ptrdiff_t lda,
                         const ysymm_T *b, ptrdiff_t ldb,
                         ysymm_T *c, ptrdiff_t ldc, char UPLO);

/* SIDE='L' general path, one column panel [jc, jc+jb): beta pre-scale +
 * I/K block loops (ygemm_serial trailing) + scalar diagonal block. */
void ysymm_L_panel(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t m, ysymm_T alpha, ysymm_T beta,
                   const ysymm_T *a, ptrdiff_t lda, const ysymm_T *b, ptrdiff_t ldb,
                   ysymm_T *c, ptrdiff_t ldc, char UPLO, ptrdiff_t nb);

/* SIDE='R' general path, one row panel [ic, ic+ib): beta pre-scale +
 * J/K block loops (ygemm_serial trailing) + scalar diagonal block. */
void ysymm_R_panel(ptrdiff_t ic, ptrdiff_t ib, ptrdiff_t n, ysymm_T alpha, ysymm_T beta,
                   const ysymm_T *a, ptrdiff_t lda, const ysymm_T *b, ptrdiff_t ldb,
                   ysymm_T *c, ptrdiff_t ldc, char UPLO, ptrdiff_t nb);

/* Pure-serial by-value core (no OpenMP). */
void ysymm_serial(
    char side, char uplo,
    ptrdiff_t m, ptrdiff_t n,
    const ysymm_T *alpha_,
    const ysymm_T *a, ptrdiff_t lda,
    const ysymm_T *b, ptrdiff_t ldb,
    const ysymm_T *beta_,
    ysymm_T *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_YSYMM_KERNEL_H */
