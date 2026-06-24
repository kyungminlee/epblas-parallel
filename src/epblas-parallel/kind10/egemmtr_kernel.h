/*
 * egemmtr_kernel.h — internal kernel surface shared by the two translation
 * units the kind10 egemmtr overlay is split across:
 *
 *   egemmtr_serial.c    The pure single-thread triangular GEMM-update (no
 *                       OpenMP). Owns ALL the math — packers, MR×NR
 *                       micro-kernels, the rectangular and triangle-aware
 *                       macro kernels, the triangle beta-scale, block-size
 *                       policy — and the public `egemmtr_serial` entry.
 *                       Called by egemmtr_ as its serial branch (and by any
 *                       L3 routine that needs the serial path).
 *
 *                       Owns `egemmtr_scalar_fallback_cols`, the buffer-free
 *                       per-column-band path the threaded entry uses on OOM.
 *
 *   egemmtr_parallel.c  The public Fortran entry `egemmtr_` — threading
 *                       orchestration only. Delegates to egemmtr_serial when
 *                       called from inside a parallel region (the libgomp
 *                       barrier wedge guard, project-etrsm-omp4-wedge);
 *                       otherwise fans these same kernel pieces across an
 *                       OpenMP team by COLUMN PANEL (each thread private Ap+Bp,
 *                       `omp for schedule(dynamic,1)` over jc, no barrier).
 */
#ifndef EPBLAS_PARALLEL_KIND10_EGEMMTR_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_EGEMMTR_KERNEL_H

#include <stddef.h>

typedef long double egemmtr_TR;

/* Register-tile dims. */
#define EGEMMTR_MR 2
#define EGEMMTR_NR 2

/* TN orientation at or below this N uses the unpacked stride-1 path (no GotoBLAS
 * pack — pure overhead when the problem is L2-resident). */
#define EGEMMTR_UNPACKED_TN_MAX 64



/* Cache-block sizes (env-overridable EGEMMTR_MC/KC/NC); lazy-initialized. */
void egemmtr_block_sizes(ptrdiff_t *MC, ptrdiff_t *KC, ptrdiff_t *NC);

/* C := beta*C over the UPLO triangle of columns [j_start, j_end). */
void egemmtr_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, char UPLO,
                        egemmtr_TR beta, egemmtr_TR *c, ptrdiff_t ldc);

/* Packers (egemm-style, take separate A and B). */
void egemmtr_pack_A(const egemmtr_TR *restrict A, ptrdiff_t lda,
                    ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb, char ta,
                    egemmtr_TR *restrict Ap);
void egemmtr_pack_B(const egemmtr_TR *restrict B, ptrdiff_t ldb,
                    ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb, char tb,
                    egemmtr_TR *restrict Bp);

/* Rectangular macro-tile (entirely inside the stored triangle). */
void egemmtr_macro_kernel_rect(ptrdiff_t ib, ptrdiff_t jb, ptrdiff_t pb, egemmtr_TR alpha,
                               const egemmtr_TR *restrict Ap,
                               const egemmtr_TR *restrict Bp,
                               egemmtr_TR *restrict C, ptrdiff_t ldc);

/* Triangle-aware macro-tile (crosses the diagonal). */
void egemmtr_macro_kernel_tri(ptrdiff_t ib, ptrdiff_t jb, ptrdiff_t pb, egemmtr_TR alpha,
                              const egemmtr_TR *restrict Ap,
                              const egemmtr_TR *restrict Bp,
                              egemmtr_TR *restrict C, ptrdiff_t ldc,
                              ptrdiff_t row_base, ptrdiff_t col_base, char UPLO);

/* Unpacked stride-1 TN fast path (tiny N; skips the A/B pack). Bit-identical to
 * the blocked path. Honours the UPLO triangle per 2x2 tile. */
void egemmtr_unpacked_tn(char UPLO, ptrdiff_t n, ptrdiff_t k, egemmtr_TR alpha,
                         const egemmtr_TR *restrict a, ptrdiff_t lda,
                         const egemmtr_TR *restrict b, ptrdiff_t ldb,
                         egemmtr_TR *restrict c, ptrdiff_t ldc);

/* O(N²·K) scalar fallback (alloc failure path). */
void egemmtr_scalar_fallback(ptrdiff_t n, ptrdiff_t k, char UPLO, char ta, char tb,
                             egemmtr_TR alpha,
                             const egemmtr_TR *a, ptrdiff_t lda,
                             const egemmtr_TR *b, ptrdiff_t ldb,
                             egemmtr_TR *c, ptrdiff_t ldc);

/* Buffer-free fallback over a column band [j_lo, j_hi) — per-thread OOM path
 * for the column-threaded entry (disjoint columns, no double-compute). */
void egemmtr_scalar_fallback_cols(ptrdiff_t j_lo, ptrdiff_t j_hi,
                                  ptrdiff_t n, ptrdiff_t k, char UPLO, char ta, char tb,
                                  egemmtr_TR alpha,
                                  const egemmtr_TR *a, ptrdiff_t lda,
                                  const egemmtr_TR *b, ptrdiff_t ldb,
                                  egemmtr_TR *c, ptrdiff_t ldc);

/* Pure single-thread entry (by-value core). No OpenMP. */
void egemmtr_serial(char uplo, char transa, char transb,
                    ptrdiff_t n, ptrdiff_t k,
                    const egemmtr_TR *alpha_,
                    const egemmtr_TR *restrict a, ptrdiff_t lda,
                    const egemmtr_TR *restrict b, ptrdiff_t ldb,
                    const egemmtr_TR *beta_,
                    egemmtr_TR *restrict c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_EGEMMTR_KERNEL_H */
