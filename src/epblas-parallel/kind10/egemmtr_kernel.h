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
 *   egemmtr_parallel.c  The public Fortran entry `egemmtr_` — threading
 *                       orchestration only. Delegates to egemmtr_serial when
 *                       called from inside a parallel region (the libgomp
 *                       barrier wedge guard, project-etrsm-omp4-wedge);
 *                       otherwise fans these same kernel pieces across an
 *                       OpenMP team (shared Bp under `omp single`, `omp for`
 *                       over the ic loop with the triangular load balanced).
 */
#ifndef EPBLAS_PARALLEL_KIND10_EGEMMTR_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_EGEMMTR_KERNEL_H

#include <stddef.h>

typedef long double egemmtr_T;

/* Register-tile dims. */
#define EGEMMTR_MR 2
#define EGEMMTR_NR 2

/* Normalize a Fortran trans char to a code ('C' ≡ 'T' for real input). */
ptrdiff_t egemmtr_trans_code(const char *p);

ptrdiff_t egemmtr_round_up(ptrdiff_t v, ptrdiff_t m);

/* Cache-block sizes (env-overridable EGEMMTR_MC/KC/NC); lazy-initialized. */
void egemmtr_block_sizes(ptrdiff_t *MC, ptrdiff_t *KC, ptrdiff_t *NC);

/* C := beta*C over the UPLO triangle of columns [j_start, j_end). */
void egemmtr_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, char UPLO,
                        egemmtr_T beta, egemmtr_T *c, ptrdiff_t ldc);

/* Packers (egemm-style, take separate A and B). */
void egemmtr_pack_A(const egemmtr_T *restrict A, ptrdiff_t lda,
                    ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb, char ta,
                    egemmtr_T *restrict Ap);
void egemmtr_pack_B(const egemmtr_T *restrict B, ptrdiff_t ldb,
                    ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb, char tb,
                    egemmtr_T *restrict Bp);

/* Rectangular macro-tile (entirely inside the stored triangle). */
void egemmtr_macro_kernel_rect(ptrdiff_t ib, ptrdiff_t jb, ptrdiff_t pb, egemmtr_T alpha,
                               const egemmtr_T *restrict Ap,
                               const egemmtr_T *restrict Bp,
                               egemmtr_T *restrict C, ptrdiff_t ldc);

/* Triangle-aware macro-tile (crosses the diagonal). */
void egemmtr_macro_kernel_tri(ptrdiff_t ib, ptrdiff_t jb, ptrdiff_t pb, egemmtr_T alpha,
                              const egemmtr_T *restrict Ap,
                              const egemmtr_T *restrict Bp,
                              egemmtr_T *restrict C, ptrdiff_t ldc,
                              ptrdiff_t row_base, ptrdiff_t col_base, char UPLO);

/* O(N²·K) scalar fallback (alloc failure path). */
void egemmtr_scalar_fallback(ptrdiff_t n, ptrdiff_t k, char UPLO, char ta, char tb,
                             egemmtr_T alpha,
                             const egemmtr_T *a, ptrdiff_t lda,
                             const egemmtr_T *b, ptrdiff_t ldb,
                             egemmtr_T *c, ptrdiff_t ldc);

/* Pure single-thread entry (by-value core). No OpenMP. */
void egemmtr_serial(char uplo, char transa, char transb,
                    ptrdiff_t n, ptrdiff_t k,
                    const egemmtr_T *alpha_,
                    const egemmtr_T *restrict a, ptrdiff_t lda,
                    const egemmtr_T *restrict b, ptrdiff_t ldb,
                    const egemmtr_T *beta_,
                    egemmtr_T *restrict c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_EGEMMTR_KERNEL_H */
