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
int egemmtr_trans_code(const char *p);

int egemmtr_round_up(int v, int m);

/* Cache-block sizes (env-overridable EGEMMTR_MC/KC/NC); lazy-initialized. */
void egemmtr_block_sizes(int *MC, int *KC, int *NC);

/* C := beta*C over the UPLO triangle of columns [j_start, j_end). */
void egemmtr_beta_scale(int j_start, int j_end, int N, char UPLO,
                        egemmtr_T beta, egemmtr_T *c, int ldc);

/* Packers (egemm-style, take separate A and B). */
void egemmtr_pack_A(const egemmtr_T *restrict A, int lda,
                    int ic, int pc, int ib, int pb, int ta,
                    egemmtr_T *restrict Ap);
void egemmtr_pack_B(const egemmtr_T *restrict B, int ldb,
                    int pc, int jc, int pb, int jb, int tb,
                    egemmtr_T *restrict Bp);

/* Rectangular macro-tile (entirely inside the stored triangle). */
void egemmtr_macro_kernel_rect(int ib, int jb, int pb, egemmtr_T alpha,
                               const egemmtr_T *restrict Ap,
                               const egemmtr_T *restrict Bp,
                               egemmtr_T *restrict C, int ldc);

/* Triangle-aware macro-tile (crosses the diagonal). */
void egemmtr_macro_kernel_tri(int ib, int jb, int pb, egemmtr_T alpha,
                              const egemmtr_T *restrict Ap,
                              const egemmtr_T *restrict Bp,
                              egemmtr_T *restrict C, int ldc,
                              int row_base, int col_base, char UPLO);

/* O(N²·K) scalar fallback (alloc failure path). */
void egemmtr_scalar_fallback(int N, int K, char UPLO, int ta, int tb,
                             egemmtr_T alpha,
                             const egemmtr_T *a, int lda,
                             const egemmtr_T *b, int ldb,
                             egemmtr_T *c, int ldc);

/* Pure single-thread entry. Same signature as egemmtr_ — no OpenMP. */
void egemmtr_serial(const char *uplo, const char *transa, const char *transb,
                    const int *n_, const int *k_,
                    const egemmtr_T *alpha_,
                    const egemmtr_T *a, const int *lda_,
                    const egemmtr_T *b, const int *ldb_,
                    const egemmtr_T *beta_,
                    egemmtr_T *c, const int *ldc_,
                    size_t uplo_len, size_t ta_len, size_t tb_len);

#endif /* EPBLAS_PARALLEL_KIND10_EGEMMTR_KERNEL_H */
