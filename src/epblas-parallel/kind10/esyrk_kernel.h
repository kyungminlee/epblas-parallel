/*
 * esyrk_kernel.h — internal kernel surface shared by the two translation
 * units the kind10 esyrk overlay is split across:
 *
 *   esyrk_serial.c    The pure single-thread GotoBLAS rank-k (no OpenMP).
 *                     Owns ALL the leaf math — packers, MR×NR micro-kernels,
 *                     rectangular + triangle-aware macro kernels, block
 *                     policy — plus the inline serial driver
 *                     (esyrk_serial_inline) and the public Fortran-ABI
 *                     serial entry `esyrk_serial`.
 *
 *   esyrk_parallel.c  The public Fortran entry `esyrk_` — threading only.
 *                     Owns the cooperative OpenBLAS-style path (quadratic
 *                     N-partition, cross-thread buffer-sharing flags,
 *                     inner_syrk), reusing the leaf kernels from the header.
 *                     Delegates to esyrk_serial when called from inside a
 *                     parallel region (the libgomp barrier wedge guard,
 *                     project-etrsm-omp4-wedge) and as its OOM fallback.
 */
#ifndef EPBLAS_PARALLEL_KIND10_ESYRK_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ESYRK_KERNEL_H

#include <stddef.h>

typedef long double esyrk_T;

#define ESYRK_MR 2
#define ESYRK_NR 2

int esyrk_round_up(int v, int m);

/* Cache-block sizes (env EBLAS-style overrides) and the cooperative switch
 * ratio; both lazy-initialized. */
void esyrk_block_sizes(int *MC, int *KC, int *NC);
int esyrk_switch_ratio(void);

/* Packers (op(A) row panels / column panels) — see esyrk_serial.c. */
void esyrk_pack_A_panel(const esyrk_T *restrict A, int lda,
                        int i0, int pc, int min_i, int min_l,
                        int TR, esyrk_T *restrict Apack);
void esyrk_pack_B_panel(const esyrk_T *restrict A, int lda,
                        int j0, int pc, int min_j, int min_l,
                        int TR, esyrk_T *restrict Bpack);

/* Macro-kernels over a packed (ib,jb,pb) tile. */
void esyrk_macro_kernel_rect(int ib, int jb, int pb, esyrk_T alpha,
                             const esyrk_T *restrict Ap,
                             const esyrk_T *restrict Bp,
                             esyrk_T *restrict C, int ldc);
void esyrk_macro_kernel_tri(int ib, int jb, int pb, esyrk_T alpha,
                            const esyrk_T *restrict Ap,
                            const esyrk_T *restrict Bp,
                            esyrk_T *restrict C, int ldc,
                            int row_base, int col_base, char UPLO);

/* Inline single-thread GotoBLAS driver (beta-scale + tile-classified loop
 * nest). Takes the already-normalized UPLO/TR and explicit beta. */
void esyrk_serial_inline(char UPLO, char TR, int N, int K,
                         esyrk_T alpha, const esyrk_T *restrict a, int lda,
                         esyrk_T beta, esyrk_T *restrict c, int ldc);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same signature as esyrk_. */
void esyrk_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const esyrk_T *alpha_,
    const esyrk_T *a, const int *lda_,
    const esyrk_T *beta_,
    esyrk_T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len);

#endif /* EPBLAS_PARALLEL_KIND10_ESYRK_KERNEL_H */
