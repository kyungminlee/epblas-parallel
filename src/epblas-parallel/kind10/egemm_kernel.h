/*
 * egemm_kernel.h — internal kernel surface shared by the two translation
 * units the kind10 egemm overlay is split across:
 *
 *   egemm_serial.c    The pure single-thread GEMM (no OpenMP). Owns ALL
 *                     the math — packers, MR×NR micro-kernel, beta
 *                     pre-pass, block-size policy — and the public
 *                     `egemm_serial` entry. Called directly by the L3
 *                     routines (etrsm, etrmm, …) that run egemm trailing
 *                     updates inside their OWN parallel region, and by
 *                     egemm_ as its serial branch.
 *
 *   egemm_parallel.c  The public Fortran entry `egemm_` — threading
 *                     orchestration only. Delegates to egemm_serial when
 *                     called from inside a parallel region (the nested
 *                     case that used to trip the libgomp barrier wedge,
 *                     see memory project-etrsm-omp4-wedge); otherwise
 *                     fans these same kernel pieces across an OpenMP team.
 *
 * Everything here is internal to the overlay. `egemm_serial` keeps the
 * exact Fortran-ABI signature of egemm_ so callers already inside a
 * parallel region can swap the symbol name only.
 */
#ifndef EPBLAS_PARALLEL_KIND10_EGEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_EGEMM_KERNEL_H

#include <stddef.h>

typedef long double egemm_T;

/* Register-tile dims (compile-time constants; deliberately small so the
 * four fp80 accumulators fit the 8-deep x87 register stack). */
#define EGEMM_MR 2
#define EGEMM_NR 2

/* Normalize a Fortran trans char to a code ('C' ≡ 'T' for real input). */
int egemm_trans_code(const char *p, size_t len);

int egemm_round_up(int v, int m);

/* Cache-block sizes (env-overridable EBLAS_MC/KC/NC) with OpenBLAS-style
 * adaptive MC when K fits one panel. */
void egemm_choose_blocks(int K, int *MC, int *KC, int *NC);

/* C := beta*C pre-pass over the full M×N tile (handles K==0 / alpha==0). */
void egemm_beta_prepass(int M, int N, egemm_T beta, egemm_T *c, int ldc);

/* Packers (panel-packed, OpenBLAS-style). */
void egemm_pack_A(const egemm_T *restrict A, int lda,
                  int ic, int pc, int ib, int pb, int ta,
                  egemm_T *restrict Ap);
void egemm_pack_B(const egemm_T *restrict B, int ldb,
                  int pc, int jc, int pb, int jb, int tb,
                  egemm_T *restrict Bp);

/* Drive one packed (ib,jb,pb) macro-tile via MR×NR sub-tiles. */
void egemm_macro_kernel(int ib, int jb, int pb, egemm_T alpha,
                        const egemm_T *restrict Ap, const egemm_T *restrict Bp,
                        egemm_T *restrict C, int ldc);

/* Fast path TA='T',TB='N': one C-column j2 (stride-1 dot, no packing). */
void egemm_fast_col(int j2, int M, int K, egemm_T alpha,
                    const egemm_T *a, int lda, const egemm_T *b, int ldb,
                    egemm_T *c, int ldc);

/* Pure single-thread GEMM. Same signature as egemm_ — no OpenMP. */
void egemm_serial(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const egemm_T *alpha_,
    const egemm_T *a, const int *lda_,
    const egemm_T *b, const int *ldb_,
    const egemm_T *beta_,
    egemm_T *c, const int *ldc_,
    size_t transa_len, size_t transb_len);

#endif /* EPBLAS_PARALLEL_KIND10_EGEMM_KERNEL_H */
