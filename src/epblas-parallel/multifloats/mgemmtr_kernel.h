/*
 * mgemmtr_kernel.h — internal kernel surface shared by the two translation
 * units the multifloats (double-double / float64x2) mgemmtr overlay is split
 * across:
 *
 *   mgemmtr_serial.cpp    The pure single-thread triangular GEMM update (no
 *                         OpenMP). Owns the block-size policy, the scalar
 *                         diagonal-triangle update, and the two per-block
 *                         cores (beta-only + full block), plus the public
 *                         `mgemmtr_serial` entry. The full block routes its
 *                         off-diagonal rectangle through `mgemm_serial`
 *                         (declared in mgemm_kernel.h) — no nested OpenMP.
 *
 *   mgemmtr_parallel.cpp  The public Fortran entry `mgemmtr_` — threading
 *                         orchestration only. Fans the jc-block loop across
 *                         an OpenMP team (each block owns a disjoint column
 *                         range, so the partition is race-free and bitwise-
 *                         identical to the serial sweep); delegates to
 *                         mgemmtr_serial when called from inside a parallel
 *                         region or below the threading threshold.
 *
 * Leaf names are mgemmtr_-prefixed so they keep external linkage without
 * colliding with the other routines' helpers in the same archive.
 */
#ifndef EPBLAS_PARALLEL_MULTIFLOATS_MGEMMTR_KERNEL_H
#define EPBLAS_PARALLEL_MULTIFLOATS_MGEMMTR_KERNEL_H

#include <cstddef>
#include <multifloats.h>

using mgemmtr_T = multifloats::float64x2;

/* Block size over the column axis (env-overridable MGEMMTR_NB). */
int mgemmtr_block_nb(void);

/* Threading threshold (N below this stays serial). */
#define MGEMMTR_OMP_MIN 32

/* Beta-only pass over columns [j0,j1): scales (or zeros) the UPLO triangle of
 * each column by beta. The body of the alpha==0 / K==0 early-exit loop. */
void mgemmtr_beta_core(int j0, int j1, int N, bool upper,
                       mgemmtr_T beta, mgemmtr_T *c, int ldc);

/* One jc-block of the full update: beta-scale the triangle slice for columns
 * [jc, jc+jb), add the scalar jb×jb diagonal triangle, then route the off-
 * diagonal rectangle through mgemm_serial (the SIMD kernel). Each jc-block is
 * column-disjoint → race-free. ta/tb are pre-normalized ('C'→'T'). */
void mgemmtr_block_core(int jc, int jb, int N, int K,
                        mgemmtr_T alpha, mgemmtr_T beta,
                        const mgemmtr_T *a, int lda,
                        const mgemmtr_T *b, int ldb,
                        mgemmtr_T *c, int ldc,
                        bool upper, char ta, char tb);

/* Pure-serial Fortran entry. No OpenMP on this path; same ABI as mgemmtr_. */
extern "C" void mgemmtr_serial(
    const char *uplo, const char *transa, const char *transb,
    const int *n_, const int *k_,
    const mgemmtr_T *alpha_,
    const mgemmtr_T *a, const int *lda_,
    const mgemmtr_T *b, const int *ldb_,
    const mgemmtr_T *beta_,
    mgemmtr_T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t ta_len, std::size_t tb_len);

#endif /* EPBLAS_PARALLEL_MULTIFLOATS_MGEMMTR_KERNEL_H */
