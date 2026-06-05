/*
 * qgemm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 qgemm overlay is split across:
 *
 *   qgemm_serial.c    The pure single-thread GEMM (no OpenMP). Owns the
 *                     per-tile compute kernel, the trans-char decode, and
 *                     the public `qgemm_serial_` entry. Called directly by
 *                     the L3 routines that run qgemm trailing updates inside
 *                     their OWN parallel region, and by qgemm_ as its
 *                     serial branch.
 *
 *   qgemm_parallel.c  The public Fortran entry `qgemm_` — threading
 *                     orchestration only (2D tile grid, env-tunable tile
 *                     side). Delegates to the serial kernel when called from
 *                     inside a parallel region; otherwise fans the tile grid
 *                     across an OpenMP team.
 *
 * kind16 is arithmetic-bound (__float128 lowers to libquadmath calls), so
 * the overlay uses the unblocked reference algorithm with no packing — the
 * shared surface is just the trans decode and the per-tile compute kernel.
 */
#ifndef EPBLAS_PARALLEL_KIND16_QGEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_QGEMM_KERNEL_H

#include <stddef.h>

typedef __float128 qgemm_T;

/* Normalize a Fortran trans char to a code ('C' ≡ 'T' for real input). */
int qgemm_trans_code(const char *p, size_t len);

/* Compute one tile of C[i0:i1, j0:j1]:
 *
 *   C[i,j] = beta * C[i,j] + alpha * op(A) * op(B) [k summed]
 *
 * No OpenMP pragmas — pure sequential per-tile work. Each (i,j) is owned by
 * exactly one tile, so the beta pass is race-free under a tile partition. */
void qgemm_tile_compute(
    int i0, int i1, int j0, int j1, int K,
    int trans_a, int trans_b,
    qgemm_T alpha, qgemm_T beta,
    const qgemm_T *a, int lda,
    const qgemm_T *b, int ldb,
    qgemm_T *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Keeps
 * the exact Fortran-ABI signature of qgemm_ so callers already inside a
 * parallel region can swap the symbol name only. */
void qgemm_serial_(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const qgemm_T *alpha_,
    const qgemm_T *a, const int *lda_,
    const qgemm_T *b, const int *ldb_,
    const qgemm_T *beta_,
    qgemm_T *c, const int *ldc_,
    size_t transa_len, size_t transb_len);

#endif /* EPBLAS_PARALLEL_KIND16_QGEMM_KERNEL_H */
