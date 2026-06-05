/*
 * xgemm_kernel.h — internal kernel surface shared by the two translation
 * units the kind16 complex xgemm overlay is split across:
 *
 *   xgemm_serial.c    The pure single-thread complex GEMM (no OpenMP). Owns
 *                     the per-tile compute kernel, the trans-char decode, and
 *                     the public `xgemm_serial_` entry. Called directly by
 *                     the L3 routines that run xgemm trailing updates inside
 *                     their OWN parallel region, and by xgemm_ as its
 *                     serial branch.
 *
 *   xgemm_parallel.c  The public Fortran entry `xgemm_` — threading
 *                     orchestration only (2D tile grid, env-tunable tile
 *                     side). Delegates to the serial kernel when called from
 *                     inside a parallel region; otherwise fans the tile grid
 *                     across an OpenMP team.
 *
 * kind16 complex is arithmetic-bound (__complex128 lowers to libquadmath
 * calls), so the overlay uses the unblocked reference algorithm with no
 * packing — the shared surface is just the trans decode and the per-tile
 * compute kernel. TRANSA / TRANSB independently in {N, T, C}; conjugation
 * under 'C' is applied at element access time.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XGEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XGEMM_KERNEL_H

#include <stddef.h>
#include <quadmath.h>   /* __complex128 */

typedef __complex128 xgemm_T;

/* Normalize a Fortran trans char to its uppercase code ('N'/'T'/'C'). */
int xgemm_trans_code(const char *p, size_t len);

/* Compute one tile of C[i0:i1, j0:j1]:
 *
 *   C[i,j] = beta * C[i,j] + alpha * op(A) * op(B) [k summed]
 *
 * No OpenMP pragmas — pure sequential per-tile work. Each (i,j) is owned by
 * exactly one tile, so the beta pass is race-free under a tile partition.
 * Conjugation under 'C' is applied at element access time. */
void xgemm_tile_compute(
    int i0, int i1, int j0, int j1, int K,
    int trans_a, int conj_a, int trans_b, int conj_b,
    xgemm_T alpha, xgemm_T beta,
    const xgemm_T *a, int lda,
    const xgemm_T *b, int ldb,
    xgemm_T *c, int ldc);

/* Pure-serial Fortran entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region. Keeps
 * the exact Fortran-ABI signature of xgemm_ so callers already inside a
 * parallel region can swap the symbol name only. */
void xgemm_serial_(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const xgemm_T *alpha_,
    const xgemm_T *a, const int *lda_,
    const xgemm_T *b, const int *ldb_,
    const xgemm_T *beta_,
    xgemm_T *c, const int *ldc_,
    size_t transa_len, size_t transb_len);

#endif /* EPBLAS_PARALLEL_KIND16_XGEMM_KERNEL_H */
