/*
 * xsymm_kernel.h — internal shared declarations for the kind16 complex
 * SYMM overlay (COMPLEX(KIND=16) / __complex128), split across two
 * translation units (mirrors xgemm):
 *
 *   xsymm_serial.c   — the block plan, the B-panel packer, the per-M-slab
 *                      level3 worker (ICOPY(A) + microkernel) and the
 *                      pure-serial by-value entry `xsymm_serial`.
 *                      No `#pragma omp`.
 *   xsymm_parallel.c — the ptrdiff_t core `xsymm_core` plus the two
 *                      Fortran-ABI facades (LP64 `xsymm_` + ILP64
 *                      `xsymm_64_`) emitted by EPBLAS_FACADE_SYMM: M-axis
 *                      threading with an `omp_in_parallel()` nesting guard.
 *
 * Faithful port of the OpenBLAS GotoBLAS SYMM driver (ob clone
 * src/epblas-openblas/kind16/xsymm.c) over the shared packed substrate
 * (xl3_complex.c): C := alpha*A*B + beta*C (SIDE=L) or alpha*B*A + beta*C
 * (SIDE=R), A complex-SYMMETRIC (A == A^T, no conjugation). The symmetric
 * factor is packed via the SYMM-aware copiers; the regular factor goes
 * through the standard GEMM packer. The 2×2 register microkernel forms
 * four independent accumulator chains that overlap libquadmath soft-float
 * call latency, which the blocked-trailing rank loop cannot.
 *
 * `xsymm_serial` shares the ptrdiff_t by-value core ABI of xsymm_core;
 * a/b/c/alpha/beta are __complex128 (interleaved re,im), reached by
 * reinterpreting as __float128*.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XSYMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XSYMM_KERNEL_H

#include <stddef.h>
#include <quadmath.h>

typedef __complex128 xsymm_T;

/* Blocking plan for one xsymm call. side/uplo are uppercase Fortran codes;
 * K is the contraction dim (M for SIDE=L, N for SIDE=R). */
typedef struct {
    ptrdiff_t MC, KC, NC;
    size_t ap_bytes, bp_bytes;
    int side;   /* 'L' / 'R' */
    int uplo;   /* 'U' / 'L' */
    ptrdiff_t K;
} xsymm_plan_t;

void xsymm_make_plan(ptrdiff_t M, ptrdiff_t N, int side, int uplo, xsymm_plan_t *p);

/* OCOPY(B): pack the (pb × jb) panel at (ls, js) into Bp. For SIDE=L the
 * regular factor B goes through the standard GEMM ncopy; for SIDE=R the
 * symmetric matrix is packed through the SYMM copy. */
void xsymm_pack_B(const xsymm_plan_t *p,
                  const __float128 *B_eff, ptrdiff_t ldb_eff,
                  ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                  __float128 *Bp);

/* One (m_lo..m_hi) × (js..js+jb) slab: ICOPY(A) per M-block + microkernel
 * against an already-packed Bp. */
void xsymm_level3_slab(ptrdiff_t m_lo, ptrdiff_t m_hi, const xsymm_plan_t *p,
                       __float128 alphar, __float128 alphai,
                       const __float128 *A_eff, ptrdiff_t lda_eff, __float128 *Ap,
                       const __float128 *Bp,
                       ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                       __float128 *C, ptrdiff_t ldc);

/* Pure-serial by-value entry (no OpenMP). Shares the ptrdiff_t core ABI of
 * xsymm_core so callers already inside a parallel region can swap the symbol
 * name only. */
void xsymm_serial(
    char side, char uplo,
    ptrdiff_t M, ptrdiff_t N,
    const xsymm_T *alpha_,
    const xsymm_T *a, ptrdiff_t lda,
    const xsymm_T *b, ptrdiff_t ldb,
    const xsymm_T *beta_,
    xsymm_T *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_XSYMM_KERNEL_H */
