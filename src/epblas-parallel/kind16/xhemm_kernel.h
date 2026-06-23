/*
 * xhemm_kernel.h — internal shared declarations for the kind16 complex
 * HEMM overlay (COMPLEX(KIND=16) / __complex128), split across two
 * translation units (mirrors xgemm/xsymm):
 *
 *   xhemm_serial.c   — the block plan, the B-panel packer, the per-M-slab
 *                      level3 worker (ICOPY(A) + microkernel) and the
 *                      pure-serial by-value entry `xhemm_serial`.
 *                      No `#pragma omp`.
 *   xhemm_parallel.c — the ptrdiff_t core `xhemm_core` plus the two
 *                      Fortran-ABI facades (LP64 `xhemm_` + ILP64
 *                      `xhemm_64_`) emitted by EPBLAS_FACADE_SYMM: M-axis
 *                      threading with an `omp_in_parallel()` nesting guard.
 *
 * Faithful port of the OpenBLAS GotoBLAS HEMM driver (ob clone
 * src/epblas-openblas/kind16/xhemm.c) over the shared packed substrate
 * (xl3_complex.c): C := alpha*A*B + beta*C (SIDE=L) or alpha*B*A + beta*C
 * (SIDE=R), A HERMITIAN (A == A^H). The reflected half is the complex
 * conjugate of the stored half and the diagonal imag is discarded; the
 * HEMM-aware copiers bake this in (the SIDE=R role uses the _oc variants
 * because (posX,posY) reinterprets as (col,row)). The regular factor goes
 * through the standard GEMM packer (conj=0). The 2×2 register microkernel
 * forms four independent accumulator chains that overlap libquadmath
 * soft-float call latency.
 *
 * `xhemm_serial` shares the ptrdiff_t by-value core ABI of xhemm_core;
 * a/b/c/alpha/beta are __complex128 (interleaved re,im), reached by
 * reinterpreting as __float128*.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XHEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XHEMM_KERNEL_H

#include <stddef.h>
#include <quadmath.h>

typedef __complex128 xhemm_TC;

/* Blocking plan for one xhemm call. side/uplo are uppercase Fortran codes;
 * K is the contraction dim (M for SIDE=L, N for SIDE=R). */
typedef struct {
    ptrdiff_t MC, KC, NC;
    size_t ap_bytes, bp_bytes;
    char side;   /* 'L' / 'R' */
    char uplo;   /* 'U' / 'L' */
    ptrdiff_t k;
} xhemm_plan_t;

void xhemm_make_plan(ptrdiff_t m, ptrdiff_t n, char side, char uplo, xhemm_plan_t *p);

/* OCOPY(B): pack the (pb × jb) panel at (ls, js) into Bp. For SIDE=L the
 * regular factor B goes through the standard GEMM ncopy; for SIDE=R the
 * Hermitian matrix is packed through the HEMM _oc copy. */
void xhemm_pack_B(const xhemm_plan_t *p,
                  const __float128 *B_eff, ptrdiff_t ldb_eff,
                  ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                  __float128 *Bp);

/* One (m_lo..m_hi) × (js..js+jb) slab: ICOPY(A) per M-block + microkernel
 * against an already-packed Bp. */
void xhemm_level3_slab(ptrdiff_t m_lo, ptrdiff_t m_hi, const xhemm_plan_t *p,
                       __float128 alphar, __float128 alphai,
                       const __float128 *A_eff, ptrdiff_t lda_eff, __float128 *Ap,
                       const __float128 *Bp,
                       ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                       __float128 *C, ptrdiff_t ldc);

/* Pure-serial by-value entry (no OpenMP). Shares the ptrdiff_t core ABI of
 * xhemm_core so callers already inside a parallel region can swap the symbol
 * name only. */
void xhemm_serial(
    char side, char uplo,
    ptrdiff_t m, ptrdiff_t n,
    const xhemm_TC *alpha_,
    const xhemm_TC *a, ptrdiff_t lda,
    const xhemm_TC *b, ptrdiff_t ldb,
    const xhemm_TC *beta_,
    xhemm_TC *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_XHEMM_KERNEL_H */
