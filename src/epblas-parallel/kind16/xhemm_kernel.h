/*
 * xhemm_kernel.h — internal shared declarations for the kind16 complex
 * HEMM overlay (COMPLEX(KIND=16) / __complex128), split across two
 * translation units (mirrors xgemm/xsymm):
 *
 *   xhemm_serial.c   — the block plan, the B-panel packer, the per-M-slab
 *                      level3 worker (ICOPY(A) + microkernel) and the
 *                      pure-serial Fortran-ABI entry `xhemm_serial_`.
 *                      No `#pragma omp`.
 *   xhemm_parallel.c — the public Fortran entry `xhemm_`: M-axis threading
 *                      with an `omp_in_parallel()` nesting guard.
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
 * `xhemm_serial_` keeps the exact int Fortran ABI of xhemm_; a/b/c/alpha/
 * beta are __complex128 (interleaved re,im), reached by reinterpreting as
 * __float128*.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XHEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XHEMM_KERNEL_H

#include <stddef.h>
#include <quadmath.h>

typedef __complex128 xhemm_T;

/* Blocking plan for one xhemm call. side/uplo are uppercase Fortran codes;
 * K is the contraction dim (M for SIDE=L, N for SIDE=R). */
typedef struct {
    int MC, KC, NC;
    size_t ap_bytes, bp_bytes;
    int side;   /* 'L' / 'R' */
    int uplo;   /* 'U' / 'L' */
    int K;
} xhemm_plan_t;

void xhemm_make_plan(int M, int N, int side, int uplo, xhemm_plan_t *p);

/* OCOPY(B): pack the (pb × jb) panel at (ls, js) into Bp. For SIDE=L the
 * regular factor B goes through the standard GEMM ncopy; for SIDE=R the
 * Hermitian matrix is packed through the HEMM _oc copy. */
void xhemm_pack_B(const xhemm_plan_t *p,
                  const __float128 *B_eff, int ldb_eff,
                  int js, int ls, int pb, int jb,
                  __float128 *Bp);

/* One (m_lo..m_hi) × (js..js+jb) slab: ICOPY(A) per M-block + microkernel
 * against an already-packed Bp. */
void xhemm_level3_slab(int m_lo, int m_hi, const xhemm_plan_t *p,
                       __float128 alphar, __float128 alphai,
                       const __float128 *A_eff, int lda_eff, __float128 *Ap,
                       const __float128 *Bp,
                       int js, int ls, int pb, int jb,
                       __float128 *C, int ldc);

/* Pure-serial Fortran-ABI entry (no OpenMP). Same int signature as xhemm_. */
void xhemm_serial_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const xhemm_T *alpha_,
    const xhemm_T *a, const int *lda_,
    const xhemm_T *b, const int *ldb_,
    const xhemm_T *beta_,
    xhemm_T *c, const int *ldc_,
    size_t side_len, size_t uplo_len);

#endif /* EPBLAS_PARALLEL_KIND16_XHEMM_KERNEL_H */
