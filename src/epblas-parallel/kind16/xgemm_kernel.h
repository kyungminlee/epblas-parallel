/*
 * xgemm_kernel.h — internal shared declarations for the kind16 complex
 * GEMM overlay (COMPLEX(KIND=16) / __complex128), split across two
 * translation units:
 *
 *   xgemm_serial.c   — all the math: the block plan, the per-M-slab level3
 *                      worker (ICOPY(A) + microkernel), the B-panel packer,
 *                      and the pure-serial Fortran-ABI entry `xgemm_serial_`.
 *                      No `#pragma omp`. Called directly by the complex L3
 *                      routines (xtrsm, xtrmm, xsyrk, … ) that run xgemm
 *                      trailing updates inside their OWN parallel region.
 *   xgemm_parallel.c — the public Fortran entry `xgemm_`: threading
 *                      orchestration only (M-axis partition with per-thread
 *                      Ap and shared Bp), with an `omp_in_parallel()` guard
 *                      that delegates to `xgemm_serial_` when called from
 *                      inside another routine's parallel region.
 *
 * Both drivers run the OpenBLAS GotoBLAS blocking nest over the shared
 * packed substrate (xl3_complex.c): qblas_ygemm_beta / _blocks / _ncopy /
 * _tcopy / _kernel. __complex128 has no SIMD path — every multiply lowers to
 * a libquadmath soft-float call — but the 2×2 register microkernel forms
 * four independent accumulator chains that overlap that call latency, which
 * the plain reference rank-1 loop cannot. This is a faithful port of the ob
 * clone src/epblas-openblas/kind16/xgemm.c.
 *
 * `xgemm_serial_` keeps the exact int Fortran-ABI signature of xgemm_ so
 * callers already inside a parallel region (e.g. xtrsm) swap the symbol name
 * only. The a/b/c/alpha/beta pointers are __complex128 (interleaved re,im);
 * the substrate is reached by reinterpreting them as __float128*.
 */
#ifndef EPBLAS_PARALLEL_KIND16_XGEMM_KERNEL_H
#define EPBLAS_PARALLEL_KIND16_XGEMM_KERNEL_H

#include <stddef.h>
#include <quadmath.h>

typedef __complex128 xgemm_TC;

/* Normalize a Fortran trans char to its uppercase code ('N'/'T'/'C'/'R'). */
char xgemm_trans_code(char c);

/* Blocking plan for one xgemm call. MC may be grown adaptively for small K.
 * The conj/trans flags are decoded once from the (ta, tb) codes. */
typedef struct {
    ptrdiff_t MC, KC, NC;
    size_t ap_bytes, bp_bytes;
    bool conj_a, trans_a;
    bool conj_b, trans_b;
} xgemm_plan_t;

/* Fill *p from the problem dims and the (ta, tb) trans codes. */
void xgemm_make_plan(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, char ta, char tb, xgemm_plan_t *p);

/* OCOPY(B): pack the (pb × jb) panel at (ls, js) into Bp. */
void xgemm_pack_B(const xgemm_plan_t *p,
                  const __float128 *b, ptrdiff_t ldb,
                  ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                  __float128 *Bp);

/* One (m_lo..m_hi) × (js..js+jb) slab: ICOPY(A) per M-block + microkernel
 * against an already-packed Bp. */
void xgemm_level3_slab(ptrdiff_t m_lo, ptrdiff_t m_hi, const xgemm_plan_t *p,
                       __float128 alphar, __float128 alphai,
                       const __float128 *A, ptrdiff_t lda, __float128 *Ap,
                       const __float128 *Bp,
                       ptrdiff_t js, ptrdiff_t ls, ptrdiff_t pb, ptrdiff_t jb,
                       __float128 *C, ptrdiff_t ldc);

/* Pure-serial by-value entry (no OpenMP). Shares the ptrdiff_t core ABI of
 * xgemm_core so callers already inside a parallel region (e.g. xtrsm) can swap
 * the symbol name only. */
void xgemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const xgemm_TC *alpha_,
    const xgemm_TC *a, ptrdiff_t lda,
    const xgemm_TC *b, ptrdiff_t ldb,
    const xgemm_TC *beta_,
    xgemm_TC *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND16_XGEMM_KERNEL_H */
