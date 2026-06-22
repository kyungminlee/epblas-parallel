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
 *                     case that used to trip the libgomp barrier wedge);
 *                     otherwise fans these same kernel pieces across an
 *                     OpenMP team.
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
ptrdiff_t egemm_trans_code(char c);

ptrdiff_t egemm_round_up(ptrdiff_t v, ptrdiff_t m);

/* Cache-block sizes (env-overridable EBLAS_MC/KC/NC) with OpenBLAS-style
 * adaptive MC when K fits one panel. */
void egemm_choose_blocks(ptrdiff_t k, ptrdiff_t *MC, ptrdiff_t *KC, ptrdiff_t *NC);

/* C := beta*C pre-pass over the full M×N tile (handles K==0 / alpha==0). */
void egemm_beta_prepass(ptrdiff_t m, ptrdiff_t n, egemm_T beta, egemm_T *c, ptrdiff_t ldc);

/* Packers (panel-packed, OpenBLAS-style). */
void egemm_pack_A(const egemm_T *restrict A, ptrdiff_t lda,
                  ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb, char ta,
                  egemm_T *restrict Ap);
void egemm_pack_B(const egemm_T *restrict B, ptrdiff_t ldb,
                  ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb, char tb,
                  egemm_T *restrict Bp);

/* Drive one packed (ib,jb,pb) macro-tile via MR×NR sub-tiles. */
void egemm_macro_kernel(ptrdiff_t ib, ptrdiff_t jb, ptrdiff_t pb, egemm_T alpha,
                        const egemm_T *restrict Ap, const egemm_T *restrict Bp,
                        egemm_T *restrict C, ptrdiff_t ldc);

/* Fast path TA='T',TB='N': one C-column j2 (stride-1 dot, no packing). */
void egemm_fast_col(ptrdiff_t j2, ptrdiff_t m, ptrdiff_t k, egemm_T alpha,
                    const egemm_T *a, ptrdiff_t lda, const egemm_T *b, ptrdiff_t ldb,
                    egemm_T *c, ptrdiff_t ldc);

/*
 * Gate for the TN (ta='T', tb='N') no-pack fast_col path. fast_col runs the
 * stride-1 dot with a SINGLE fp80 accumulator, so the ~3-cyc x87 `fadd`
 * latency serializes the reduction (~5.1 cyc/MAC). The blocked packed path
 * keeps four independent MR×NR accumulator chains that hide that latency
 * (~2.9 cyc/MAC) — so it is faster per FLOP whenever its packing + buffer
 * alloc can be amortized. fast_col only wins where it can't: a SKINNY output
 * (min(M,N) tiny, so the packed element count ≈ the MAC count and packing
 * never pays off) or a TINY total FLOP (a 32³-ish cube the alloc dwarfs).
 *
 * The earlier `K <= 64` proxy was wrong: it mis-routed the moderate 64×64×64
 * cube — neither skinny nor tiny — to fast_col, where its serialized fadd
 * lost ~25% to the always-blocked ob reference (par1 243k vs ob1 191k ns).
 * Gating on min(M,N) and total work routes that cube to the blocked path.
 * Everything genuinely skinny/tiny still takes fast_col. */
static inline ptrdiff_t egemm_tn_use_fast(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k) {
    const ptrdiff_t mn = (m < n) ? m : n;   /* skinny (min) dimension */
    return mn <= 8 || (long)m * (long)n * (long)k <= 32L * 32L * 32L;
}

/* Pure single-thread GEMM (by-value core). Same math as egemm_ — no OpenMP. */
void egemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const egemm_T *alpha_,
    const egemm_T *a, ptrdiff_t lda,
    const egemm_T *b, ptrdiff_t ldb,
    const egemm_T *beta_,
    egemm_T *c, ptrdiff_t ldc);

#endif /* EPBLAS_PARALLEL_KIND10_EGEMM_KERNEL_H */
