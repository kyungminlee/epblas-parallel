/*
 * etri_kernel.h — the ob-convention L3 GEMM substrate shared by the kind10
 * (REAL(KIND=10) / 80-bit long double) triangular routines (etrsm, etrmm).
 *
 * This is a private MR=NR=2 GEMM micro-kernel plus its matching ncopy/tcopy
 * packers and a zero-then-accumulate "store" variant — faithful ports of
 * OpenBLAS kernel/generic/{gemmkernel_2x2,gemm_ncopy_2,gemm_tcopy_2}.c (the
 * same code the openblas overlay carries in eblas_l3_real.c).
 *
 * Why a PRIVATE substrate rather than par's egemm primitives: a triangular
 * routine's solve/trmm micro-kernel and its trailing GEMM share one packed
 * diagonal-block buffer at MR/NR granularity, and OpenBLAS's
 * contiguous-odd-tail packing convention is baked into the packer ↔
 * solve/trmm ↔ kernel triad. par's egemm packs odd tails zero-padded at
 * stride MR instead, so its kernel reads those bytes differently (proven
 * mismatch on every odd m/n/k). This substrate is therefore self-consistent
 * with the triangular packers (etrsm_pack.c, etrmm_pack.c). The
 * layout-AGNOSTIC block-size policy is still shared with egemm
 * (egemm_choose_blocks / egemm_beta_prepass / blas_round_up).
 *
 * Both etrsm and etrmm depend on this substrate: etrsm pairs it with the
 * diagonal solve kernel (etrsm_kernel.c), etrmm with the TRMM kernel
 * (etrmm_kernel.c).
 */
#ifndef EPBLAS_PARALLEL_KIND10_ETRI_KERNEL_H
#define EPBLAS_PARALLEL_KIND10_ETRI_KERNEL_H

#include <stddef.h>
#include <stdbool.h>

typedef long double etri_TR;

/* C += alpha·Ap·Bp over one packed (bm,bn,bk) tile (accumulate). */
void etri_gemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk, etri_TR alpha,
                      const etri_TR *Ap, const etri_TR *Bp,
                      etri_TR *C, ptrdiff_t ldc);

/* C -= Ap·Bp over one packed tile — the alpha = -1 trailing update the
 * triangular solve/trmm drivers issue. Negate-specialized: the per-element
 * store is a bare subtract (no alpha multiply), matching OpenBLAS's
 * constant-propagated kernel. Use this instead of etri_gemm_kernel(...,-1,...)
 * on the hot trailing-GEMM path. */
void etri_gemm_kernel_msub(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                           const etri_TR *Ap, const etri_TR *Bp,
                           etri_TR *C, ptrdiff_t ldc);

/* C := alpha·Ap·Bp over one packed tile (overwrite): zero C then accumulate.
 * Mirrors OpenBLAS's GEMM_KERNEL(beta=0) call inside the TRMM driver. */
void etri_kernel_store(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk, etri_TR alpha,
                       const etri_TR *Ap, const etri_TR *Bp,
                       etri_TR *C, ptrdiff_t ldc);

/* Shared body for the general (C += alpha·Ap·Bp) and negate-specialized
 * (C -= Ap·Bp) kernels. `negate` is a compile-time constant at each
 * instantiation, so GCC folds the store: the negate=1 clone drops the
 * per-element `alpha *` multiply (and its alpha reload) to a bare `fsubp`.
 * Lives in the header so a caller TU can instantiate its OWN static copy:
 * the trsm solve kernel issues thousands of tiny (2,2,kk) trailing updates
 * per call, and a cross-TU call denies GCC the IPA-RA and bm/bn constprop
 * it applies when caller and kernel share a TU (which is what the openblas
 * overlay gets from having its solve and GEMM in one eblas_l3_real.c —
 * its interior calls compile to a constprop clone with no caller-side
 * spill/reload). Extern users keep the etri_kernel.c instantiations. */
static inline __attribute__((always_inline)) void
etri_gemm_body(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
               etri_TR alpha, bool negate,
               const etri_TR *Ap, const etri_TR *Bp,
               etri_TR *C, ptrdiff_t ldc)
{
    /* Register-tile dims MR = NR = 2 (must match the packed layout). */
    /* Walk B in NR=2-col panels. */
    const etri_TR *ptrba_base = Ap;
    const etri_TR *ptrbb = Bp;
    etri_TR *Cj = C;

    for (ptrdiff_t j = 0; j < bn / 2; ++j) {
        etri_TR *C0 = Cj;
        etri_TR *C1 = C0 + ldc;
        const etri_TR *ptrba = ptrba_base;

        /* MR=2 row panels (full 2x2 tiles). */
        for (ptrdiff_t i = 0; i < bm / 2; ++i) {
            const etri_TR *ptrbb_loc = ptrbb;
            etri_TR r0 = 0, r1 = 0, r2 = 0, r3 = 0;

            /* K-loop unrolled by 4 (same shape as OpenBLAS gen kernel). */
            ptrdiff_t k4 = bk / 4;
            for (ptrdiff_t k = 0; k < k4; ++k) {
                etri_TR a0 = ptrba[0], a1 = ptrba[1];
                etri_TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                a0 = ptrba[2]; a1 = ptrba[3];
                b0 = ptrbb_loc[2]; b1 = ptrbb_loc[3];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                a0 = ptrba[4]; a1 = ptrba[5];
                b0 = ptrbb_loc[4]; b1 = ptrbb_loc[5];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                a0 = ptrba[6]; a1 = ptrba[7];
                b0 = ptrbb_loc[6]; b1 = ptrbb_loc[7];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                ptrba += 8;
                ptrbb_loc += 8;
            }
            for (ptrdiff_t k = 0; k < (bk & 3); ++k) {
                etri_TR a0 = ptrba[0], a1 = ptrba[1];
                etri_TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                ptrba += 2;
                ptrbb_loc += 2;
            }

            if (negate) {
                C0[0] -= r0; C0[1] -= r1; C1[0] -= r2; C1[1] -= r3;
            } else {
                C0[0] += alpha * r0; C0[1] += alpha * r1;
                C1[0] += alpha * r2; C1[1] += alpha * r3;
            }
            C0 += 2;
            C1 += 2;
        }

        /* bm & 1 — single-row tail (mr=1). */
        for (ptrdiff_t i = 0; i < (bm & 1); ++i) {
            const etri_TR *ptrbb_loc = ptrbb;
            etri_TR r0 = 0, r1 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                etri_TR a0 = ptrba[0];
                etri_TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                r0 += a0 * b0;
                r1 += a0 * b1;
                ptrba += 1;
                ptrbb_loc += 2;
            }
            if (negate) { C0[0] -= r0; C1[0] -= r1; }
            else        { C0[0] += alpha * r0; C1[0] += alpha * r1; }
            C0 += 1;
            C1 += 1;
        }

        ptrbb += bk * 2;       /* advance to next 2-col B panel */
        Cj += 2 * ldc;
    }

    /* bn & 1 — single-col tail (nr=1). */
    for (ptrdiff_t j = 0; j < (bn & 1); ++j) {
        etri_TR *C0 = Cj;
        const etri_TR *ptrba = ptrba_base;

        for (ptrdiff_t i = 0; i < bm / 2; ++i) {
            const etri_TR *ptrbb_loc = ptrbb;
            etri_TR r0 = 0, r1 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                etri_TR a0 = ptrba[0], a1 = ptrba[1];
                etri_TR b0 = ptrbb_loc[0];
                r0 += a0 * b0;
                r1 += a1 * b0;
                ptrba += 2;
                ptrbb_loc += 1;
            }
            if (negate) { C0[0] -= r0; C0[1] -= r1; }
            else        { C0[0] += alpha * r0; C0[1] += alpha * r1; }
            C0 += 2;
        }
        for (ptrdiff_t i = 0; i < (bm & 1); ++i) {
            const etri_TR *ptrbb_loc = ptrbb;
            etri_TR r0 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                r0 += ptrba[0] * ptrbb_loc[0];
                ptrba += 1;
                ptrbb_loc += 1;
            }
            if (negate) C0[0] -= r0;
            else        C0[0] += alpha * r0;
            C0 += 1;
        }
        ptrbb += bk;           /* advance over single-col B panel */
        Cj += ldc;
    }
}

/* Pack a plain (non-triangular) A/B slab into the packed layout. */
void etri_ncopy(ptrdiff_t m, ptrdiff_t n, const etri_TR *a, ptrdiff_t lda,
                etri_TR *b);
void etri_tcopy(ptrdiff_t m, ptrdiff_t n, const etri_TR *a, ptrdiff_t lda,
                etri_TR *b);

/* Pack-scratch overrun guards. The trsm/trmm drivers place Ap and Bp in one
 * allocation (or in adjacent heap blocks), so a pack that overruns its
 * segment corrupts the neighbouring data SILENTLY — wrong results with no
 * fault, which square-shape benches and fuzz cannot catch (the 9ff020a
 * R-side mc_eff bug shipped exactly this way). Each segment therefore ends
 * in a poisoned region (its size slack plus one dedicated ETRI_PACK_GUARD
 * line) written before the block nest runs and verified after it returns:
 * pack writes are contiguous streams, so any overrun crosses the poison and
 * aborts loudly. Always on — the cost is a ~128-byte write + readback per
 * top-level call, nothing per panel (and NDEBUG would compile assert() out
 * of the Release library that every test binary links). */
enum { ETRI_PACK_GUARD = 64 };
#define ETRI_PACK_POISON 0x5A

void etri_pack_guard_fail(const char *where);

/* Poison bytes [used, end) of the segment starting at seg. */
static inline void etri_pack_guard_poison(void *seg, size_t used, size_t end)
{
    unsigned char *p = (unsigned char *)seg;
    for (size_t i = used; i < end; ++i) p[i] = ETRI_PACK_POISON;
}

static inline void etri_pack_guard_check(const void *seg, size_t used, size_t end,
                                         const char *where)
{
    const unsigned char *p = (const unsigned char *)seg;
    for (size_t i = used; i < end; ++i)
        if (p[i] != ETRI_PACK_POISON) etri_pack_guard_fail(where);
}

#endif /* EPBLAS_PARALLEL_KIND10_ETRI_KERNEL_H */
