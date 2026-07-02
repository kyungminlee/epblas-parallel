/*
 * etri_kernel — the ob-convention L3 GEMM substrate shared by the kind10
 * (REAL(KIND=10) / long double) triangular routines (etrsm, etrmm): a
 * private MR=NR=2 GEMM micro-kernel, its ncopy/tcopy packers, and a
 * zero-then-accumulate "store" variant. Faithful ports of OpenBLAS
 * kernel/generic/gemm{kernel,_ncopy,_tcopy}_2.c.
 *
 * These are NOT par's egemm primitives: a triangular solve/trmm and its
 * trailing GEMM share one packed diagonal-block buffer at MR/NR granularity,
 * and OpenBLAS's contiguous-odd-tail packing convention is baked into the
 * packer ↔ kernel pair. par's egemm zero-pads odd tails at stride MR, so its
 * kernel reads those bytes differently. See etri_kernel.h for the full
 * rationale.
 *
 *   - etri_gemm_kernel:  C += alpha·Ap·Bp over one packed (bm,bn,bk) tile.
 *   - etri_kernel_store: C := alpha·Ap·Bp (zero C then accumulate).
 *   - etri_{n,t}copy:    pack a plain (non-triangular) A/B slab.
 */

#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

#include "etri_kernel.h"

typedef etri_TR TR;

/* Cold abort path for the pack-scratch overrun guards (etri_kernel.h) —
 * out of line so the inline check carries only a compare and a call. */
void etri_pack_guard_fail(const char *where)
{
    fprintf(stderr,
            "epblas-parallel: %s: pack scratch overrun detected "
            "(poisoned segment tail was overwritten)\n", where);
    abort();
}

/* Register-tile dims — must match the triangular packers' packed layout. */
#define MR 2
#define NR 2

/* Shared body for both the general (C += alpha·Ap·Bp) and the negate-
 * specialized (C -= Ap·Bp) kernels. `negate` is a compile-time constant at
 * each instantiation, so GCC folds the store: the negate=1 clone drops the
 * per-element `alpha *` multiply (and its alpha reload) to a bare `fsubp`,
 * matching what OpenBLAS gets for free from IPA constant-propagation when its
 * solve and GEMM live in one TU (the alpha = dm1 = -1 callers). `C += (-1)·r`
 * and `C -= r` are bit-identical — negation is exact in fp80, so subtraction
 * carries the single identical rounding. The multiply elided here is amortized
 * over bk, so it is the small-N (shallow trailing-GEMM) cost that closes the
 * uniform ~4% par/ob gap at N=64. */
static inline __attribute__((always_inline)) void
etri_gemm_body(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
               TR alpha, int negate,
               const TR *Ap, const TR *Bp,
               TR *C, ptrdiff_t ldc)
{
    /* Walk B in NR=2-col panels. */
    const TR *ptrba_base = Ap;
    const TR *ptrbb = Bp;
    TR *Cj = C;

    for (ptrdiff_t j = 0; j < bn / NR; ++j) {
        TR *C0 = Cj;
        TR *C1 = C0 + ldc;
        const TR *ptrba = ptrba_base;

        /* MR=2 row panels (full 2x2 tiles). */
        for (ptrdiff_t i = 0; i < bm / MR; ++i) {
            const TR *ptrbb_loc = ptrbb;
            TR r0 = 0, r1 = 0, r2 = 0, r3 = 0;

            /* K-loop unrolled by 4 (same shape as OpenBLAS gen kernel). */
            ptrdiff_t k4 = bk / 4;
            for (ptrdiff_t k = 0; k < k4; ++k) {
                TR a0 = ptrba[0], a1 = ptrba[1];
                TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
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
                TR a0 = ptrba[0], a1 = ptrba[1];
                TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
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
            const TR *ptrbb_loc = ptrbb;
            TR r0 = 0, r1 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                TR a0 = ptrba[0];
                TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
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
        TR *C0 = Cj;
        const TR *ptrba = ptrba_base;

        for (ptrdiff_t i = 0; i < bm / MR; ++i) {
            const TR *ptrbb_loc = ptrbb;
            TR r0 = 0, r1 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                TR a0 = ptrba[0], a1 = ptrba[1];
                TR b0 = ptrbb_loc[0];
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
            const TR *ptrbb_loc = ptrbb;
            TR r0 = 0;
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

/* General C += alpha·Ap·Bp. Instantiates the body with negate=0, so the
 * subtract path folds away — this function is byte-for-byte the original
 * (etrmm and any non-(-1) alpha caller land here, unchanged). */
void etri_gemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        TR alpha,
                        const TR *Ap,
                        const TR *Bp,
                        TR *C, ptrdiff_t ldc)
{
    etri_gemm_body(bm, bn, bk, alpha, 0, Ap, Bp, C, ldc);
}

/* C -= Ap·Bp — the alpha = -1 trailing update every triangular solve/trmm
 * driver issues. A SEPARATE function (not a branch inside etri_gemm_kernel):
 * instantiating the body with the compile-time negate=1 folds the per-element
 * `alpha *` multiply and its alpha reload down to a bare `fsubp`, so this
 * function is no larger than the original. Routing the solve's calls here
 * (rather than dispatching on `alpha == -1` inside one fattened kernel) keeps
 * each kernel compact — carrying both store paths in one function bloats it
 * and regressed the icache-bound small-N cells. This is what OpenBLAS gets
 * for free via IPA constant-propagation (its solve and GEMM share a TU); we
 * do it explicitly across the etri_kernel.c boundary. */
void etri_gemm_kernel_msub(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                           const TR *Ap, const TR *Bp,
                           TR *C, ptrdiff_t ldc)
{
    etri_gemm_body(bm, bn, bk, -1.0L, 1, Ap, Bp, C, ldc);
}

void etri_ncopy(ptrdiff_t m, ptrdiff_t n,
                       const TR *a, ptrdiff_t lda,
                       TR *b)
{
    const TR *a_off = a;
    TR *b_off = b;
    ptrdiff_t j = n >> 1;

    while (j > 0) {
        const TR *a_off1 = a_off;
        const TR *a_off2 = a_off + lda;
        a_off += 2 * lda;

        ptrdiff_t i = m >> 2;
        while (i > 0) {
            b_off[0] = a_off1[0]; b_off[1] = a_off2[0];
            b_off[2] = a_off1[1]; b_off[3] = a_off2[1];
            b_off[4] = a_off1[2]; b_off[5] = a_off2[2];
            b_off[6] = a_off1[3]; b_off[7] = a_off2[3];
            a_off1 += 4;
            a_off2 += 4;
            b_off += 8;
            --i;
        }
        for (i = m & 3; i > 0; --i) {
            b_off[0] = a_off1[0];
            b_off[1] = a_off2[0];
            ++a_off1;
            ++a_off2;
            b_off += 2;
        }
        --j;
    }

    if (n & 1) {
        ptrdiff_t i = m >> 3;
        while (i > 0) {
            b_off[0] = a_off[0]; b_off[1] = a_off[1];
            b_off[2] = a_off[2]; b_off[3] = a_off[3];
            b_off[4] = a_off[4]; b_off[5] = a_off[5];
            b_off[6] = a_off[6]; b_off[7] = a_off[7];
            a_off += 8;
            b_off += 8;
            --i;
        }
        for (i = m & 7; i > 0; --i) {
            *b_off++ = *a_off++;
        }
    }
}

void etri_tcopy(ptrdiff_t m, ptrdiff_t n,
                       const TR *a, ptrdiff_t lda,
                       TR *b)
{
    const TR *a_off = a;
    TR *b_off = b;
    TR *b_off2 = b + m * (n & ~(ptrdiff_t)1);

    ptrdiff_t i = m >> 1;
    while (i > 0) {
        const TR *a_off1 = a_off;
        const TR *a_off2 = a_off + lda;
        a_off += 2 * lda;

        TR *b_off1 = b_off;
        b_off += 4;

        ptrdiff_t j = n >> 1;
        while (j > 0) {
            b_off1[0] = a_off1[0];
            b_off1[1] = a_off1[1];
            b_off1[2] = a_off2[0];
            b_off1[3] = a_off2[1];
            a_off1 += 2;
            a_off2 += 2;
            b_off1 += m * 2;
            --j;
        }
        if (n & 1) {
            b_off2[0] = a_off1[0];
            b_off2[1] = a_off2[0];
            b_off2 += 2;
        }
        --i;
    }

    if (m & 1) {
        ptrdiff_t j = n >> 1;
        while (j > 0) {
            b_off[0] = a_off[0];
            b_off[1] = a_off[1];
            a_off += 2;
            b_off += m * 2;
            --j;
        }
        if (n & 1) {
            b_off2[0] = a_off[0];
        }
    }
}

/* C := alpha·Ap·Bp (overwrite): zero the tile, then accumulate via the
 * shared kernel. OpenBLAS uses GEMM_KERNEL with beta=0 for off-diagonal
 * sub-tiles inside the TRMM driver; etri_gemm_kernel only accumulates. */
void etri_kernel_store(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk, TR alpha,
                       const TR *Ap, const TR *Bp, TR *C, ptrdiff_t ldc)
{
    for (ptrdiff_t j = 0; j < bn; ++j) {
        TR *cj = C + j * ldc;
        for (ptrdiff_t i = 0; i < bm; ++i) cj[i] = 0;
    }
    etri_gemm_kernel(bm, bn, bk, alpha, Ap, Bp, C, ldc);
}
