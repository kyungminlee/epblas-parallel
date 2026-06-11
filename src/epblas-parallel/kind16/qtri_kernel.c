/*
 * qtri_kernel — the ob-convention L3 GEMM substrate shared by the kind16
 * (REAL(KIND=16) / __float128) packed rank-k routines (qsyrk, qsyr2k): a
 * private MR=NR=2 GEMM micro-kernel, its ncopy/tcopy packers, and a
 * zero-then-accumulate "store" variant. A faithful __float128 port of
 * kind10's etri_kernel.c (OpenBLAS kernel/generic/gemm{kernel,_ncopy,_tcopy}_2.c).
 *
 * These are NOT par's qgemm primitives: a SYRK/SYR2K diagonal kernel GEMMs
 * strict-triangle remainders at arbitrary (possibly odd) offsets into the
 * packed buffers, and OpenBLAS's contiguous-odd-tail packing convention is
 * baked into the packer ↔ kernel pair. par's qgemm zero-pads odd tails at
 * stride MR, so its kernel reads those bytes differently. See qtri_kernel.h
 * for the full rationale.
 *
 *   - qtri_gemm_kernel:  C += alpha·Ap·Bp over one packed (bm,bn,bk) tile.
 *   - qtri_kernel_store: C := alpha·Ap·Bp (zero C then accumulate).
 *   - qtri_{n,t}copy:    pack a plain (non-triangular) A/B slab.
 */

#include <stddef.h>

#include "qtri_kernel.h"

typedef qtri_T T;

/* Register-tile dims — must match the syrk/syr2k diagonal kernels' layout. */
#define MR 2
#define NR 2

void qtri_gemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        T alpha,
                        const T *Ap,
                        const T *Bp,
                        T *C, ptrdiff_t ldc)
{
    /* Walk B in NR=2-col panels. */
    const T *ptrba_base = Ap;
    const T *ptrbb = Bp;
    T *Cj = C;

    for (ptrdiff_t j = 0; j < bn / NR; ++j) {
        T *C0 = Cj;
        T *C1 = C0 + ldc;
        const T *ptrba = ptrba_base;

        /* MR=2 row panels (full 2x2 tiles). */
        for (ptrdiff_t i = 0; i < bm / MR; ++i) {
            const T *ptrbb_loc = ptrbb;
            T r0 = 0, r1 = 0, r2 = 0, r3 = 0;

            /* K-loop unrolled by 4 (same shape as OpenBLAS gen kernel). */
            ptrdiff_t k4 = bk / 4;
            for (ptrdiff_t k = 0; k < k4; ++k) {
                T a0 = ptrba[0], a1 = ptrba[1];
                T b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
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
                T a0 = ptrba[0], a1 = ptrba[1];
                T b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                r0 += a0 * b0; r1 += a1 * b0;
                r2 += a0 * b1; r3 += a1 * b1;
                ptrba += 2;
                ptrbb_loc += 2;
            }

            C0[0] += alpha * r0;
            C0[1] += alpha * r1;
            C1[0] += alpha * r2;
            C1[1] += alpha * r3;
            C0 += 2;
            C1 += 2;
        }

        /* bm & 1 — single-row tail (mr=1). */
        for (ptrdiff_t i = 0; i < (bm & 1); ++i) {
            const T *ptrbb_loc = ptrbb;
            T r0 = 0, r1 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                T a0 = ptrba[0];
                T b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                r0 += a0 * b0;
                r1 += a0 * b1;
                ptrba += 1;
                ptrbb_loc += 2;
            }
            C0[0] += alpha * r0;
            C1[0] += alpha * r1;
            C0 += 1;
            C1 += 1;
        }

        ptrbb += bk * 2;       /* advance to next 2-col B panel */
        Cj += 2 * ldc;
    }

    /* bn & 1 — single-col tail (nr=1). */
    for (ptrdiff_t j = 0; j < (bn & 1); ++j) {
        T *C0 = Cj;
        const T *ptrba = ptrba_base;

        for (ptrdiff_t i = 0; i < bm / MR; ++i) {
            const T *ptrbb_loc = ptrbb;
            T r0 = 0, r1 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                T a0 = ptrba[0], a1 = ptrba[1];
                T b0 = ptrbb_loc[0];
                r0 += a0 * b0;
                r1 += a1 * b0;
                ptrba += 2;
                ptrbb_loc += 1;
            }
            C0[0] += alpha * r0;
            C0[1] += alpha * r1;
            C0 += 2;
        }
        for (ptrdiff_t i = 0; i < (bm & 1); ++i) {
            const T *ptrbb_loc = ptrbb;
            T r0 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                r0 += ptrba[0] * ptrbb_loc[0];
                ptrba += 1;
                ptrbb_loc += 1;
            }
            C0[0] += alpha * r0;
            C0 += 1;
        }
        ptrbb += bk;           /* advance over single-col B panel */
        Cj += ldc;
    }
}

void qtri_ncopy(ptrdiff_t m, ptrdiff_t n,
                       const T *a, ptrdiff_t lda,
                       T *b)
{
    const T *a_off = a;
    T *b_off = b;
    ptrdiff_t j = n >> 1;

    while (j > 0) {
        const T *a_off1 = a_off;
        const T *a_off2 = a_off + lda;
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

void qtri_tcopy(ptrdiff_t m, ptrdiff_t n,
                       const T *a, ptrdiff_t lda,
                       T *b)
{
    const T *a_off = a;
    T *b_off = b;
    T *b_off2 = b + m * (n & ~(ptrdiff_t)1);

    ptrdiff_t i = m >> 1;
    while (i > 0) {
        const T *a_off1 = a_off;
        const T *a_off2 = a_off + lda;
        a_off += 2 * lda;

        T *b_off1 = b_off;
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
 * sub-tiles inside the TRMM driver; qtri_gemm_kernel only accumulates. */
void qtri_kernel_store(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk, T alpha,
                       const T *Ap, const T *Bp, T *C, ptrdiff_t ldc)
{
    for (ptrdiff_t j = 0; j < bn; ++j) {
        T *cj = C + j * ldc;
        for (ptrdiff_t i = 0; i < bm; ++i) cj[i] = 0;
    }
    qtri_gemm_kernel(bm, bn, bk, alpha, Ap, Bp, C, ldc);
}
