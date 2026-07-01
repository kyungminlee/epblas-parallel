/*
 * xl3_complex.c — shared L3 microkernel + packers (kind16 complex), par
 * overlay copy. A faithful transcription of the epblas-openblas substrate
 * src/epblas-openblas/common/qblas_l3_complex.c: the par overlay keeps its
 * own copy (mirrors the kind10 etri_kernel.c convention) so the parallel
 * x* drivers below it never reach across into the openblas tree.
 *
 * Port source: OpenBLAS.
 *   - kernel/generic/zgemmkernel_2x2.c   ← qblas_xgemm_kernel
 *                                          (NN path only; conjugation absorbed
 *                                           into the packers via the `conj` flag)
 *   - kernel/generic/zgemm_ncopy_2.c     ← qblas_xgemm_ncopy
 *   - kernel/generic/zgemm_tcopy_2.c     ← qblas_xgemm_tcopy
 *   - kernel/generic/zgemm_beta.c        ← qblas_xgemm_beta
 */

#include "xl3_complex.h"
#include <quadmath.h>
#include <stdbool.h>
#include <stdlib.h>

typedef __float128 TR;

#define MR QBLAS_XGEMM_MR
#define NR QBLAS_XGEMM_NR


/* ── Microkernel: 2x2 complex outer-product over K (NN path) ──────────
 *
 * Per K-iter (unconjugated complex product):
 *   res0,res1   = alpha-acc of (a0+a1*i)(b0+b1*i)  for tile [0,0]
 *   res2,res3   = (a2+a3*i)(b0+b1*i)               for tile [1,0]
 *   res4,res5   = (a0+a1*i)(b2+b3*i)               for tile [0,1]
 *   res6,res7   = (a2+a3*i)(b2+b3*i)               for tile [1,1]
 *
 * where (a0,a1,a2,a3) = ptrba[0..3], (b0,b1,b2,b3) = ptrbb[0..3].
 *
 * Order of operations matches OpenBLAS zgemmkernel_2x2.c so the
 * float-summation order is identical across the K-loop.
 */
void qblas_xgemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        TR alphar, TR alphai,
                        const TR *Ap,
                        const TR *Bp,
                        TR *C, ptrdiff_t ldc)
{
    /* ldc passed in complex elements; ldc2 = stride in __float128s. */
    const ptrdiff_t ldc2 = 2 * ldc;

    const TR *ptrba_base = Ap;
    const TR *ptrbb = Bp;
    TR *Cj = C;

    for (ptrdiff_t j = 0; j < bn / NR; ++j) {
        TR *C0 = Cj;
        TR *C1 = C0 + ldc2;
        const TR *ptrba = ptrba_base;

        /* MR=2 full tiles. */
        for (ptrdiff_t i = 0; i < bm / MR; ++i) {
            const TR *ptrbb_loc = ptrbb;
            TR res0 = 0, res1 = 0, res2 = 0, res3 = 0;
            TR res4 = 0, res5 = 0, res6 = 0, res7 = 0;

            /* K-loop unrolled by 4 to match OpenBLAS's hand-unroll;
               4 sub-iters per outer iter, each consumes 4 floats
               from each of Ap and Bp. */
            ptrdiff_t k4 = bk / 4;
            for (ptrdiff_t k = 0; k < k4; ++k) {
                for (ptrdiff_t u = 0; u < 4; ++u) {
                    TR a0 = ptrba[0], a1 = ptrba[1];
                    TR a2 = ptrba[2], a3 = ptrba[3];
                    TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                    TR b2 = ptrbb_loc[2], b3 = ptrbb_loc[3];
                    res0 += a0 * b0;  res1 += a1 * b0;
                    res0 -= a1 * b1;  res1 += a0 * b1;
                    res2 += a2 * b0;  res3 += a3 * b0;
                    res2 -= a3 * b1;  res3 += a2 * b1;
                    res4 += a0 * b2;  res5 += a1 * b2;
                    res4 -= a1 * b3;  res5 += a0 * b3;
                    res6 += a2 * b2;  res7 += a3 * b2;
                    res6 -= a3 * b3;  res7 += a2 * b3;
                    ptrba += 4;
                    ptrbb_loc += 4;
                }
            }
            for (ptrdiff_t k = 0; k < (bk & 3); ++k) {
                TR a0 = ptrba[0], a1 = ptrba[1];
                TR a2 = ptrba[2], a3 = ptrba[3];
                TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                TR b2 = ptrbb_loc[2], b3 = ptrbb_loc[3];
                res0 += a0 * b0;  res1 += a1 * b0;
                res0 -= a1 * b1;  res1 += a0 * b1;
                res2 += a2 * b0;  res3 += a3 * b0;
                res2 -= a3 * b1;  res3 += a2 * b1;
                res4 += a0 * b2;  res5 += a1 * b2;
                res4 -= a1 * b3;  res5 += a0 * b3;
                res6 += a2 * b2;  res7 += a3 * b2;
                res6 -= a3 * b3;  res7 += a2 * b3;
                ptrba += 4;
                ptrbb_loc += 4;
            }

            /* Apply complex alpha and accumulate.
               (res0+res1*i)*(alphar+alphai*i) = (res0*ar - res1*ai)
                                                + (res1*ar + res0*ai)*i */
            C0[0] += res0 * alphar - res1 * alphai;
            C0[1] += res1 * alphar + res0 * alphai;
            C0[2] += res2 * alphar - res3 * alphai;
            C0[3] += res3 * alphar + res2 * alphai;
            C1[0] += res4 * alphar - res5 * alphai;
            C1[1] += res5 * alphar + res4 * alphai;
            C1[2] += res6 * alphar - res7 * alphai;
            C1[3] += res7 * alphar + res6 * alphai;

            C0 += 4;  /* 2 complex M-rows */
            C1 += 4;
        }

        /* bm & 1 — single-M-row tail (mr=1, nr=2). */
        for (ptrdiff_t i = 0; i < (bm & 1); ++i) {
            const TR *ptrbb_loc = ptrbb;
            TR res0 = 0, res1 = 0, res2 = 0, res3 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                TR a0 = ptrba[0], a1 = ptrba[1];
                TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                TR b2 = ptrbb_loc[2], b3 = ptrbb_loc[3];
                res0 += a0 * b0;  res1 += a1 * b0;
                res0 -= a1 * b1;  res1 += a0 * b1;
                res2 += a0 * b2;  res3 += a1 * b2;
                res2 -= a1 * b3;  res3 += a0 * b3;
                ptrba += 2;
                ptrbb_loc += 4;
            }
            C0[0] += res0 * alphar - res1 * alphai;
            C0[1] += res1 * alphar + res0 * alphai;
            C1[0] += res2 * alphar - res3 * alphai;
            C1[1] += res3 * alphar + res2 * alphai;
            C0 += 2;
            C1 += 2;
        }

        ptrbb += bk * 4;       /* advance to next NR=2 B panel */
        Cj += 2 * ldc2;        /* advance C by 2 complex cols */
    }

    /* bn & 1 — single-N-col tail. */
    for (ptrdiff_t j = 0; j < (bn & 1); ++j) {
        TR *C0 = Cj;
        const TR *ptrba = ptrba_base;

        for (ptrdiff_t i = 0; i < bm / MR; ++i) {
            const TR *ptrbb_loc = ptrbb;
            TR res0 = 0, res1 = 0, res2 = 0, res3 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                TR a0 = ptrba[0], a1 = ptrba[1];
                TR a2 = ptrba[2], a3 = ptrba[3];
                TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                res0 += a0 * b0;  res1 += a1 * b0;
                res0 -= a1 * b1;  res1 += a0 * b1;
                res2 += a2 * b0;  res3 += a3 * b0;
                res2 -= a3 * b1;  res3 += a2 * b1;
                ptrba += 4;
                ptrbb_loc += 2;
            }
            C0[0] += res0 * alphar - res1 * alphai;
            C0[1] += res1 * alphar + res0 * alphai;
            C0[2] += res2 * alphar - res3 * alphai;
            C0[3] += res3 * alphar + res2 * alphai;
            C0 += 4;
        }
        for (ptrdiff_t i = 0; i < (bm & 1); ++i) {
            const TR *ptrbb_loc = ptrbb;
            TR res0 = 0, res1 = 0;
            for (ptrdiff_t k = 0; k < bk; ++k) {
                TR a0 = ptrba[0], a1 = ptrba[1];
                TR b0 = ptrbb_loc[0], b1 = ptrbb_loc[1];
                res0 += a0 * b0;  res1 += a1 * b0;
                res0 -= a1 * b1;  res1 += a0 * b1;
                ptrba += 2;
                ptrbb_loc += 2;
            }
            C0[0] += res0 * alphar - res1 * alphai;
            C0[1] += res1 * alphar + res0 * alphai;
            C0 += 2;
        }
        ptrbb += bk * 2;       /* single-N-col B panel */
        Cj += ldc2;
    }
}


/* ── ncopy: pack 2 source cols per panel, optional conj ──────────────
 *
 * Faithful translation of OpenBLAS zgemm_ncopy_2.c with `conj` added.
 * When `conj` is set, every imag float (odd-index in the interleaved
 * pair) is negated as it's written into `b`.
 */
void qblas_xgemm_ncopy(ptrdiff_t m, ptrdiff_t n,
                       bool conj,
                       const TR *a, ptrdiff_t lda,
                       TR *b)
{
    const TR sign = conj ? -1.0Q : 1.0Q;
    const TR *a_off = a;
    TR *b_off = b;
    const ptrdiff_t lda2 = lda * 2;
    ptrdiff_t j = n >> 1;

    while (j > 0) {
        const TR *a_off1 = a_off;
        const TR *a_off2 = a_off + lda2;
        a_off += 2 * lda2;

        ptrdiff_t i = m >> 2;
        while (i > 0) {
            b_off[ 0] = a_off1[0]; b_off[ 1] = sign * a_off1[1];
            b_off[ 2] = a_off2[0]; b_off[ 3] = sign * a_off2[1];

            b_off[ 4] = a_off1[2]; b_off[ 5] = sign * a_off1[3];
            b_off[ 6] = a_off2[2]; b_off[ 7] = sign * a_off2[3];

            b_off[ 8] = a_off1[4]; b_off[ 9] = sign * a_off1[5];
            b_off[10] = a_off2[4]; b_off[11] = sign * a_off2[5];

            b_off[12] = a_off1[6]; b_off[13] = sign * a_off1[7];
            b_off[14] = a_off2[6]; b_off[15] = sign * a_off2[7];

            a_off1 += 8;
            a_off2 += 8;
            b_off += 16;
            --i;
        }
        for (i = m & 3; i > 0; --i) {
            b_off[0] = a_off1[0]; b_off[1] = sign * a_off1[1];
            b_off[2] = a_off2[0]; b_off[3] = sign * a_off2[1];
            a_off1 += 2;
            a_off2 += 2;
            b_off += 4;
        }
        --j;
    }

    if (n & 1) {
        ptrdiff_t i = m >> 2;
        while (i > 0) {
            b_off[0] = a_off[0]; b_off[1] = sign * a_off[1];
            b_off[2] = a_off[2]; b_off[3] = sign * a_off[3];
            b_off[4] = a_off[4]; b_off[5] = sign * a_off[5];
            b_off[6] = a_off[6]; b_off[7] = sign * a_off[7];
            a_off += 8;
            b_off += 8;
            --i;
        }
        for (i = m & 3; i > 0; --i) {
            b_off[0] = a_off[0]; b_off[1] = sign * a_off[1];
            a_off += 2;
            b_off += 2;
        }
    }
}


/* ── tcopy: faithful port of OpenBLAS zgemm_tcopy_2.c, with conj ──── */
void qblas_xgemm_tcopy(ptrdiff_t m, ptrdiff_t n,
                       bool conj,
                       const TR *a, ptrdiff_t lda,
                       TR *b)
{
    const TR sign = conj ? -1.0Q : 1.0Q;
    const TR *a_off = a;
    TR *b_off = b;
    TR *b_off2 = b + m * (n & ~(ptrdiff_t)1) * 2;
    const ptrdiff_t lda2 = lda * 2;

    ptrdiff_t i = m >> 1;
    while (i > 0) {
        const TR *a_off1 = a_off;
        const TR *a_off2 = a_off + lda2;
        a_off += 2 * lda2;

        TR *b_off1 = b_off;
        b_off += 8;

        ptrdiff_t j = n >> 2;
        while (j > 0) {
            /* K-pair (2 source cols) × M-rows (4j, 4j+1):  8 floats */
            b_off1[0] = a_off1[0]; b_off1[1] = sign * a_off1[1];
            b_off1[2] = a_off1[2]; b_off1[3] = sign * a_off1[3];
            b_off1[4] = a_off2[0]; b_off1[5] = sign * a_off2[1];
            b_off1[6] = a_off2[2]; b_off1[7] = sign * a_off2[3];

            b_off1 += m * 4;

            /* K-pair × M-rows (4j+2, 4j+3):  8 floats */
            b_off1[0] = a_off1[4]; b_off1[1] = sign * a_off1[5];
            b_off1[2] = a_off1[6]; b_off1[3] = sign * a_off1[7];
            b_off1[4] = a_off2[4]; b_off1[5] = sign * a_off2[5];
            b_off1[6] = a_off2[6]; b_off1[7] = sign * a_off2[7];

            b_off1 += m * 4;

            a_off1 += 8;
            a_off2 += 8;
            --j;
        }
        if (n & 2) {
            b_off1[0] = a_off1[0]; b_off1[1] = sign * a_off1[1];
            b_off1[2] = a_off1[2]; b_off1[3] = sign * a_off1[3];
            b_off1[4] = a_off2[0]; b_off1[5] = sign * a_off2[1];
            b_off1[6] = a_off2[2]; b_off1[7] = sign * a_off2[3];
            a_off1 += 4;
            a_off2 += 4;
            /* no b_off1 advance — this is the last sub-iter in the strip */
        }
        if (n & 1) {
            b_off2[0] = a_off1[0]; b_off2[1] = sign * a_off1[1];
            b_off2[2] = a_off2[0]; b_off2[3] = sign * a_off2[1];
            b_off2 += 4;
        }
        --i;
    }

    if (m & 1) {
        TR *b_off1 = b_off;
        ptrdiff_t j = n >> 2;
        while (j > 0) {
            b_off1[0] = a_off[0]; b_off1[1] = sign * a_off[1];
            b_off1[2] = a_off[2]; b_off1[3] = sign * a_off[3];
            b_off1 += m * 4;
            b_off1[0] = a_off[4]; b_off1[1] = sign * a_off[5];
            b_off1[2] = a_off[6]; b_off1[3] = sign * a_off[7];
            b_off1 += m * 4;
            a_off += 8;
            --j;
        }
        if (n & 2) {
            b_off1[0] = a_off[0]; b_off1[1] = sign * a_off[1];
            b_off1[2] = a_off[2]; b_off1[3] = sign * a_off[3];
            a_off += 4;
        }
        if (n & 1) {
            b_off2[0] = a_off[0]; b_off2[1] = sign * a_off[1];
        }
    }
}


/* ── Beta pre-pass: C := beta * C with complex beta ──────────────── */
void qblas_xgemm_beta(ptrdiff_t m, ptrdiff_t n,
                      TR beta_r, TR beta_i,
                      TR *c, ptrdiff_t ldc)
{
    const ptrdiff_t ldc2 = ldc * 2;

    if (beta_r == 1.0Q && beta_i == 0.0Q) return;

    if (beta_r == 0.0Q && beta_i == 0.0Q) {
        for (ptrdiff_t j = 0; j < n; ++j) {
            TR *cj = c + j * ldc2;
            for (ptrdiff_t i = 0; i < m * 2; ++i) cj[i] = 0;
        }
        return;
    }

    for (ptrdiff_t j = 0; j < n; ++j) {
        TR *cj = c + j * ldc2;
        for (ptrdiff_t i = 0; i < m; ++i) {
            TR re = cj[2*i + 0];
            TR im = cj[2*i + 1];
            cj[2*i + 0] = beta_r * re - beta_i * im;
            cj[2*i + 1] = beta_r * im + beta_i * re;
        }
    }
}


/* ── Block-size constants ───────────────────────────────────────── */
void qblas_xgemm_blocks(ptrdiff_t *mc, ptrdiff_t *kc, ptrdiff_t *nc) {
    *mc = QBLAS_XGEMM_GEMM_P; *kc = QBLAS_XGEMM_GEMM_Q; *nc = QBLAS_XGEMM_GEMM_R;
}


/* ── SYMM-aware packers (complex) ────────────────────────────────────
 *
 * Faithful ports of OpenBLAS kernel/generic/zsymm_{u,l}copy_2.c.
 *
 * Per element: 2 __float128s (re, im) interleaved. lda passed in
 * complex elements; doubled internally to lda2 for float-stride
 * arithmetic. SYMM has no conjugation, so the imag part is copied
 * through unchanged in both branches.
 *
 * The "+2 vs +lda2" advance pattern mirrors the real path's "+1 vs
 * +lda" — each path corresponds to a different direction of mirror
 * across the diagonal (column-walk vs row-walk in storage).
 */
void qblas_xsymm_ucopy(ptrdiff_t m, ptrdiff_t n,
                       const TR *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       TR *b)
{
    ptrdiff_t i, js, offset;
    TR data01, data02, data03, data04;
    const TR *ao1, *ao2;
    const ptrdiff_t lda2 = lda * 2;

    js = n >> 1;
    while (js > 0) {
        offset = posX - posY;

        if (offset >  0) ao1 = a + posY * 2 + (posX + 0) * lda2;
        else             ao1 = a + (posX + 0) * 2 + posY * lda2;
        if (offset > -1) ao2 = a + posY * 2 + (posX + 1) * lda2;
        else             ao2 = a + (posX + 1) * 2 + posY * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];
            data03 = ao2[0];
            data04 = ao2[1];

            if (offset >   0) ao1 += 2;   else ao1 += lda2;
            if (offset >  -1) ao2 += 2;   else ao2 += lda2;

            b[0] = data01;
            b[1] = data02;
            b[2] = data03;
            b[3] = data04;
            b += 4;

            offset--;
            i--;
        }
        posX += 2;
        js--;
    }

    if (n & 1) {
        offset = posX - posY;

        if (offset > 0) ao1 = a + posY * 2 + (posX + 0) * lda2;
        else            ao1 = a + (posX + 0) * 2 + posY * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];
            if (offset > 0) ao1 += 2; else ao1 += lda2;
            b[0] = data01;
            b[1] = data02;
            b += 2;
            offset--;
            i--;
        }
    }
}


void qblas_xsymm_lcopy(ptrdiff_t m, ptrdiff_t n,
                       const TR *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       TR *b)
{
    ptrdiff_t i, js, offset;
    TR data01, data02, data03, data04;
    const TR *ao1, *ao2;
    const ptrdiff_t lda2 = lda * 2;

    js = n >> 1;
    while (js > 0) {
        offset = posX - posY;

        if (offset >  0) ao1 = a + (posX + 0) * 2 + posY * lda2;
        else             ao1 = a + posY * 2 + (posX + 0) * lda2;
        if (offset > -1) ao2 = a + (posX + 1) * 2 + posY * lda2;
        else             ao2 = a + posY * 2 + (posX + 1) * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];
            data03 = ao2[0];
            data04 = ao2[1];

            if (offset >   0) ao1 += lda2; else ao1 += 2;
            if (offset >  -1) ao2 += lda2; else ao2 += 2;

            b[0] = data01;
            b[1] = data02;
            b[2] = data03;
            b[3] = data04;
            b += 4;

            offset--;
            i--;
        }
        posX += 2;
        js--;
    }

    if (n & 1) {
        offset = posX - posY;

        if (offset > 0) ao1 = a + (posX + 0) * 2 + posY * lda2;
        else            ao1 = a + posY * 2 + (posX + 0) * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];
            if (offset > 0) ao1 += lda2; else ao1 += 2;
            b[0] = data01;
            b[1] = data02;
            b += 2;
            offset--;
            i--;
        }
    }
}


/* ── HEMM-aware packers (complex Hermitian) ──────────────────────────
 *
 * Faithful ports of OpenBLAS kernel/generic/zhemm_utcopy_2.c and
 * zhemm_ltcopy_2.c.
 *
 * The geometry mirrors the SYMM packers exactly — same posX/posY
 * mirror branches across the diagonal — with three Hermitian-specific
 * tweaks:
 *   - reflected-across-diagonal half: imag is negated (conjugation)
 *   - diagonal element: imag is set to ZERO (Hermitian diagonal is
 *     real by definition; the LAPACK ZHEMM contract discards the
 *     input's diagonal imag)
 *   - directly-stored half: imag passes through unchanged
 *
 * Which branch is "directly stored" vs "reflected" differs between
 * ucopy (upper triangle stored) and lcopy (lower triangle stored).
 */
void qblas_xhemm_ucopy(ptrdiff_t m, ptrdiff_t n,
                       const TR *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       TR *b)
{
    ptrdiff_t i, js, offset;
    TR data01, data02, data03, data04;
    const TR *ao1, *ao2;
    const ptrdiff_t lda2 = lda * 2;

    js = n >> 1;
    while (js > 0) {
        offset = posX - posY;

        if (offset >  0) ao1 = a + posY * 2 + (posX + 0) * lda2;
        else             ao1 = a + (posX + 0) * 2 + posY * lda2;
        if (offset > -1) ao2 = a + posY * 2 + (posX + 1) * lda2;
        else             ao2 = a + (posX + 1) * 2 + posY * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];
            data03 = ao2[0];
            data04 = ao2[1];

            if (offset >   0) ao1 += 2;   else ao1 += lda2;
            if (offset >  -1) ao2 += 2;   else ao2 += lda2;

            if (offset > 0) {
                b[0] = data01;
                b[1] = -data02;
                b[2] = data03;
                b[3] = -data04;
            } else if (offset < -1) {
                b[0] = data01;
                b[1] = data02;
                b[2] = data03;
                b[3] = data04;
            } else if (offset == 0) {
                b[0] = data01;
                b[1] = 0;
                b[2] = data03;
                b[3] = -data04;
            } else { /* offset == -1 */
                b[0] = data01;
                b[1] = data02;
                b[2] = data03;
                b[3] = 0;
            }

            b += 4;
            offset--;
            i--;
        }
        posX += 2;
        js--;
    }

    if (n & 1) {
        offset = posX - posY;

        if (offset > 0) ao1 = a + posY * 2 + (posX + 0) * lda2;
        else            ao1 = a + (posX + 0) * 2 + posY * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];

            if (offset > 0) ao1 += 2; else ao1 += lda2;

            if (offset > 0) {
                b[0] = data01;
                b[1] = -data02;
            } else if (offset < 0) {
                b[0] = data01;
                b[1] = data02;
            } else { /* offset == 0 */
                b[0] = data01;
                b[1] = 0;
            }

            b += 2;
            offset--;
            i--;
        }
    }
}


void qblas_xhemm_lcopy(ptrdiff_t m, ptrdiff_t n,
                       const TR *a, ptrdiff_t lda,
                       ptrdiff_t posX, ptrdiff_t posY,
                       TR *b)
{
    ptrdiff_t i, js, offset;
    TR data01, data02, data03, data04;
    const TR *ao1, *ao2;
    const ptrdiff_t lda2 = lda * 2;

    js = n >> 1;
    while (js > 0) {
        offset = posX - posY;

        if (offset >  0) ao1 = a + (posX + 0) * 2 + posY * lda2;
        else             ao1 = a + posY * 2 + (posX + 0) * lda2;
        if (offset > -1) ao2 = a + (posX + 1) * 2 + posY * lda2;
        else             ao2 = a + posY * 2 + (posX + 1) * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];
            data03 = ao2[0];
            data04 = ao2[1];

            if (offset >   0) ao1 += lda2; else ao1 += 2;
            if (offset >  -1) ao2 += lda2; else ao2 += 2;

            if (offset > 0) {
                b[0] = data01;
                b[1] = data02;
                b[2] = data03;
                b[3] = data04;
            } else if (offset < -1) {
                b[0] = data01;
                b[1] = -data02;
                b[2] = data03;
                b[3] = -data04;
            } else if (offset == 0) {
                b[0] = data01;
                b[1] = 0;
                b[2] = data03;
                b[3] = data04;
            } else { /* offset == -1 */
                b[0] = data01;
                b[1] = -data02;
                b[2] = data03;
                b[3] = 0;
            }

            b += 4;
            offset--;
            i--;
        }
        posX += 2;
        js--;
    }

    if (n & 1) {
        offset = posX - posY;

        if (offset > 0) ao1 = a + (posX + 0) * 2 + posY * lda2;
        else            ao1 = a + posY * 2 + (posX + 0) * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];

            if (offset > 0) ao1 += lda2; else ao1 += 2;

            if (offset > 0) {
                b[0] = data01;
                b[1] = data02;
            } else if (offset < 0) {
                b[0] = data01;
                b[1] = -data02;
            } else { /* offset == 0 */
                b[0] = data01;
                b[1] = 0;
            }

            b += 2;
            offset--;
            i--;
        }
    }
}


/* ── HEMM packers (OCOPY variants — SIDE=R role) ─────────────────────
 *
 * Same geometry and address calculations as the IC variants above; the
 * imag-sign decisions are inverted on the off-diagonal branches to
 * match the (col/row) reinterpretation of (posX/posY) that the SIDE=R
 * call site triggers.
 *
 * See the header comment on qblas_xhemm_ucopy_oc for the rationale
 * (why we need a distinct OC variant instead of reusing the IC one as
 * upstream OpenBLAS does).
 */
void qblas_xhemm_ucopy_oc(ptrdiff_t m, ptrdiff_t n,
                          const TR *a, ptrdiff_t lda,
                          ptrdiff_t posX, ptrdiff_t posY,
                          TR *b)
{
    ptrdiff_t i, js, offset;
    TR data01, data02, data03, data04;
    const TR *ao1, *ao2;
    const ptrdiff_t lda2 = lda * 2;

    js = n >> 1;
    while (js > 0) {
        offset = posX - posY;

        if (offset >  0) ao1 = a + posY * 2 + (posX + 0) * lda2;
        else             ao1 = a + (posX + 0) * 2 + posY * lda2;
        if (offset > -1) ao2 = a + posY * 2 + (posX + 1) * lda2;
        else             ao2 = a + (posX + 1) * 2 + posY * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];
            data03 = ao2[0];
            data04 = ao2[1];

            if (offset >   0) ao1 += 2;   else ao1 += lda2;
            if (offset >  -1) ao2 += 2;   else ao2 += lda2;

            if (offset > 0) {
                /* OC: stored directly, no conj */
                b[0] = data01;
                b[1] = data02;
                b[2] = data03;
                b[3] = data04;
            } else if (offset < -1) {
                /* OC: reflected, conjugate */
                b[0] = data01;
                b[1] = -data02;
                b[2] = data03;
                b[3] = -data04;
            } else if (offset == 0) {
                b[0] = data01;
                b[1] = 0;
                b[2] = data03;
                b[3] = data04;
            } else { /* offset == -1 */
                b[0] = data01;
                b[1] = -data02;
                b[2] = data03;
                b[3] = 0;
            }

            b += 4;
            offset--;
            i--;
        }
        posX += 2;
        js--;
    }

    if (n & 1) {
        offset = posX - posY;

        if (offset > 0) ao1 = a + posY * 2 + (posX + 0) * lda2;
        else            ao1 = a + (posX + 0) * 2 + posY * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];

            if (offset > 0) ao1 += 2; else ao1 += lda2;

            if (offset > 0) {
                b[0] = data01;
                b[1] = data02;
            } else if (offset < 0) {
                b[0] = data01;
                b[1] = -data02;
            } else { /* offset == 0 */
                b[0] = data01;
                b[1] = 0;
            }

            b += 2;
            offset--;
            i--;
        }
    }
}


void qblas_xhemm_lcopy_oc(ptrdiff_t m, ptrdiff_t n,
                          const TR *a, ptrdiff_t lda,
                          ptrdiff_t posX, ptrdiff_t posY,
                          TR *b)
{
    ptrdiff_t i, js, offset;
    TR data01, data02, data03, data04;
    const TR *ao1, *ao2;
    const ptrdiff_t lda2 = lda * 2;

    js = n >> 1;
    while (js > 0) {
        offset = posX - posY;

        if (offset >  0) ao1 = a + (posX + 0) * 2 + posY * lda2;
        else             ao1 = a + posY * 2 + (posX + 0) * lda2;
        if (offset > -1) ao2 = a + (posX + 1) * 2 + posY * lda2;
        else             ao2 = a + posY * 2 + (posX + 1) * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];
            data03 = ao2[0];
            data04 = ao2[1];

            if (offset >   0) ao1 += lda2; else ao1 += 2;
            if (offset >  -1) ao2 += lda2; else ao2 += 2;

            if (offset > 0) {
                /* OC: reflected, conjugate */
                b[0] = data01;
                b[1] = -data02;
                b[2] = data03;
                b[3] = -data04;
            } else if (offset < -1) {
                /* OC: stored directly, no conj */
                b[0] = data01;
                b[1] = data02;
                b[2] = data03;
                b[3] = data04;
            } else if (offset == 0) {
                b[0] = data01;
                b[1] = 0;
                b[2] = data03;
                b[3] = -data04;
            } else { /* offset == -1 */
                b[0] = data01;
                b[1] = data02;
                b[2] = data03;
                b[3] = 0;
            }

            b += 4;
            offset--;
            i--;
        }
        posX += 2;
        js--;
    }

    if (n & 1) {
        offset = posX - posY;

        if (offset > 0) ao1 = a + (posX + 0) * 2 + posY * lda2;
        else            ao1 = a + posY * 2 + (posX + 0) * lda2;

        i = m;
        while (i > 0) {
            data01 = ao1[0];
            data02 = ao1[1];

            if (offset > 0) ao1 += lda2; else ao1 += 2;

            if (offset > 0) {
                b[0] = data01;
                b[1] = -data02;
            } else if (offset < 0) {
                b[0] = data01;
                b[1] = data02;
            } else { /* offset == 0 */
                b[0] = data01;
                b[1] = 0;
            }

            b += 2;
            offset--;
            i--;
        }
    }
}


/* ── Triangular beta pre-pass (complex) ──────────────────────────────
 *
 * `c` is interleaved (re, im); `ldc` in complex elements. We unroll
 * inside the row loop instead of touching `qblas_xgemm_beta`'s full-
 * rectangle helper.
 */
static inline void xsyrk_scale_strip(TR *cj_re,
                                     ptrdiff_t lo, ptrdiff_t hi,
                                     TR br, TR bi)
{
    if (br == 1.0Q && bi == 0.0Q) return;
    if (br == 0.0Q && bi == 0.0Q) {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            cj_re[2*i + 0] = 0;
            cj_re[2*i + 1] = 0;
        }
    } else {
        for (ptrdiff_t i = lo; i < hi; ++i) {
            TR re = cj_re[2*i + 0];
            TR im = cj_re[2*i + 1];
            cj_re[2*i + 0] = br * re - bi * im;
            cj_re[2*i + 1] = br * im + bi * re;
        }
    }
}

void qblas_xsyrk_beta_u(ptrdiff_t n, TR br, TR bi, TR *c, ptrdiff_t ldc) {
    if (br == 1.0Q && bi == 0.0Q) return;
    const ptrdiff_t ldc2 = 2 * ldc;
    for (ptrdiff_t j = 0; j < n; ++j) {
        xsyrk_scale_strip(c + j * ldc2, 0, j + 1, br, bi);
    }
}

void qblas_xsyrk_beta_l(ptrdiff_t n, TR br, TR bi, TR *c, ptrdiff_t ldc) {
    if (br == 1.0Q && bi == 0.0Q) return;
    const ptrdiff_t ldc2 = 2 * ldc;
    for (ptrdiff_t j = 0; j < n; ++j) {
        xsyrk_scale_strip(c + j * ldc2, j, n, br, bi);
    }
}


/* ── HERK beta pre-pass (complex Hermitian C) ────────────────────────
 *
 * Faithful port of OpenBLAS driver/level3/zherk_beta.c. Scales the
 * UPLO triangle by REAL beta and forces diag imag = 0 (Hermitian C
 * contract). Off-UPLO triangle is untouched (HERK contract).
 *
 * Diagonal handling deviates from xsyrk_beta: we ALWAYS write imag=0
 * on the diagonal, even when beta == 1.0Q (matches zherk_beta).
 */
void qblas_xherk_beta_u(ptrdiff_t n, TR br, TR *c, ptrdiff_t ldc) {
    const ptrdiff_t ldc2 = 2 * ldc;
    for (ptrdiff_t j = 0; j < n; ++j) {
        TR *col = c + j * ldc2;
        if (br == 0.0Q) {
            for (ptrdiff_t i = 0; i < j; ++i) {
                col[2*i + 0] = 0;
                col[2*i + 1] = 0;
            }
            col[2*j + 0] = 0;
            col[2*j + 1] = 0;
        } else if (br != 1.0Q) {
            for (ptrdiff_t i = 0; i < j; ++i) {
                col[2*i + 0] *= br;
                col[2*i + 1] *= br;
            }
            col[2*j + 0] *= br;
            col[2*j + 1]  = 0;
        } else {
            col[2*j + 1] = 0;
        }
    }
}

void qblas_xherk_beta_l(ptrdiff_t n, TR br, TR *c, ptrdiff_t ldc) {
    const ptrdiff_t ldc2 = 2 * ldc;
    for (ptrdiff_t j = 0; j < n; ++j) {
        TR *col = c + j * ldc2;
        if (br == 0.0Q) {
            col[2*j + 0] = 0;
            col[2*j + 1] = 0;
            for (ptrdiff_t i = j + 1; i < n; ++i) {
                col[2*i + 0] = 0;
                col[2*i + 1] = 0;
            }
        } else if (br != 1.0Q) {
            col[2*j + 0] *= br;
            col[2*j + 1]  = 0;
            for (ptrdiff_t i = j + 1; i < n; ++i) {
                col[2*i + 0] *= br;
                col[2*i + 1] *= br;
            }
        } else {
            col[2*j + 1] = 0;
        }
    }
}


/* ── SYR2K kernel: two-pass diagonal-aware GEMM (complex) ────────────
 *
 * Faithful port of OpenBLAS driver/level3/syr2k_kernel.c (complex).
 * Strip writes accumulate `alpha * Ap * Bp`; the diagonal NR×NR block
 * is merged with the symmetric mirror `subbuf[i,j] + subbuf[j,i]` and
 * is flag-gated. See the real twin for the two-pass calling convention.
 *
 * All pointer arithmetic accounts for COMPSIZE=2 __float128s per
 * complex element. ldc, k, m, n are in complex elements.
 */
void qblas_xsyr2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           TR alphar, TR alphai,
                           const TR *a, const TR *b,
                           TR *c, ptrdiff_t ldc, ptrdiff_t offset, bool flag)
{
    TR subbuf[NR * (NR + 1) * 2];
    const ptrdiff_t ldc2 = 2 * ldc;

    if (m + offset < 0) {
        qblas_xgemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
        return;
    }
    if (n < offset) {
        return;
    }
    if (offset > 0) {
        b += offset * k * 2;
        c += offset * ldc2;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        qblas_xgemm_kernel(m, n - m - offset, k, alphar, alphai,
                           a, b + (m + offset) * k * 2,
                           c + (m + offset) * ldc2, ldc);
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        qblas_xgemm_kernel(-offset, n, k, alphar, alphai, a, b, c, ldc);
        a -= offset * k * 2;
        c -= offset * 2;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (loop > 0) {
            qblas_xgemm_kernel(loop, nn, k, alphar, alphai,
                               a, b + loop * k * 2,
                               c + loop * ldc2, ldc);
        }

        if (flag) {
            for (ptrdiff_t z = 0; z < nn * nn * 2; ++z) subbuf[z] = 0;
            qblas_xgemm_kernel(nn, nn, k, alphar, alphai,
                               a + loop * k * 2, b + loop * k * 2,
                               subbuf, nn);

            TR *cc = c + 2 * loop + loop * ldc2;
            for (ptrdiff_t j = 0; j < nn; ++j) {
                for (ptrdiff_t i = 0; i <= j; ++i) {
                    cc[2*i + 0 + j * ldc2] += subbuf[(i + j * nn) * 2 + 0]
                                            + subbuf[(j + i * nn) * 2 + 0];
                    cc[2*i + 1 + j * ldc2] += subbuf[(i + j * nn) * 2 + 1]
                                            + subbuf[(j + i * nn) * 2 + 1];
                }
            }
        }
    }
}

void qblas_xsyr2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           TR alphar, TR alphai,
                           const TR *a, const TR *b,
                           TR *c, ptrdiff_t ldc, ptrdiff_t offset, bool flag)
{
    TR subbuf[NR * (NR + 1) * 2];
    const ptrdiff_t ldc2 = 2 * ldc;

    if (m + offset < 0) {
        return;
    }
    if (n < offset) {
        qblas_xgemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
        return;
    }
    if (offset > 0) {
        qblas_xgemm_kernel(m, offset, k, alphar, alphai, a, b, c, ldc);
        b += offset * k * 2;
        c += offset * ldc2;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        a -= offset * k * 2;
        c -= offset * 2;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }
    if (m > n - offset) {
        qblas_xgemm_kernel(m - n + offset, n, k, alphar, alphai,
                           a + (n - offset) * k * 2, b,
                           c + (n - offset) * 2, ldc);
        m = n + offset;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (flag) {
            for (ptrdiff_t z = 0; z < nn * nn * 2; ++z) subbuf[z] = 0;
            qblas_xgemm_kernel(nn, nn, k, alphar, alphai,
                               a + loop * k * 2, b + loop * k * 2,
                               subbuf, nn);

            TR *cc = c + 2 * loop + loop * ldc2;
            for (ptrdiff_t j = 0; j < nn; ++j) {
                for (ptrdiff_t i = j; i < nn; ++i) {
                    cc[2*i + 0 + j * ldc2] += subbuf[(i + j * nn) * 2 + 0]
                                            + subbuf[(j + i * nn) * 2 + 0];
                    cc[2*i + 1 + j * ldc2] += subbuf[(i + j * nn) * 2 + 1]
                                            + subbuf[(j + i * nn) * 2 + 1];
                }
            }
        }

        if (m > mm + nn) {
            qblas_xgemm_kernel(m - mm - nn, nn, k, alphar, alphai,
                               a + (mm + nn) * k * 2, b + loop * k * 2,
                               c + (mm + nn) * 2 + loop * ldc2, ldc);
        }
    }
}


/* ── HER2K kernel: two-pass diagonal-aware GEMM (complex Hermitian) ──
 *
 * Faithful port of OpenBLAS driver/level3/zher2k_kernel.c. Structural
 * twin of qblas_xsyr2k_kernel_{u,l}; differs only in the diagonal
 * NR×NR subblock writeback:
 *
 *   imag part subtracts subbuf[j,i] (instead of adding) and the
 *   actual diagonal element is forced imag = 0 (Hermitian C contract).
 *
 * See header for the two-pass calling convention. Conjugation is
 * absorbed by the caller's packers per upstream's GEMM_KERNEL_R/L pick.
 */
void qblas_xher2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           TR alphar, TR alphai,
                           const TR *a, const TR *b,
                           TR *c, ptrdiff_t ldc, ptrdiff_t offset, bool flag)
{
    TR subbuf[NR * (NR + 1) * 2];
    const ptrdiff_t ldc2 = 2 * ldc;

    if (m + offset < 0) {
        qblas_xgemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
        return;
    }
    if (n < offset) {
        return;
    }
    if (offset > 0) {
        b += offset * k * 2;
        c += offset * ldc2;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        qblas_xgemm_kernel(m, n - m - offset, k, alphar, alphai,
                           a, b + (m + offset) * k * 2,
                           c + (m + offset) * ldc2, ldc);
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        qblas_xgemm_kernel(-offset, n, k, alphar, alphai, a, b, c, ldc);
        a -= offset * k * 2;
        c -= offset * 2;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (loop > 0) {
            qblas_xgemm_kernel(loop, nn, k, alphar, alphai,
                               a, b + loop * k * 2,
                               c + loop * ldc2, ldc);
        }

        if (flag) {
            for (ptrdiff_t z = 0; z < nn * nn * 2; ++z) subbuf[z] = 0;
            qblas_xgemm_kernel(nn, nn, k, alphar, alphai,
                               a + loop * k * 2, b + loop * k * 2,
                               subbuf, nn);

            TR *cc = c + 2 * loop + loop * ldc2;
            for (ptrdiff_t j = 0; j < nn; ++j) {
                for (ptrdiff_t i = 0; i <= j; ++i) {
                    cc[2*i + 0 + j * ldc2] += subbuf[(i + j * nn) * 2 + 0]
                                            + subbuf[(j + i * nn) * 2 + 0];
                    if (i != j) {
                        cc[2*i + 1 + j * ldc2] += subbuf[(i + j * nn) * 2 + 1]
                                                - subbuf[(j + i * nn) * 2 + 1];
                    } else {
                        cc[2*i + 1 + j * ldc2] = 0;
                    }
                }
            }
        }
    }
}

void qblas_xher2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           TR alphar, TR alphai,
                           const TR *a, const TR *b,
                           TR *c, ptrdiff_t ldc, ptrdiff_t offset, bool flag)
{
    TR subbuf[NR * (NR + 1) * 2];
    const ptrdiff_t ldc2 = 2 * ldc;

    if (m + offset < 0) {
        return;
    }
    if (n < offset) {
        qblas_xgemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
        return;
    }
    if (offset > 0) {
        qblas_xgemm_kernel(m, offset, k, alphar, alphai, a, b, c, ldc);
        b += offset * k * 2;
        c += offset * ldc2;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        a -= offset * k * 2;
        c -= offset * 2;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }
    if (m > n - offset) {
        qblas_xgemm_kernel(m - n + offset, n, k, alphar, alphai,
                           a + (n - offset) * k * 2, b,
                           c + (n - offset) * 2, ldc);
        m = n + offset;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (flag) {
            for (ptrdiff_t z = 0; z < nn * nn * 2; ++z) subbuf[z] = 0;
            qblas_xgemm_kernel(nn, nn, k, alphar, alphai,
                               a + loop * k * 2, b + loop * k * 2,
                               subbuf, nn);

            TR *cc = c + 2 * loop + loop * ldc2;
            for (ptrdiff_t j = 0; j < nn; ++j) {
                for (ptrdiff_t i = j; i < nn; ++i) {
                    cc[2*i + 0 + j * ldc2] += subbuf[(i + j * nn) * 2 + 0]
                                            + subbuf[(j + i * nn) * 2 + 0];
                    if (i != j) {
                        cc[2*i + 1 + j * ldc2] += subbuf[(i + j * nn) * 2 + 1]
                                                - subbuf[(j + i * nn) * 2 + 1];
                    } else {
                        cc[2*i + 1 + j * ldc2] = 0;
                    }
                }
            }
        }

        if (m > mm + nn) {
            qblas_xgemm_kernel(m - mm - nn, nn, k, alphar, alphai,
                               a + (mm + nn) * k * 2, b + loop * k * 2,
                               c + (mm + nn) * 2 + loop * ldc2, ldc);
        }
    }
}


/* ── SYRK kernel: single-product diagonal-aware GEMM (complex) ───────
 *
 * Rank-K twin of qblas_xsyr2k_kernel_{u,l}: one product A·Bᵀ (with Bp packed
 * from the same A) instead of the two-product A·Bᵀ+B·Aᵀ. The strict-triangle
 * rectangular strips are plain GEMMs; the diagonal NR×NR block is GEMMed into
 * a zeroed subbuffer and the UPLO triangle merged SINGLY (no symmetric
 * mirror — a single product is already symmetric on the diagonal, so adding
 * subbuf[j,i] would double it). Single-pass, no `flag`. Faithful complex port
 * of the real qsyrk_kernel_{u,l} (OpenBLAS driver/level3/syrk_kernel.c). */
void qblas_xsyrk_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          TR alphar, TR alphai,
                          const TR *a, const TR *b,
                          TR *c, ptrdiff_t ldc, ptrdiff_t offset)
{
    TR subbuf[NR * (NR + 1) * 2];
    const ptrdiff_t ldc2 = 2 * ldc;

    if (m + offset < 0) {
        qblas_xgemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
        return;
    }
    if (n < offset) {
        return;
    }
    if (offset > 0) {
        b += offset * k * 2;
        c += offset * ldc2;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        qblas_xgemm_kernel(m, n - m - offset, k, alphar, alphai,
                           a, b + (m + offset) * k * 2,
                           c + (m + offset) * ldc2, ldc);
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        qblas_xgemm_kernel(-offset, n, k, alphar, alphai, a, b, c, ldc);
        a -= offset * k * 2;
        c -= offset * 2;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (loop > 0) {
            qblas_xgemm_kernel(loop, nn, k, alphar, alphai,
                               a, b + loop * k * 2,
                               c + loop * ldc2, ldc);
        }

        for (ptrdiff_t z = 0; z < nn * nn * 2; ++z) subbuf[z] = 0;
        qblas_xgemm_kernel(nn, nn, k, alphar, alphai,
                           a + loop * k * 2, b + loop * k * 2,
                           subbuf, nn);

        TR *cc = c + 2 * loop + loop * ldc2;
        for (ptrdiff_t j = 0; j < nn; ++j) {
            for (ptrdiff_t i = 0; i <= j; ++i) {
                cc[2*i + 0 + j * ldc2] += subbuf[(i + j * nn) * 2 + 0];
                cc[2*i + 1 + j * ldc2] += subbuf[(i + j * nn) * 2 + 1];
            }
        }
    }
}

void qblas_xsyrk_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          TR alphar, TR alphai,
                          const TR *a, const TR *b,
                          TR *c, ptrdiff_t ldc, ptrdiff_t offset)
{
    TR subbuf[NR * (NR + 1) * 2];
    const ptrdiff_t ldc2 = 2 * ldc;

    if (m + offset < 0) {
        return;
    }
    if (n < offset) {
        qblas_xgemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
        return;
    }
    if (offset > 0) {
        qblas_xgemm_kernel(m, offset, k, alphar, alphai, a, b, c, ldc);
        b += offset * k * 2;
        c += offset * ldc2;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        a -= offset * k * 2;
        c -= offset * 2;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }
    if (m > n - offset) {
        qblas_xgemm_kernel(m - n + offset, n, k, alphar, alphai,
                           a + (n - offset) * k * 2, b,
                           c + (n - offset) * 2, ldc);
        m = n + offset;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        for (ptrdiff_t z = 0; z < nn * nn * 2; ++z) subbuf[z] = 0;
        qblas_xgemm_kernel(nn, nn, k, alphar, alphai,
                           a + loop * k * 2, b + loop * k * 2,
                           subbuf, nn);

        TR *cc = c + 2 * loop + loop * ldc2;
        for (ptrdiff_t j = 0; j < nn; ++j) {
            for (ptrdiff_t i = j; i < nn; ++i) {
                cc[2*i + 0 + j * ldc2] += subbuf[(i + j * nn) * 2 + 0];
                cc[2*i + 1 + j * ldc2] += subbuf[(i + j * nn) * 2 + 1];
            }
        }

        if (m > mm + nn) {
            qblas_xgemm_kernel(m - mm - nn, nn, k, alphar, alphai,
                               a + (mm + nn) * k * 2, b + loop * k * 2,
                               c + (mm + nn) * 2 + loop * ldc2, ldc);
        }
    }
}


/* ── HERK kernel: single-product diagonal-aware GEMM (complex Hermitian) ──
 *
 * Rank-K twin of qblas_xher2k_kernel_{u,l}: one product A·Bᴴ (Bp packed from
 * the same A, conjugated by the caller's packer) instead of the two-product
 * her2k fold. Strict strips are plain GEMMs; the diagonal NR×NR block merges
 * SINGLY into the UPLO triangle — off-diagonal entries take subbuf[i,j]
 * directly, and the true diagonal element is realified (imag forced 0, the
 * Hermitian-C contract). Single-pass, no `flag`. Mirrors OpenBLAS
 * driver/level3/zherk_kernel.c. */
void qblas_xherk_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          TR alphar, TR alphai,
                          const TR *a, const TR *b,
                          TR *c, ptrdiff_t ldc, ptrdiff_t offset)
{
    TR subbuf[NR * (NR + 1) * 2];
    const ptrdiff_t ldc2 = 2 * ldc;

    if (m + offset < 0) {
        qblas_xgemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
        return;
    }
    if (n < offset) {
        return;
    }
    if (offset > 0) {
        b += offset * k * 2;
        c += offset * ldc2;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        qblas_xgemm_kernel(m, n - m - offset, k, alphar, alphai,
                           a, b + (m + offset) * k * 2,
                           c + (m + offset) * ldc2, ldc);
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        qblas_xgemm_kernel(-offset, n, k, alphar, alphai, a, b, c, ldc);
        a -= offset * k * 2;
        c -= offset * 2;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (loop > 0) {
            qblas_xgemm_kernel(loop, nn, k, alphar, alphai,
                               a, b + loop * k * 2,
                               c + loop * ldc2, ldc);
        }

        for (ptrdiff_t z = 0; z < nn * nn * 2; ++z) subbuf[z] = 0;
        qblas_xgemm_kernel(nn, nn, k, alphar, alphai,
                           a + loop * k * 2, b + loop * k * 2,
                           subbuf, nn);

        TR *cc = c + 2 * loop + loop * ldc2;
        for (ptrdiff_t j = 0; j < nn; ++j) {
            for (ptrdiff_t i = 0; i <= j; ++i) {
                cc[2*i + 0 + j * ldc2] += subbuf[(i + j * nn) * 2 + 0];
                if (i != j)
                    cc[2*i + 1 + j * ldc2] += subbuf[(i + j * nn) * 2 + 1];
                else
                    cc[2*i + 1 + j * ldc2] = 0;
            }
        }
    }
}

void qblas_xherk_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                          TR alphar, TR alphai,
                          const TR *a, const TR *b,
                          TR *c, ptrdiff_t ldc, ptrdiff_t offset)
{
    TR subbuf[NR * (NR + 1) * 2];
    const ptrdiff_t ldc2 = 2 * ldc;

    if (m + offset < 0) {
        return;
    }
    if (n < offset) {
        qblas_xgemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
        return;
    }
    if (offset > 0) {
        qblas_xgemm_kernel(m, offset, k, alphar, alphai, a, b, c, ldc);
        b += offset * k * 2;
        c += offset * ldc2;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        a -= offset * k * 2;
        c -= offset * 2;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }
    if (m > n - offset) {
        qblas_xgemm_kernel(m - n + offset, n, k, alphar, alphai,
                           a + (n - offset) * k * 2, b,
                           c + (n - offset) * 2, ldc);
        m = n + offset;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        for (ptrdiff_t z = 0; z < nn * nn * 2; ++z) subbuf[z] = 0;
        qblas_xgemm_kernel(nn, nn, k, alphar, alphai,
                           a + loop * k * 2, b + loop * k * 2,
                           subbuf, nn);

        TR *cc = c + 2 * loop + loop * ldc2;
        for (ptrdiff_t j = 0; j < nn; ++j) {
            for (ptrdiff_t i = j; i < nn; ++i) {
                cc[2*i + 0 + j * ldc2] += subbuf[(i + j * nn) * 2 + 0];
                if (i != j)
                    cc[2*i + 1 + j * ldc2] += subbuf[(i + j * nn) * 2 + 1];
                else
                    cc[2*i + 1 + j * ldc2] = 0;
            }
        }

        if (m > mm + nn) {
            qblas_xgemm_kernel(m - mm - nn, nn, k, alphar, alphai,
                               a + (mm + nn) * k * 2, b + loop * k * 2,
                               c + (mm + nn) * 2 + loop * ldc2, ldc);
        }
    }
}


/* ══════════════════════════════════════════════════════════════════
 * TRSM packed substrate — faithful complex-quad port of the ytrsm
 * pieces in src/epblas-openblas/common/qblas_l3_complex.c
 * (qblas_ytrsm_* -> qblas_xtrsm_*, qblas_ygemm_kernel -> qblas_xgemm_kernel).
 * compinv_ld + zsolve_{LN,LT,RN,RT} are the diagonal micro-solves;
 * the i{ln,lt,un,ut}copy pack the triangular A panel (conjugation
 * absorbed at pack time); qblas_xtrsm_kernel drives the blocked solve
 * against the already-packed panels, reusing qblas_xgemm_kernel for
 * the trailing GEMM update. Interleaved-complex storage: lda*=2, two
 * __float128 per element. Used by the SIDE=L/R blocked xtrsm driver. */
typedef __float128 T;

static inline void compinv_ld(T *b, T ar, T ai, int unit) {
    if (unit) {
        b[0] = 1.0Q;
        b[1] = 0.0Q;
        return;
    }
    T ratio, den;
    if (__builtin_fabsf128(ar) >= __builtin_fabsf128(ai)) {
        ratio = ai / ar;
        den   = 1.0Q / (ar * (1.0Q + ratio * ratio));
        b[0] =  den;
        b[1] = -ratio * den;
    } else {
        ratio = ar / ai;
        den   = 1.0Q / (ai * (1.0Q + ratio * ratio));
        b[0] =  ratio * den;
        b[1] = -den;
    }
}


void qblas_xtrsm_ilncopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t offset, T *b, int unit, int conj)
{
    ptrdiff_t i, ii, j, jj;
    T data01 = 0.0Q, data02 = 0.0Q, data03, data04;
    T data05, data06, data07 = 0.0Q, data08 = 0.0Q;
    const T *a1, *a2;

    lda *= 2;
    jj = offset;

    j = (n >> 1);
    while (j > 0) {
        a1 = a + 0 * lda;
        a2 = a + 1 * lda;

        i = (m >> 1);
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                data03 = *(a1 + 2);
                data04 = *(a1 + 3);
                if (!unit) {
                    data07 = *(a2 + 2);
                    data08 = *(a2 + 3);
                }
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
                b[4] = data03;
                b[5] = conj ? -data04 : data04;
                compinv_ld(b + 6, data07, conj ? -data08 : data08, unit);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a1 + 2);
                data04 = *(a1 + 3);
                data05 = *(a2 + 0);
                data06 = *(a2 + 1);
                data07 = *(a2 + 2);
                data08 = *(a2 + 3);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
                b[2] = data05;
                b[3] = conj ? -data06 : data06;
                b[4] = data03;
                b[5] = conj ? -data04 : data04;
                b[6] = data07;
                b[7] = conj ? -data08 : data08;
            }
            a1 += 4;
            a2 += 4;
            b += 8;
            i--;
            ii += 2;
        }

        if (m & 1) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data05 = *(a2 + 0);
                data06 = *(a2 + 1);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
                b[2] = data05;
                b[3] = conj ? -data06 : data06;
            }
            b += 4;
        }

        a += 2 * lda;
        jj += 2;
        j--;
    }

    if (n & 1) {
        a1 = a + 0 * lda;
        i = m;
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
            }
            a1 += 2;
            b += 2;
            i--;
            ii += 1;
        }
    }
}


void qblas_xtrsm_iltcopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t offset, T *b, int unit, int conj)
{
    ptrdiff_t i, ii, j, jj;
    T data01 = 0.0Q, data02 = 0.0Q, data03, data04;
    T data05, data06, data07 = 0.0Q, data08 = 0.0Q;
    const T *a1, *a2;

    lda *= 2;
    jj = offset;

    j = (n >> 1);
    while (j > 0) {
        a1 = a + 0 * lda;
        a2 = a + 1 * lda;

        i = (m >> 1);
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                data03 = *(a1 + 2);
                data04 = *(a1 + 3);
                if (!unit) {
                    data07 = *(a2 + 2);
                    data08 = *(a2 + 3);
                }
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
                b[2] = data03;
                b[3] = conj ? -data04 : data04;
                compinv_ld(b + 6, data07, conj ? -data08 : data08, unit);
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a1 + 2);
                data04 = *(a1 + 3);
                data05 = *(a2 + 0);
                data06 = *(a2 + 1);
                data07 = *(a2 + 2);
                data08 = *(a2 + 3);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
                b[2] = data03;
                b[3] = conj ? -data04 : data04;
                b[4] = data05;
                b[5] = conj ? -data06 : data06;
                b[6] = data07;
                b[7] = conj ? -data08 : data08;
            }
            a1 += 2 * lda;
            a2 += 2 * lda;
            b += 8;
            i--;
            ii += 2;
        }

        if (m & 1) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                data03 = *(a1 + 2);
                data04 = *(a1 + 3);
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
                b[2] = data03;
                b[3] = conj ? -data04 : data04;
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a1 + 2);
                data04 = *(a1 + 3);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
                b[2] = data03;
                b[3] = conj ? -data04 : data04;
            }
            b += 4;
        }

        a += 4;
        jj += 2;
        j--;
    }

    if (n & 1) {
        a1 = a + 0 * lda;
        i = m;
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
            }
            a1 += lda;
            b += 2;
            i--;
            ii += 1;
        }
    }
}


void qblas_xtrsm_iuncopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t offset, T *b, int unit, int conj)
{
    ptrdiff_t i, ii, j, jj;
    T data01 = 0.0Q, data02 = 0.0Q, data03, data04;
    T data05, data06, data07 = 0.0Q, data08 = 0.0Q;
    const T *a1, *a2;

    lda *= 2;
    jj = offset;

    j = (n >> 1);
    while (j > 0) {
        a1 = a + 0 * lda;
        a2 = a + 1 * lda;

        i = (m >> 1);
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                data05 = *(a2 + 0);
                data06 = *(a2 + 1);
                if (!unit) {
                    data07 = *(a2 + 2);
                    data08 = *(a2 + 3);
                }
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
                b[2] = data05;
                b[3] = conj ? -data06 : data06;
                compinv_ld(b + 6, data07, conj ? -data08 : data08, unit);
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a1 + 2);
                data04 = *(a1 + 3);
                data05 = *(a2 + 0);
                data06 = *(a2 + 1);
                data07 = *(a2 + 2);
                data08 = *(a2 + 3);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
                b[2] = data05;
                b[3] = conj ? -data06 : data06;
                b[4] = data03;
                b[5] = conj ? -data04 : data04;
                b[6] = data07;
                b[7] = conj ? -data08 : data08;
            }
            a1 += 4;
            a2 += 4;
            b += 8;
            i--;
            ii += 2;
        }

        if (m & 1) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                data05 = *(a2 + 0);
                data06 = *(a2 + 1);
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
                b[2] = data05;
                b[3] = conj ? -data06 : data06;
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a2 + 0);
                data04 = *(a2 + 1);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
                b[2] = data03;
                b[3] = conj ? -data04 : data04;
            }
            b += 4;
        }

        a += 2 * lda;
        jj += 2;
        j--;
    }

    if (n & 1) {
        a1 = a + 0 * lda;
        i = m;
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
            }
            if (ii < jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
            }
            a1 += 2;
            b += 2;
            i--;
            ii += 1;
        }
    }
}


void qblas_xtrsm_iutcopy(ptrdiff_t m, ptrdiff_t n,
                         const T *a, ptrdiff_t lda,
                         ptrdiff_t offset, T *b, int unit, int conj)
{
    ptrdiff_t i, ii, j, jj;
    T data01 = 0.0Q, data02 = 0.0Q, data03, data04;
    T data05, data06, data07 = 0.0Q, data08 = 0.0Q;
    const T *a1, *a2;

    lda *= 2;
    jj = offset;

    j = (n >> 1);
    while (j > 0) {
        a1 = a + 0 * lda;
        a2 = a + 1 * lda;

        i = (m >> 1);
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                data05 = *(a2 + 0);
                data06 = *(a2 + 1);
                if (!unit) {
                    data07 = *(a2 + 2);
                    data08 = *(a2 + 3);
                }
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
                b[4] = data05;
                b[5] = conj ? -data06 : data06;
                compinv_ld(b + 6, data07, conj ? -data08 : data08, unit);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a1 + 2);
                data04 = *(a1 + 3);
                data05 = *(a2 + 0);
                data06 = *(a2 + 1);
                data07 = *(a2 + 2);
                data08 = *(a2 + 3);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
                b[2] = data03;
                b[3] = conj ? -data04 : data04;
                b[4] = data05;
                b[5] = conj ? -data06 : data06;
                b[6] = data07;
                b[7] = conj ? -data08 : data08;
            }
            a1 += 2 * lda;
            a2 += 2 * lda;
            b += 8;
            i--;
            ii += 2;
        }

        if (m & 1) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                data03 = *(a1 + 2);
                data04 = *(a1 + 3);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
                b[2] = data03;
                b[3] = conj ? -data04 : data04;
            }
            b += 4;
        }

        a += 4;
        jj += 2;
        j--;
    }

    if (n & 1) {
        a1 = a + 0 * lda;
        i = m;
        ii = 0;
        while (i > 0) {
            if (ii == jj) {
                if (!unit) {
                    data01 = *(a1 + 0);
                    data02 = *(a1 + 1);
                }
                compinv_ld(b + 0, data01, conj ? -data02 : data02, unit);
            }
            if (ii > jj) {
                data01 = *(a1 + 0);
                data02 = *(a1 + 1);
                b[0] = data01;
                b[1] = conj ? -data02 : data02;
            }
            a1 += lda;
            b += 2;
            i--;
            ii += 1;
        }
    }
}


/* ── TRSM diagonal-aware microkernel (complex) ───────────────────────
 *
 * Faithful port of OpenBLAS kernel/generic/trsm_kernel_{LN,LT,RN,RT}.c
 * (Z-variant, unconjugated branch — `conj` is absorbed at pack time so
 * the kernel always runs the NN form of the per-element complex
 * multiply).
 *
 * solve() in the complex variant uses 2 __float128s per element
 * (re,im); each "scalar" multiplication becomes a 2x2 outer-product
 * `(aa1, aa2) * (bb1, bb2) = (aa1*bb1 - aa2*bb2, aa1*bb2 + aa2*bb1)`.
 */

static inline void zsolve_LN(ptrdiff_t m, ptrdiff_t n,
                             T *a, T *b, T *c, ptrdiff_t ldc)
{
    T aa1, aa2, bb1, bb2, cc1, cc2;
    ptrdiff_t i, j, k;
    ldc *= 2;
    a += (m - 1) * m * 2;
    b += (m - 1) * n * 2;
    for (i = m - 1; i >= 0; i--) {
        aa1 = *(a + i * 2 + 0);
        aa2 = *(a + i * 2 + 1);
        for (j = 0; j < n; j++) {
            bb1 = *(c + i * 2 + 0 + j * ldc);
            bb2 = *(c + i * 2 + 1 + j * ldc);
            cc1 = aa1 * bb1 - aa2 * bb2;
            cc2 = aa1 * bb2 + aa2 * bb1;
            *(b + 0) = cc1;
            *(b + 1) = cc2;
            *(c + i * 2 + 0 + j * ldc) = cc1;
            *(c + i * 2 + 1 + j * ldc) = cc2;
            b += 2;
            for (k = 0; k < i; k++) {
                *(c + k * 2 + 0 + j * ldc) -= cc1 * *(a + k * 2 + 0) - cc2 * *(a + k * 2 + 1);
                *(c + k * 2 + 1 + j * ldc) -= cc1 * *(a + k * 2 + 1) + cc2 * *(a + k * 2 + 0);
            }
        }
        a -= m * 2;
        b -= 4 * n;
    }
}

static inline void zsolve_LT(ptrdiff_t m, ptrdiff_t n,
                             T *a, T *b, T *c, ptrdiff_t ldc)
{
    T aa1, aa2, bb1, bb2, cc1, cc2;
    ptrdiff_t i, j, k;
    ldc *= 2;
    for (i = 0; i < m; i++) {
        aa1 = *(a + i * 2 + 0);
        aa2 = *(a + i * 2 + 1);
        for (j = 0; j < n; j++) {
            bb1 = *(c + i * 2 + 0 + j * ldc);
            bb2 = *(c + i * 2 + 1 + j * ldc);
            cc1 = aa1 * bb1 - aa2 * bb2;
            cc2 = aa1 * bb2 + aa2 * bb1;
            *(b + 0) = cc1;
            *(b + 1) = cc2;
            *(c + i * 2 + 0 + j * ldc) = cc1;
            *(c + i * 2 + 1 + j * ldc) = cc2;
            b += 2;
            for (k = i + 1; k < m; k++) {
                *(c + k * 2 + 0 + j * ldc) -= cc1 * *(a + k * 2 + 0) - cc2 * *(a + k * 2 + 1);
                *(c + k * 2 + 1 + j * ldc) -= cc1 * *(a + k * 2 + 1) + cc2 * *(a + k * 2 + 0);
            }
        }
        a += m * 2;
    }
}

static inline void zsolve_RN(ptrdiff_t m, ptrdiff_t n,
                             T *a, T *b, T *c, ptrdiff_t ldc)
{
    T aa1, aa2, bb1, bb2, cc1, cc2;
    ptrdiff_t i, j, k;
    ldc *= 2;
    for (i = 0; i < n; i++) {
        bb1 = *(b + i * 2 + 0);
        bb2 = *(b + i * 2 + 1);
        for (j = 0; j < m; j++) {
            aa1 = *(c + j * 2 + 0 + i * ldc);
            aa2 = *(c + j * 2 + 1 + i * ldc);
            cc1 = aa1 * bb1 - aa2 * bb2;
            cc2 = aa1 * bb2 + aa2 * bb1;
            *(a + 0) = cc1;
            *(a + 1) = cc2;
            *(c + j * 2 + 0 + i * ldc) = cc1;
            *(c + j * 2 + 1 + i * ldc) = cc2;
            a += 2;
            for (k = i + 1; k < n; k++) {
                *(c + j * 2 + 0 + k * ldc) -= cc1 * *(b + k * 2 + 0) - cc2 * *(b + k * 2 + 1);
                *(c + j * 2 + 1 + k * ldc) -= cc1 * *(b + k * 2 + 1) + cc2 * *(b + k * 2 + 0);
            }
        }
        b += n * 2;
    }
}

static inline void zsolve_RT(ptrdiff_t m, ptrdiff_t n,
                             T *a, T *b, T *c, ptrdiff_t ldc)
{
    T aa1, aa2, bb1, bb2, cc1, cc2;
    ptrdiff_t i, j, k;
    ldc *= 2;
    a += (n - 1) * m * 2;
    b += (n - 1) * n * 2;
    for (i = n - 1; i >= 0; i--) {
        bb1 = *(b + i * 2 + 0);
        bb2 = *(b + i * 2 + 1);
        for (j = 0; j < m; j++) {
            aa1 = *(c + j * 2 + 0 + i * ldc);
            aa2 = *(c + j * 2 + 1 + i * ldc);
            cc1 = aa1 * bb1 - aa2 * bb2;
            cc2 = aa1 * bb2 + aa2 * bb1;
            *(a + 0) = cc1;
            *(a + 1) = cc2;
            *(c + j * 2 + 0 + i * ldc) = cc1;
            *(c + j * 2 + 1 + i * ldc) = cc2;
            a += 2;
            for (k = 0; k < i; k++) {
                *(c + j * 2 + 0 + k * ldc) -= cc1 * *(b + k * 2 + 0) - cc2 * *(b + k * 2 + 1);
                *(c + j * 2 + 1 + k * ldc) -= cc1 * *(b + k * 2 + 1) + cc2 * *(b + k * 2 + 0);
            }
        }
        b -= n * 2;
        a -= 4 * m;
    }
}


void qblas_xtrsm_kernel(int left, int trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        const T *ba, const T *bb,
                        T *C, ptrdiff_t ldc,
                        ptrdiff_t offset)
{
    const T dm1r = -1.0Q;
    const T dm1i = 0.0Q;
    const ptrdiff_t UR = MR;
    const ptrdiff_t UN = NR;

    T *a_buf = (T *)ba;
    T *b_buf = (T *)bb;

    if (left && !trans) {
        ptrdiff_t i, j;
        T *aa, *cc;
        ptrdiff_t kk;

        j = (bn / UN);
        while (j > 0) {
            kk = bm + offset;

            if (bm & (UR - 1)) {
                for (i = 1; i < UR; i *= 2) {
                    if (bm & i) {
                        aa = a_buf + ((bm & ~(i - 1)) - i) * bk * 2;
                        cc = C + ((bm & ~(i - 1)) - i) * 2;
                        if (bk - kk > 0) {
                            qblas_xgemm_kernel(i, UN, bk - kk, dm1r, dm1i,
                                               aa + i * kk * 2,
                                               b_buf + UN * kk * 2, cc, ldc);
                        }
                        zsolve_LN(i, UN,
                                  aa + (kk - i) * i * 2,
                                  b_buf + (kk - i) * UN * 2,
                                  cc, ldc);
                        kk -= i;
                    }
                }
            }

            i = (bm / UR);
            if (i > 0) {
                aa = a_buf + ((bm & ~(UR - 1)) - UR) * bk * 2;
                cc = C + ((bm & ~(UR - 1)) - UR) * 2;
                do {
                    if (bk - kk > 0) {
                        qblas_xgemm_kernel(UR, UN, bk - kk, dm1r, dm1i,
                                           aa + UR * kk * 2,
                                           b_buf + UN * kk * 2, cc, ldc);
                    }
                    zsolve_LN(UR, UN,
                              aa + (kk - UR) * UR * 2,
                              b_buf + (kk - UR) * UN * 2,
                              cc, ldc);
                    aa -= UR * bk * 2;
                    cc -= UR * 2;
                    kk -= UR;
                    i--;
                } while (i > 0);
            }

            b_buf += UN * bk * 2;
            C += UN * ldc * 2;
            j--;
        }

        if (bn & (UN - 1)) {
            j = (UN >> 1);
            while (j > 0) {
                if (bn & j) {
                    kk = bm + offset;
                    if (bm & (UR - 1)) {
                        for (i = 1; i < UR; i *= 2) {
                            if (bm & i) {
                                aa = a_buf + ((bm & ~(i - 1)) - i) * bk * 2;
                                cc = C + ((bm & ~(i - 1)) - i) * 2;
                                if (bk - kk > 0) {
                                    qblas_xgemm_kernel(i, j, bk - kk, dm1r, dm1i,
                                                       aa + i * kk * 2,
                                                       b_buf + j * kk * 2, cc, ldc);
                                }
                                zsolve_LN(i, j,
                                          aa + (kk - i) * i * 2,
                                          b_buf + (kk - i) * j * 2,
                                          cc, ldc);
                                kk -= i;
                            }
                        }
                    }
                    i = (bm / UR);
                    if (i > 0) {
                        aa = a_buf + ((bm & ~(UR - 1)) - UR) * bk * 2;
                        cc = C + ((bm & ~(UR - 1)) - UR) * 2;
                        do {
                            if (bk - kk > 0) {
                                qblas_xgemm_kernel(UR, j, bk - kk, dm1r, dm1i,
                                                   aa + UR * kk * 2,
                                                   b_buf + j * kk * 2, cc, ldc);
                            }
                            zsolve_LN(UR, j,
                                      aa + (kk - UR) * UR * 2,
                                      b_buf + (kk - UR) * j * 2,
                                      cc, ldc);
                            aa -= UR * bk * 2;
                            cc -= UR * 2;
                            kk -= UR;
                            i--;
                        } while (i > 0);
                    }
                    b_buf += j * bk * 2;
                    C += j * ldc * 2;
                }
                j >>= 1;
            }
        }
    } else if (left && trans) {
        ptrdiff_t i, j;
        T *aa, *cc;
        ptrdiff_t kk;

        j = (bn / UN);
        while (j > 0) {
            kk = offset;
            aa = a_buf;
            cc = C;
            i = (bm / UR);
            while (i > 0) {
                if (kk > 0) {
                    qblas_xgemm_kernel(UR, UN, kk, dm1r, dm1i, aa, b_buf, cc, ldc);
                }
                zsolve_LT(UR, UN,
                          aa + kk * UR * 2,
                          b_buf + kk * UN * 2,
                          cc, ldc);
                aa += UR * bk * 2;
                cc += UR * 2;
                kk += UR;
                i--;
            }
            if (bm & (UR - 1)) {
                i = (UR >> 1);
                while (i > 0) {
                    if (bm & i) {
                        if (kk > 0) {
                            qblas_xgemm_kernel(i, UN, kk, dm1r, dm1i, aa, b_buf, cc, ldc);
                        }
                        zsolve_LT(i, UN,
                                  aa + kk * i * 2,
                                  b_buf + kk * UN * 2,
                                  cc, ldc);
                        aa += i * bk * 2;
                        cc += i * 2;
                        kk += i;
                    }
                    i >>= 1;
                }
            }
            b_buf += UN * bk * 2;
            C += UN * ldc * 2;
            j--;
        }

        if (bn & (UN - 1)) {
            j = (UN >> 1);
            while (j > 0) {
                if (bn & j) {
                    kk = offset;
                    aa = a_buf;
                    cc = C;
                    i = (bm / UR);
                    while (i > 0) {
                        if (kk > 0) {
                            qblas_xgemm_kernel(UR, j, kk, dm1r, dm1i, aa, b_buf, cc, ldc);
                        }
                        zsolve_LT(UR, j,
                                  aa + kk * UR * 2,
                                  b_buf + kk * j * 2,
                                  cc, ldc);
                        aa += UR * bk * 2;
                        cc += UR * 2;
                        kk += UR;
                        i--;
                    }
                    if (bm & (UR - 1)) {
                        i = (UR >> 1);
                        while (i > 0) {
                            if (bm & i) {
                                if (kk > 0) {
                                    qblas_xgemm_kernel(i, j, kk, dm1r, dm1i, aa, b_buf, cc, ldc);
                                }
                                zsolve_LT(i, j,
                                          aa + kk * i * 2,
                                          b_buf + kk * j * 2,
                                          cc, ldc);
                                aa += i * bk * 2;
                                cc += i * 2;
                                kk += i;
                            }
                            i >>= 1;
                        }
                    }
                    b_buf += j * bk * 2;
                    C += j * ldc * 2;
                }
                j >>= 1;
            }
        }
    } else if (!left && !trans) {
        ptrdiff_t i, j;
        T *aa, *cc;
        ptrdiff_t kk = -offset;

        j = (bn / UN);
        while (j > 0) {
            aa = a_buf;
            cc = C;
            i = (bm / UR);
            if (i > 0) {
                do {
                    if (kk > 0) {
                        qblas_xgemm_kernel(UR, UN, kk, dm1r, dm1i, aa, b_buf, cc, ldc);
                    }
                    zsolve_RN(UR, UN,
                              aa + kk * UR * 2,
                              b_buf + kk * UN * 2,
                              cc, ldc);
                    aa += UR * bk * 2;
                    cc += UR * 2;
                    i--;
                } while (i > 0);
            }
            if (bm & (UR - 1)) {
                i = (UR >> 1);
                while (i > 0) {
                    if (bm & i) {
                        if (kk > 0) {
                            qblas_xgemm_kernel(i, UN, kk, dm1r, dm1i, aa, b_buf, cc, ldc);
                        }
                        zsolve_RN(i, UN,
                                  aa + kk * i * 2,
                                  b_buf + kk * UN * 2,
                                  cc, ldc);
                        aa += i * bk * 2;
                        cc += i * 2;
                    }
                    i >>= 1;
                }
            }
            kk += UN;
            b_buf += UN * bk * 2;
            C += UN * ldc * 2;
            j--;
        }

        if (bn & (UN - 1)) {
            j = (UN >> 1);
            while (j > 0) {
                if (bn & j) {
                    aa = a_buf;
                    cc = C;
                    i = (bm / UR);
                    while (i > 0) {
                        if (kk > 0) {
                            qblas_xgemm_kernel(UR, j, kk, dm1r, dm1i, aa, b_buf, cc, ldc);
                        }
                        zsolve_RN(UR, j,
                                  aa + kk * UR * 2,
                                  b_buf + kk * j * 2,
                                  cc, ldc);
                        aa += UR * bk * 2;
                        cc += UR * 2;
                        i--;
                    }
                    if (bm & (UR - 1)) {
                        i = (UR >> 1);
                        while (i > 0) {
                            if (bm & i) {
                                if (kk > 0) {
                                    qblas_xgemm_kernel(i, j, kk, dm1r, dm1i, aa, b_buf, cc, ldc);
                                }
                                zsolve_RN(i, j,
                                          aa + kk * i * 2,
                                          b_buf + kk * j * 2,
                                          cc, ldc);
                                aa += i * bk * 2;
                                cc += i * 2;
                            }
                            i >>= 1;
                        }
                    }
                    b_buf += j * bk * 2;
                    C += j * ldc * 2;
                    kk += j;
                }
                j >>= 1;
            }
        }
    } else {
        ptrdiff_t i, j;
        T *aa, *cc;
        ptrdiff_t kk = bn - offset;
        C += bn * ldc * 2;
        b_buf += bn * bk * 2;

        if (bn & (UN - 1)) {
            j = 1;
            while (j < UN) {
                if (bn & j) {
                    aa = a_buf;
                    b_buf -= j * bk * 2;
                    C -= j * ldc * 2;
                    cc = C;
                    i = (bm / UR);
                    if (i > 0) {
                        do {
                            if (bk - kk > 0) {
                                qblas_xgemm_kernel(UR, j, bk - kk, dm1r, dm1i,
                                                   aa + UR * kk * 2,
                                                   b_buf + j * kk * 2, cc, ldc);
                            }
                            zsolve_RT(UR, j,
                                      aa + (kk - j) * UR * 2,
                                      b_buf + (kk - j) * j * 2,
                                      cc, ldc);
                            aa += UR * bk * 2;
                            cc += UR * 2;
                            i--;
                        } while (i > 0);
                    }
                    if (bm & (UR - 1)) {
                        i = (UR >> 1);
                        do {
                            if (bm & i) {
                                if (bk - kk > 0) {
                                    qblas_xgemm_kernel(i, j, bk - kk, dm1r, dm1i,
                                                       aa + i * kk * 2,
                                                       b_buf + j * kk * 2, cc, ldc);
                                }
                                zsolve_RT(i, j,
                                          aa + (kk - j) * i * 2,
                                          b_buf + (kk - j) * j * 2,
                                          cc, ldc);
                                aa += i * bk * 2;
                                cc += i * 2;
                            }
                            i >>= 1;
                        } while (i > 0);
                    }
                    kk -= j;
                }
                j <<= 1;
            }
        }

        j = (bn / UN);
        if (j > 0) {
            do {
                aa = a_buf;
                b_buf -= UN * bk * 2;
                C -= UN * ldc * 2;
                cc = C;
                i = (bm / UR);
                if (i > 0) {
                    do {
                        if (bk - kk > 0) {
                            qblas_xgemm_kernel(UR, UN, bk - kk, dm1r, dm1i,
                                               aa + UR * kk * 2,
                                               b_buf + UN * kk * 2, cc, ldc);
                        }
                        zsolve_RT(UR, UN,
                                  aa + (kk - UN) * UR * 2,
                                  b_buf + (kk - UN) * UN * 2,
                                  cc, ldc);
                        aa += UR * bk * 2;
                        cc += UR * 2;
                        i--;
                    } while (i > 0);
                }
                if (bm & (UR - 1)) {
                    i = (UR >> 1);
                    do {
                        if (bm & i) {
                            if (bk - kk > 0) {
                                qblas_xgemm_kernel(i, UN, bk - kk, dm1r, dm1i,
                                                   aa + i * kk * 2,
                                                   b_buf + UN * kk * 2, cc, ldc);
                            }
                            zsolve_RT(i, UN,
                                      aa + (kk - UN) * i * 2,
                                      b_buf + (kk - UN) * UN * 2,
                                      cc, ldc);
                            aa += i * bk * 2;
                            cc += i * 2;
                        }
                        i >>= 1;
                    } while (i > 0);
                }
                kk -= UN;
                j--;
            } while (j > 0);
        }
    }
}
