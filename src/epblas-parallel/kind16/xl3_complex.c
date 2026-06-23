/*
 * xl3_complex.c — shared L3 microkernel + packers (kind16 complex), par
 * overlay copy. A faithful transcription of the epblas-openblas substrate
 * src/epblas-openblas/common/qblas_l3_complex.c: the par overlay keeps its
 * own copy (mirrors the kind10 etri_kernel.c convention) so the parallel
 * x* drivers below it never reach across into the openblas tree.
 *
 * Port source: OpenBLAS.
 *   - kernel/generic/zgemmkernel_2x2.c   ← qblas_ygemm_kernel
 *                                          (NN path only; conjugation absorbed
 *                                           into the packers via the `conj` flag)
 *   - kernel/generic/zgemm_ncopy_2.c     ← qblas_ygemm_ncopy
 *   - kernel/generic/zgemm_tcopy_2.c     ← qblas_ygemm_tcopy
 *   - kernel/generic/zgemm_beta.c        ← qblas_ygemm_beta
 */

#include "xl3_complex.h"
#include <quadmath.h>
#include <stdlib.h>

typedef __float128 TR;

#define MR QBLAS_YGEMM_MR
#define NR QBLAS_YGEMM_NR


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
void qblas_ygemm_kernel(ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
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
void qblas_ygemm_ncopy(ptrdiff_t m, ptrdiff_t n,
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
void qblas_ygemm_tcopy(ptrdiff_t m, ptrdiff_t n,
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
void qblas_ygemm_beta(ptrdiff_t m, ptrdiff_t n,
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
void qblas_ygemm_blocks(ptrdiff_t *mc, ptrdiff_t *kc, ptrdiff_t *nc) {
    *mc = QBLAS_YGEMM_GEMM_P; *kc = QBLAS_YGEMM_GEMM_Q; *nc = QBLAS_YGEMM_GEMM_R;
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
void qblas_ysymm_ucopy(ptrdiff_t m, ptrdiff_t n,
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


void qblas_ysymm_lcopy(ptrdiff_t m, ptrdiff_t n,
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
void qblas_yhemm_ucopy(ptrdiff_t m, ptrdiff_t n,
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


void qblas_yhemm_lcopy(ptrdiff_t m, ptrdiff_t n,
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
 * See the header comment on qblas_yhemm_ucopy_oc for the rationale
 * (why we need a distinct OC variant instead of reusing the IC one as
 * upstream OpenBLAS does).
 */
void qblas_yhemm_ucopy_oc(ptrdiff_t m, ptrdiff_t n,
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


void qblas_yhemm_lcopy_oc(ptrdiff_t m, ptrdiff_t n,
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
 * inside the row loop instead of touching `qblas_ygemm_beta`'s full-
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

void qblas_ysyrk_beta_u(ptrdiff_t n, TR br, TR bi, TR *c, ptrdiff_t ldc) {
    if (br == 1.0Q && bi == 0.0Q) return;
    const ptrdiff_t ldc2 = 2 * ldc;
    for (ptrdiff_t j = 0; j < n; ++j) {
        xsyrk_scale_strip(c + j * ldc2, 0, j + 1, br, bi);
    }
}

void qblas_ysyrk_beta_l(ptrdiff_t n, TR br, TR bi, TR *c, ptrdiff_t ldc) {
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
void qblas_yherk_beta_u(ptrdiff_t n, TR br, TR *c, ptrdiff_t ldc) {
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

void qblas_yherk_beta_l(ptrdiff_t n, TR br, TR *c, ptrdiff_t ldc) {
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
void qblas_ysyr2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           TR alphar, TR alphai,
                           const TR *a, const TR *b,
                           TR *c, ptrdiff_t ldc, ptrdiff_t offset, bool flag)
{
    TR subbuf[NR * (NR + 1) * 2];
    const ptrdiff_t ldc2 = 2 * ldc;

    if (m + offset < 0) {
        qblas_ygemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
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
        qblas_ygemm_kernel(m, n - m - offset, k, alphar, alphai,
                           a, b + (m + offset) * k * 2,
                           c + (m + offset) * ldc2, ldc);
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        qblas_ygemm_kernel(-offset, n, k, alphar, alphai, a, b, c, ldc);
        a -= offset * k * 2;
        c -= offset * 2;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (loop > 0) {
            qblas_ygemm_kernel(loop, nn, k, alphar, alphai,
                               a, b + loop * k * 2,
                               c + loop * ldc2, ldc);
        }

        if (flag) {
            for (ptrdiff_t z = 0; z < nn * nn * 2; ++z) subbuf[z] = 0;
            qblas_ygemm_kernel(nn, nn, k, alphar, alphai,
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

void qblas_ysyr2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
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
        qblas_ygemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
        return;
    }
    if (offset > 0) {
        qblas_ygemm_kernel(m, offset, k, alphar, alphai, a, b, c, ldc);
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
        qblas_ygemm_kernel(m - n + offset, n, k, alphar, alphai,
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
            qblas_ygemm_kernel(nn, nn, k, alphar, alphai,
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
            qblas_ygemm_kernel(m - mm - nn, nn, k, alphar, alphai,
                               a + (mm + nn) * k * 2, b + loop * k * 2,
                               c + (mm + nn) * 2 + loop * ldc2, ldc);
        }
    }
}


/* ── HER2K kernel: two-pass diagonal-aware GEMM (complex Hermitian) ──
 *
 * Faithful port of OpenBLAS driver/level3/zher2k_kernel.c. Structural
 * twin of qblas_ysyr2k_kernel_{u,l}; differs only in the diagonal
 * NR×NR subblock writeback:
 *
 *   imag part subtracts subbuf[j,i] (instead of adding) and the
 *   actual diagonal element is forced imag = 0 (Hermitian C contract).
 *
 * See header for the two-pass calling convention. Conjugation is
 * absorbed by the caller's packers per upstream's GEMM_KERNEL_R/L pick.
 */
void qblas_yher2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
                           TR alphar, TR alphai,
                           const TR *a, const TR *b,
                           TR *c, ptrdiff_t ldc, ptrdiff_t offset, bool flag)
{
    TR subbuf[NR * (NR + 1) * 2];
    const ptrdiff_t ldc2 = 2 * ldc;

    if (m + offset < 0) {
        qblas_ygemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
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
        qblas_ygemm_kernel(m, n - m - offset, k, alphar, alphai,
                           a, b + (m + offset) * k * 2,
                           c + (m + offset) * ldc2, ldc);
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        qblas_ygemm_kernel(-offset, n, k, alphar, alphai, a, b, c, ldc);
        a -= offset * k * 2;
        c -= offset * 2;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (loop > 0) {
            qblas_ygemm_kernel(loop, nn, k, alphar, alphai,
                               a, b + loop * k * 2,
                               c + loop * ldc2, ldc);
        }

        if (flag) {
            for (ptrdiff_t z = 0; z < nn * nn * 2; ++z) subbuf[z] = 0;
            qblas_ygemm_kernel(nn, nn, k, alphar, alphai,
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

void qblas_yher2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
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
        qblas_ygemm_kernel(m, n, k, alphar, alphai, a, b, c, ldc);
        return;
    }
    if (offset > 0) {
        qblas_ygemm_kernel(m, offset, k, alphar, alphai, a, b, c, ldc);
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
        qblas_ygemm_kernel(m - n + offset, n, k, alphar, alphai,
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
            qblas_ygemm_kernel(nn, nn, k, alphar, alphai,
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
            qblas_ygemm_kernel(m - mm - nn, nn, k, alphar, alphai,
                               a + (mm + nn) * k * 2, b + loop * k * 2,
                               c + (mm + nn) * 2 + loop * ldc2, ldc);
        }
    }
}
