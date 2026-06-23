/*
 * etrmm_kernel — kind10 (REAL(KIND=10) / long double) diagonal-aware TRMM
 * micro-kernel. Faithful port of OpenBLAS kernel/generic/trmmkernel_2x2.c
 * with compile-time LEFT and TRANSA macros converted to runtime `left` and
 * `trans` flags. The 4 (left, trans) combinations correspond to OpenBLAS's
 * TRMM_KERNEL_{LN,LT,RN,RT}; collapsing to one function with runtime
 * branches is branch-predictor friendly since left/trans are loop-invariant
 * per call. C := alpha · ba · bb (overwrite — matches OpenBLAS's
 * GEMM_KERNEL(beta=0) convention inside the TRMM driver).
 *
 * Consumes the same ob contiguous-odd-tail packed layout as etri_kernel.c
 * (the off-diagonal sub-tiles run through etri_gemm_kernel/etri_kernel_store);
 * the diagonal block is packed by the etrmm_i*copy packers (etrmm_pack.c).
 *
 * Local notation matches the source: `off`, `temp`, `res0..res3`,
 * `load0..load7`; the pointer-arithmetic formulas are gated on (left, trans)
 * with the same boolean structure as the OpenBLAS preprocessor logic.
 */

#include <stddef.h>

#include "etrmm_kernel.h"

typedef etrmm_TR TR;

void etrmm_kernel(bool left, bool trans,
                        ptrdiff_t bm, ptrdiff_t bn, ptrdiff_t bk,
                        TR alpha,
                        const TR *ba, const TR *bb,
                        TR *C, ptrdiff_t ldc,
                        ptrdiff_t offset)
{
    ptrdiff_t i, j, k;
    TR *C0, *C1;
    const TR *ptrba, *ptrbb;
    TR res0, res1, res2, res3;
    TR load0, load1, load2, load3, load4, load5, load6, load7;
    ptrdiff_t off, temp;

    /* if defined(TRMMKERNEL) && !defined(LEFT) → off = -offset, else 0 */
    if (!left) off = -offset;
    else       off = 0;

    for (j = 0; j < bn / 2; j += 1) {
        C0 = C;
        C1 = C0 + ldc;

        if (left) off = offset;

        ptrba = ba;
        for (i = 0; i < bm / 2; i += 1) {
            /* if (LEFT && TRANSA) || (!LEFT && !TRANSA): ptrbb = bb;
               else                                       ptrba += off*2; ptrbb = bb + off*2; */
            if ((left && trans) || (!left && !trans)) {
                ptrbb = bb;
            } else {
                ptrba += off * 2;
                ptrbb = bb + off * 2;
            }
            res0 = 0; res1 = 0; res2 = 0; res3 = 0;

            /* temp formula: see source lines 38-45.
               (LEFT && !TRANSA) || (!LEFT && TRANSA): temp = bk - off
               else (LEFT && TRANSA || !LEFT && !TRANSA): temp = off + 2 */
            if ((left && !trans) || (!left && trans))
                temp = bk - off;
            else
                temp = off + 2;

            for (k = 0; k < temp / 4; k += 1) {
                load0 = ptrba[2*0+0];
                load1 = ptrbb[2*0+0];
                res0 = res0 + load0 * load1;
                load2 = ptrba[2*0+1];
                res1 = res1 + load2 * load1;
                load3 = ptrbb[2*0+1];
                res2 = res2 + load0 * load3;
                res3 = res3 + load2 * load3;
                load4 = ptrba[2*1+0];
                load5 = ptrbb[2*1+0];
                res0 = res0 + load4 * load5;
                load6 = ptrba[2*1+1];
                res1 = res1 + load6 * load5;
                load7 = ptrbb[2*1+1];
                res2 = res2 + load4 * load7;
                res3 = res3 + load6 * load7;
                load0 = ptrba[2*2+0];
                load1 = ptrbb[2*2+0];
                res0 = res0 + load0 * load1;
                load2 = ptrba[2*2+1];
                res1 = res1 + load2 * load1;
                load3 = ptrbb[2*2+1];
                res2 = res2 + load0 * load3;
                res3 = res3 + load2 * load3;
                load4 = ptrba[2*3+0];
                load5 = ptrbb[2*3+0];
                res0 = res0 + load4 * load5;
                load6 = ptrba[2*3+1];
                res1 = res1 + load6 * load5;
                load7 = ptrbb[2*3+1];
                res2 = res2 + load4 * load7;
                res3 = res3 + load6 * load7;
                ptrba = ptrba + 8;
                ptrbb = ptrbb + 8;
            }
            for (k = 0; k < (temp & 3); k += 1) {
                load0 = ptrba[2*0+0];
                load1 = ptrbb[2*0+0];
                res0 = res0 + load0 * load1;
                load2 = ptrba[2*0+1];
                res1 = res1 + load2 * load1;
                load3 = ptrbb[2*0+1];
                res2 = res2 + load0 * load3;
                res3 = res3 + load2 * load3;
                ptrba = ptrba + 2;
                ptrbb = ptrbb + 2;
            }
            res0 = res0 * alpha; C0[0] = res0;
            res1 = res1 * alpha; C0[1] = res1;
            res2 = res2 * alpha; C1[0] = res2;
            res3 = res3 * alpha; C1[1] = res3;

            /* if (LEFT && TRANSA) || (!LEFT && !TRANSA): advance ptrba/ptrbb */
            if ((left && trans) || (!left && !trans)) {
                temp = bk - off;
                temp -= 2;
                ptrba += temp * 2;
                ptrbb += temp * 2;
            }
            if (left) off += 2;

            C0 = C0 + 2;
            C1 = C1 + 2;
        }

        /* bm & 1 — single-row tail */
        for (i = 0; i < (bm & 1); i += 1) {
            if ((left && trans) || (!left && !trans)) {
                ptrbb = bb;
            } else {
                ptrba += off;
                ptrbb = bb + off * 2;
            }
            res0 = 0; res1 = 0;

            if ((left && !trans) || (!left && trans))
                temp = bk - off;
            else if (left)
                temp = off + 1;
            else
                temp = off + 2;

            for (k = 0; k < temp; k += 1) {
                load0 = ptrba[0+0];
                load1 = ptrbb[2*0+0];
                res0 = res0 + load0 * load1;
                load2 = ptrbb[2*0+1];
                res1 = res1 + load0 * load2;
                ptrba = ptrba + 1;
                ptrbb = ptrbb + 2;
            }
            res0 = res0 * alpha; C0[0] = res0;
            res1 = res1 * alpha; C1[0] = res1;

            if ((left && trans) || (!left && !trans)) {
                temp = bk - off;
                if (left) temp -= 1;
                else      temp -= 2;
                ptrba += temp;
                ptrbb += temp * 2;
            }
            if (left) off += 1;

            C0 = C0 + 1;
            C1 = C1 + 1;
        }

        if (!left) off += 2;

        k = (bk << 1);
        bb = bb + k;
        i = (ldc << 1);
        C = C + i;
    }

    /* bn & 1 — single-col tail */
    for (j = 0; j < (bn & 1); j += 1) {
        C0 = C;
        if (left) off = offset;
        ptrba = ba;
        for (i = 0; i < bm / 2; i += 1) {
            if ((left && trans) || (!left && !trans)) {
                ptrbb = bb;
            } else {
                ptrba += off * 2;
                ptrbb = bb + off;
            }
            res0 = 0; res1 = 0;

            if ((left && !trans) || (!left && trans))
                temp = bk - off;
            else if (left)
                temp = off + 2;
            else
                temp = off + 1;

            for (k = 0; k < temp; k += 1) {
                load0 = ptrba[2*0+0];
                load1 = ptrbb[0+0];
                res0 = res0 + load0 * load1;
                load2 = ptrba[2*0+1];
                res1 = res1 + load2 * load1;
                ptrba = ptrba + 2;
                ptrbb = ptrbb + 1;
            }
            res0 = res0 * alpha; C0[0] = res0;
            res1 = res1 * alpha; C0[1] = res1;

            if ((left && trans) || (!left && !trans)) {
                temp = bk - off;
                if (left) temp -= 2;
                else      temp -= 1;
                ptrba += temp * 2;
                ptrbb += temp;
            }
            if (left) off += 2;

            C0 = C0 + 2;
        }
        for (i = 0; i < (bm & 1); i += 1) {
            if ((left && trans) || (!left && !trans)) {
                ptrbb = bb;
            } else {
                ptrba += off;
                ptrbb = bb + off;
            }
            res0 = 0;

            if ((left && !trans) || (!left && trans))
                temp = bk - off;
            else if (left)
                temp = off + 1;
            else
                temp = off + 1;

            for (k = 0; k < temp; k += 1) {
                load0 = ptrba[0+0];
                load1 = ptrbb[0+0];
                res0 = res0 + load0 * load1;
                ptrba = ptrba + 1;
                ptrbb = ptrbb + 1;
            }
            res0 = res0 * alpha; C0[0] = res0;

            if ((left && trans) || (!left && !trans)) {
                temp = bk - off;
                temp -= 1;
                ptrba += temp;
                ptrbb += temp;
            }
            if (left) off += 1;

            C0 = C0 + 1;
        }
        if (!left) off += 1;
        k = bk;  /* (bk<<0) */
        bb = bb + k;
        C = C + ldc;
    }
}
