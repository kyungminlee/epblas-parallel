/*
 * qsymm_serial — kind16 real (__float128) symmetric matrix-multiply,
 * single-thread. Owns the SYMM-aware packers and the fused serial driver.
 *
 *   C := alpha * A * B + beta * C    (SIDE='L', A symmetric M×M)
 *   C := alpha * B * A + beta * C    (SIDE='R', A symmetric N×N)
 *
 * Structure (mirrors the kind10 esymm overlay, on par's own qgemm kernels):
 * one packed GEMM. The symmetric operand is packed via qsymm_pack_{a,b}_sym
 * — which mirror the UPLO triangle into qgemm's packed layout — and the
 * regular operand via the stock qgemm packers. The shared MR×NR macro-
 * kernel then streams diagonal and off-diagonal tiles identically.
 *
 * The kernel pieces (block policy, packers, beta pre-pass, macro-kernel)
 * are the serial qgemm primitives from qgemm_kernel.h — calling them keeps
 * qsymm free of any nested OpenMP team, so it is safe to run inside another
 * routine's parallel region.
 */

#include "qsymm_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "qgemm_kernel.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef qsymm_TR TR;

#define MR QGEMM_MR
#define NR QGEMM_NR

char qsymm_uplo(const char *p) {
    return blas_up(*p);
}

/* Pack a sub-block of an UPLO-stored symmetric matrix into qgemm's packed
 * MR/NR panel layout. The `m` depth axis streams along incremental
 * row/column pointers (ao1/ao2) chosen by the running diagonal
 * `offset = posX - posY`, two strips at a time. `offset` decreases
 * monotonically along the depth walk, so the per-element branch crosses the
 * diagonal at most once per strip. Because A is symmetric, sym(r,c) ==
 * sym(c,r), so the SIDE=L (A-into-Ap) and SIDE=R (A-into-Bp) packs are the
 * SAME walk with posX/posY swapped — one routine serves both. */
static void qsymm_pack_u(ptrdiff_t m, ptrdiff_t n, const TR *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY, TR *b)
{
    ptrdiff_t js = n >> 1;
    while (js > 0) {
        ptrdiff_t offset = posX - posY;
        const TR *ao1 = (offset >  0) ? a + (size_t)posY + (size_t)(posX + 0) * lda
                                     : a + (size_t)(posX + 0) + (size_t)posY * lda;
        const TR *ao2 = (offset > -1) ? a + (size_t)posY + (size_t)(posX + 1) * lda
                                     : a + (size_t)(posX + 1) + (size_t)posY * lda;
        for (ptrdiff_t i = m; i > 0; --i) {
            b[0] = ao1[0]; b[1] = ao2[0]; b += 2;
            ao1 += (offset >  0) ? 1 : lda;
            ao2 += (offset > -1) ? 1 : lda;
            offset--;
        }
        posX += 2;
        js--;
    }
    if (n & 1) {
        /* Lone trailing strip at panel stride MR (kernel_edge reads
         * Apanel[p*MR], so the odd panel keeps the same stride). */
        ptrdiff_t offset = posX - posY;
        const TR *ao1 = (offset > 0) ? a + (size_t)posY + (size_t)(posX + 0) * lda
                                    : a + (size_t)(posX + 0) + (size_t)posY * lda;
        for (ptrdiff_t i = m; i > 0; --i) {
            b[0] = ao1[0]; b += MR;
            ao1 += (offset > 0) ? 1 : lda;
            offset--;
        }
    }
}

static void qsymm_pack_l(ptrdiff_t m, ptrdiff_t n, const TR *a, ptrdiff_t lda,
                         ptrdiff_t posX, ptrdiff_t posY, TR *b)
{
    ptrdiff_t js = n >> 1;
    while (js > 0) {
        ptrdiff_t offset = posX - posY;
        const TR *ao1 = (offset >  0) ? a + (size_t)(posX + 0) + (size_t)posY * lda
                                     : a + (size_t)posY + (size_t)(posX + 0) * lda;
        const TR *ao2 = (offset > -1) ? a + (size_t)(posX + 1) + (size_t)posY * lda
                                     : a + (size_t)posY + (size_t)(posX + 1) * lda;
        for (ptrdiff_t i = m; i > 0; --i) {
            b[0] = ao1[0]; b[1] = ao2[0]; b += 2;
            ao1 += (offset >  0) ? lda : 1;
            ao2 += (offset > -1) ? lda : 1;
            offset--;
        }
        posX += 2;
        js--;
    }
    if (n & 1) {
        /* Lone trailing strip at panel stride MR — see qsymm_pack_u. */
        ptrdiff_t offset = posX - posY;
        const TR *ao1 = (offset > 0) ? a + (size_t)(posX + 0) + (size_t)posY * lda
                                    : a + (size_t)posY + (size_t)(posX + 0) * lda;
        for (ptrdiff_t i = m; i > 0; --i) {
            b[0] = ao1[0]; b += MR;
            ao1 += (offset > 0) ? lda : 1;
            offset--;
        }
    }
}

void qsymm_pack_a_sym(const TR *a, ptrdiff_t lda,
                      ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb,
                      char UPLO, TR *Ap)
{
    /* SIDE=L: A is the M×K symmetric operand. Rows (ib) form the panel/strip
     * axis (posX), the K depth (pb) streams (m, posY). */
    if (UPLO == 'U') qsymm_pack_u(pb, ib, a, lda, ic, pc, Ap);
    else             qsymm_pack_l(pb, ib, a, lda, ic, pc, Ap);
}

void qsymm_pack_b_sym(const TR *a, ptrdiff_t lda,
                      ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb,
                      char UPLO, TR *Bp)
{
    /* SIDE=R: A is the K×N symmetric operand in the B slot. Columns (jb) form
     * the panel/strip axis (posX), the K depth (pb) streams (m, posY). By
     * symmetry this is the identical walk to the A pack. */
    if (UPLO == 'U') qsymm_pack_u(pb, jb, a, lda, jc, pc, Bp);
    else             qsymm_pack_l(pb, jb, a, lda, jc, pc, Bp);
}

void qsymm_serial(
    char side, char uplo,
    ptrdiff_t m, ptrdiff_t n,
    const TR *alpha_,
    const TR *a, ptrdiff_t lda,
    const TR *b, ptrdiff_t ldb,
    const TR *beta_,
    TR *c, ptrdiff_t ldc)
{
    const TR alpha = *alpha_, beta = *beta_;
    const char SIDE = blas_up(side);
    const char UPLO = blas_up(uplo);

    if (m <= 0 || n <= 0) return;

    qgemm_beta_prepass(m, n, beta, c, ldc);   /* C := beta*C (handles beta 0/1) */
    if (alpha == 0.0Q) return;

    /* K is the contraction dim = the symmetric matrix's side. */
    const ptrdiff_t k = (SIDE == 'L') ? m : n;

    ptrdiff_t MC, KC, NC;
    qgemm_choose_blocks(k, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * blas_round_up(NC, NR) * sizeof(TR);
    TR *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    TR *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (Ap && Bp) {
        for (ptrdiff_t jc = 0; jc < n; jc += NC) {
            const ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
            for (ptrdiff_t pc = 0; pc < k; pc += KC) {
                const ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
                /* Pack the K×N (jc-band) right operand. SIDE='L': regular B.
                 * SIDE='R': the symmetric A goes in the B slot. */
                if (SIDE == 'L')
                    qgemm_pack_B(b, ldb, pc, jc, pb, jb, 'N', Bp);
                else
                    qsymm_pack_b_sym(a, lda, pc, jc, pb, jb, UPLO, Bp);

                for (ptrdiff_t ic = 0; ic < m; ic += MC) {
                    const ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
                    /* Pack the M×K (ic-block) left operand. SIDE='L': the
                     * symmetric A. SIDE='R': regular B in the A slot. */
                    if (SIDE == 'L')
                        qsymm_pack_a_sym(a, lda, ic, pc, ib, pb, UPLO, Ap);
                    else
                        qgemm_pack_A(b, ldb, ic, pc, ib, pb, 'N', Ap);

                    qgemm_macro_kernel(ib, jb, pb, alpha, Ap, Bp,
                                       &c[(size_t)jc * ldc + ic], ldc);
                }
            }
        }
    }
    free(Ap);
    free(Bp);
}
