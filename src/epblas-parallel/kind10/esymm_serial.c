/*
 * esymm_serial — kind10 real (long double) symmetric matrix-multiply,
 * single-thread. Owns the SYMM-aware packers and the fused serial driver.
 *
 *   C := alpha * A * B + beta * C    (SIDE='L', A symmetric M×M)
 *   C := alpha * B * A + beta * C    (SIDE='R', A symmetric N×N)
 *
 * Structure (mirrors the OpenBLAS-overlay esymm, on par's own kernels):
 * one packed GEMM. The symmetric operand is packed via esymm_pack_{a,b}_sym
 * — which mirror the UPLO triangle into egemm's packed layout — and the
 * regular operand via the stock egemm packers. The shared MR×NR macro-
 * kernel then streams diagonal and off-diagonal tiles identically. No
 * scalar diagonal kernel, no per-tile re-dispatch into egemm.
 *
 * The kernel pieces (block policy, packers, beta pre-pass, macro-kernel)
 * are the serial egemm primitives from egemm_kernel.h — calling them
 * (not egemm_) keeps esymm free of any nested OpenMP team, so it is safe
 * to run inside another routine's parallel region (the libgomp barrier
 * wedge guard).
 */

#include "esymm_kernel.h"
#include "../common/blas_char.h"
#include "egemm_kernel.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef esymm_TR TR;

#define MR EGEMM_MR
#define NR EGEMM_NR

/* Pack a sub-block of an UPLO-stored symmetric matrix into egemm's packed
 * MR/NR panel layout. Adapted from the OpenBLAS-overlay SYMM_{U,L}TCOPY
 * (common/eblas_l3_real.c eblas_esymm_{u,l}copy): the `m` depth axis streams
 * along incremental row/column pointers (ao1/ao2) chosen by the running
 * diagonal `offset = posX - posY`, two strips at a time. `offset` decreases
 * monotonically along the depth walk, so the per-element branch crosses the
 * diagonal at most once per strip — the branch predictor nails the rest. This
 * replaces a per-element sym_at() (a multiply + UPLO test for every element);
 * the packed output is bit-identical.
 *
 * Because A is symmetric, sym(r,c) == sym(c,r), so the SIDE=L (A-into-Ap,
 * posX=row-base) and SIDE=R (A-into-Bp, posX=col-base) packs are the SAME walk
 * with posX/posY playing swapped roles — one routine serves both. No tail
 * padding: ragged MR/NR panels are consumed only by kernel_edge, which reads
 * just the real rows/cols (see egemm_macro_kernel). */
static void esymm_pack_u(ptrdiff_t m, ptrdiff_t n, const TR *a, ptrdiff_t lda,
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
        /* Lone trailing strip: write the single column at panel stride MR
         * (kernel_edge reads Apanel[p*MR], so the odd panel keeps the same
         * stride as a full one — unlike OpenBLAS, which packs it contiguous). */
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

static void esymm_pack_l(ptrdiff_t m, ptrdiff_t n, const TR *a, ptrdiff_t lda,
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
        /* Lone trailing strip at panel stride MR — see esymm_pack_u. */
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

void esymm_pack_a_sym(const TR *a, ptrdiff_t lda,
                      ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb,
                      char uplo, TR *Ap)
{
    /* SIDE=L: A is the M×K symmetric operand. Rows (ib) form the panel/strip
     * axis (posX), the K depth (pb) streams (m, posY). */
    if (uplo == 'U') esymm_pack_u(pb, ib, a, lda, ic, pc, Ap);
    else             esymm_pack_l(pb, ib, a, lda, ic, pc, Ap);
}

void esymm_pack_b_sym(const TR *a, ptrdiff_t lda,
                      ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb,
                      char uplo, TR *Bp)
{
    /* SIDE=R: A is the K×N symmetric operand in the B slot. Columns (jb) form
     * the panel/strip axis (posX), the K depth (pb) streams (m, posY). By
     * symmetry this is the identical walk to the A pack. */
    if (uplo == 'U') esymm_pack_u(pb, jb, a, lda, jc, pc, Bp);
    else             esymm_pack_l(pb, jb, a, lda, jc, pc, Bp);
}

void esymm_serial(
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

    egemm_beta_prepass(m, n, beta, c, ldc);   /* C := beta*C (handles beta 0/1) */
    if (alpha == 0.0L) return;

    /* K is the contraction dim = the symmetric matrix's side. */
    const ptrdiff_t k = (SIDE == 'L') ? m : n;

    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(k, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * egemm_round_up(NC, NR) * sizeof(TR);
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
                    egemm_pack_B(b, ldb, pc, jc, pb, jb, 'N', Bp);
                else
                    esymm_pack_b_sym(a, lda, pc, jc, pb, jb, UPLO, Bp);

                for (ptrdiff_t ic = 0; ic < m; ic += MC) {
                    const ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
                    /* Pack the M×K (ic-block) left operand. SIDE='L': the
                     * symmetric A. SIDE='R': regular B in the A slot. */
                    if (SIDE == 'L')
                        esymm_pack_a_sym(a, lda, ic, pc, ib, pb, UPLO, Ap);
                    else
                        egemm_pack_A(b, ldb, ic, pc, ib, pb, 'N', Ap);

                    egemm_macro_kernel(ib, jb, pb, alpha, Ap, Bp,
                                       &c[(size_t)jc * ldc + ic], ldc);
                }
            }
        }
    }
    free(Ap);
    free(Bp);
}
