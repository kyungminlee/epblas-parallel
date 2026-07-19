/*
 * qgemm_serial — kind16 (REAL(KIND=16) / __float128) GEMM, single-thread.
 * This TU owns ALL of the qgemm math; qgemm_parallel.c only orchestrates
 * threads over these same pieces.
 *
 * Structure: GotoBLAS / OpenBLAS — three-level cache blocking
 * (NC × KC × MC), copy-and-conquer packing (op(A), op(B) absorbed into
 * Ap/Bp), register-blocked MR×NR outer-product micro-kernel, sub-NC
 * chunking by NR to keep the C-tile hot across the K sweep, adaptive
 * MC when K is small.
 *
 * Differences from a textbook dgemm:
 *   - __float128 has no hardware FP unit; every multiply/add lowers to a
 *     libquadmath soft-float call. MR=2, NR=2 keeps four independent
 *     accumulators so the out-of-order engine can overlap those calls,
 *     and the K-loop is unrolled by 4 to widen the scheduling window.
 *   - Packing turns the strided B access of the naive reference into
 *     stride-1 streams, which matters because each quad element is 16 B.
 *     (This pays off for NN/NT/TT; the TN case streams both operands
 *     stride-1 without packing and takes the unpacked qgemm_fast_col path.)
 *   - Edge tiles for the M-tail and N-tail go through a scalar dot path.
 *
 * Packing layouts (panel-packed, OpenBLAS-style):
 *   Ap: tiled by MR rows. For each MR-panel within (ic..ic+ib),
 *       Ap_panel[p*MR + ii] = op(A)[ic + panel_off + ii, pc + p].
 *   Bp: tiled by NR cols. For each NR-panel within (jc..jc+jb),
 *       Bp_panel[p*NR + jj] = op(B)[pc + p, jc + panel_off + jj].
 *
 * Block sizes (fixed compile-time constants; see qgemm_choose_blocks):
 *   MC=64   panel rows (grown adaptively at small K toward the L2 budget)
 *   KC=256  panel depth
 *   NC=512  column band per thread
 * Register-tile dims MR=2, NR=2 are compile-time constants (QGEMM_MR/NR).
 *
 * ABI: qgemm_serial is the by-value core entry (char/ptrdiff_t by value,
 * alpha/beta by pointer); the public Fortran entry qgemm_ lives in
 * qgemm_parallel.c behind common/epblas_facade.h. Character args are bare
 * `char *` by design — NO hidden trailing length args anywhere (declaring
 * them corrupts reference-PBLAS caller frames; never re-add them).
 * REAL(KIND=16) ↔ `__float128`.
 */

#include "qgemm_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <stddef.h>

typedef qgemm_TR TR;

#define MR QGEMM_MR
#define NR QGEMM_NR

/* ── Block sizes ──────────────────────────────────────────────── */

static ptrdiff_t g_mc = 0, g_kc = 0, g_nc = 0;
static long g_l2_bytes = 0;        /* adaptive-MC cache target (see below) */
static void init_blocks(void) {
    if (g_mc) return;
    g_kc = 256;
    g_nc = 512;
    /* Adaptive-MC cache budget: a packed Ap block (MC*KC) should sit in L2.
     * Detect this core's L2 at runtime rather than hardcoding one machine's
     * size. */
    long sz = sysconf(_SC_LEVEL2_CACHE_SIZE);
    ptrdiff_t l2_kb = (sz > 0) ? (ptrdiff_t)(sz / 1024) : 256;
    g_l2_bytes = (long)l2_kb * 1024L;
    g_mc = 64;  /* set last: g_mc!=0 is the init flag */
}

/*
 * OpenBLAS-style adaptive MC: when K fits in one panel, grow MC so
 * MC*KC stays roughly L2-sized (rounded to MR).
 */
void qgemm_choose_blocks(ptrdiff_t k, ptrdiff_t *MC_out, ptrdiff_t *KC_out, ptrdiff_t *NC_out) {
    init_blocks();
    ptrdiff_t MC = g_mc, KC = g_kc, NC = g_nc;
    if (k > 0 && k <= KC) {
        long target_mc = g_l2_bytes / ((long)k * (long)sizeof(TR));
        if (target_mc > MC) {
            if (target_mc > 4L * g_mc) target_mc = 4L * g_mc;
            MC = blas_round_up((ptrdiff_t)target_mc, MR);
            if (MC < g_mc) MC = g_mc;
        }
    }
    *MC_out = MC; *KC_out = KC; *NC_out = NC;
}

/* ── beta pre-pass ────────────────────────────────────────────── */

void qgemm_beta_prepass(ptrdiff_t m, ptrdiff_t n, TR beta, TR *c, ptrdiff_t ldc) {
    for (ptrdiff_t j = 0; j < n; ++j) {
        TR *cj = &c[(size_t)j * ldc];
        if (beta == 0.0Q)      for (ptrdiff_t i = 0; i < m; ++i) cj[i]  = 0.0Q;
        else if (beta != 1.0Q) for (ptrdiff_t i = 0; i < m; ++i) cj[i] *= beta;
    }
}

/* ── Packers (panel-packed, OpenBLAS-style) ───────────────────── */

void qgemm_pack_A(const TR *restrict A, ptrdiff_t lda,
                  ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb,
                  ptrdiff_t ta, TR *restrict Ap)
{
    const ptrdiff_t npanel = (ib + MR - 1) / MR;
    for (ptrdiff_t q = 0; q < npanel; ++q) {
        const ptrdiff_t i0 = ic + q * MR;
        const ptrdiff_t rows = (q == npanel - 1) ? (ib - q * MR) : MR;
        TR *Apanel = &Ap[(size_t)q * pb * MR];
        if (ta == 'N') {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                const TR *src = &A[(size_t)(pc + p) * lda + i0];
                TR *dst = &Apanel[(size_t)p * MR];
                ptrdiff_t ii;
                for (ii = 0; ii < rows; ++ii) dst[ii] = src[ii];
                for (; ii < MR; ++ii) dst[ii] = 0.0Q;
            }
        } else {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                TR *dst = &Apanel[(size_t)p * MR];
                ptrdiff_t ii;
                for (ii = 0; ii < rows; ++ii)
                    dst[ii] = A[(size_t)(i0 + ii) * lda + (pc + p)];
                for (; ii < MR; ++ii) dst[ii] = 0.0Q;
            }
        }
    }
}

void qgemm_pack_B(const TR *restrict B, ptrdiff_t ldb,
                  ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb,
                  ptrdiff_t tb, TR *restrict Bp)
{
    const ptrdiff_t npanel = (jb + NR - 1) / NR;
    for (ptrdiff_t q = 0; q < npanel; ++q) {
        const ptrdiff_t j0 = jc + q * NR;
        const ptrdiff_t cols = (q == npanel - 1) ? (jb - q * NR) : NR;
        TR *Bpanel = &Bp[(size_t)q * pb * NR];
        if (tb == 'N') {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                TR *dst = &Bpanel[(size_t)p * NR];
                ptrdiff_t jj;
                for (jj = 0; jj < cols; ++jj)
                    dst[jj] = B[(size_t)(j0 + jj) * ldb + (pc + p)];
                for (; jj < NR; ++jj) dst[jj] = 0.0Q;
            }
        } else {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                const TR *src = &B[(size_t)(pc + p) * ldb + j0];
                TR *dst = &Bpanel[(size_t)p * NR];
                ptrdiff_t jj;
                for (jj = 0; jj < cols; ++jj) dst[jj] = src[jj];
                for (; jj < NR; ++jj) dst[jj] = 0.0Q;
            }
        }
    }
}

/* ── Inner kernel: MR × NR outer-product over K ──────────────── */

static inline void kernel_2x2(ptrdiff_t pb, TR alpha,
                              const TR *restrict Apanel,
                              const TR *restrict Bpanel,
                              TR *restrict C, ptrdiff_t ldc)
{
    TR c00 = 0.0Q, c01 = 0.0Q, c10 = 0.0Q, c11 = 0.0Q;
    /* K-loop unrolled by 4: amortizes the loop-counter/pointer arithmetic
     * over 4 MR×NR MAC-sets and widens the scheduling window for the four
     * independent soft-float accumulator chains. */
    const TR *ap = Apanel, *bp = Bpanel;
    ptrdiff_t p = pb;
    for (; p >= 4; p -= 4) {
        c00 += ap[0] * bp[0]; c10 += ap[1] * bp[0];
        c01 += ap[0] * bp[1]; c11 += ap[1] * bp[1];
        c00 += ap[2] * bp[2]; c10 += ap[3] * bp[2];
        c01 += ap[2] * bp[3]; c11 += ap[3] * bp[3];
        c00 += ap[4] * bp[4]; c10 += ap[5] * bp[4];
        c01 += ap[4] * bp[5]; c11 += ap[5] * bp[5];
        c00 += ap[6] * bp[6]; c10 += ap[7] * bp[6];
        c01 += ap[6] * bp[7]; c11 += ap[7] * bp[7];
        ap += 4 * MR;
        bp += 4 * NR;
    }
    for (; p > 0; --p) {
        const TR a0 = ap[0], a1 = ap[1];
        const TR b0 = bp[0], b1 = bp[1];
        c00 += a0 * b0; c10 += a1 * b0;
        c01 += a0 * b1; c11 += a1 * b1;
        ap += MR;
        bp += NR;
    }
    C[0]         += alpha * c00;
    C[1]         += alpha * c10;
    C[ldc + 0]   += alpha * c01;
    C[ldc + 1]   += alpha * c11;
}

/* Edge tile: arbitrary mr ∈ [1, MR], nr ∈ [1, NR] — scalar fallback. */
static void kernel_edge(ptrdiff_t mr, ptrdiff_t nr, ptrdiff_t pb, TR alpha,
                        const TR *restrict Apanel,
                        const TR *restrict Bpanel,
                        TR *restrict C, ptrdiff_t ldc)
{
    for (ptrdiff_t jj = 0; jj < nr; ++jj) {
        TR *cj = &C[(size_t)jj * ldc];
        for (ptrdiff_t ii = 0; ii < mr; ++ii) {
            TR sum = 0.0Q;
            for (ptrdiff_t p = 0; p < pb; ++p)
                sum += Apanel[(size_t)p * MR + ii] *
                       Bpanel[(size_t)p * NR + jj];
            cj[ii] += alpha * sum;
        }
    }
}

/* Drive one (ib, jb, pb) macro-tile via MR×NR sub-tiles. */
void qgemm_macro_kernel(ptrdiff_t ib, ptrdiff_t jb, ptrdiff_t pb, TR alpha,
                        const TR *restrict Ap, const TR *restrict Bp,
                        TR *restrict C, ptrdiff_t ldc)
{
    const ptrdiff_t npA = (ib + MR - 1) / MR;
    const ptrdiff_t npB = (jb + NR - 1) / NR;
    for (ptrdiff_t q = 0; q < npB; ++q) {
        const ptrdiff_t jj0  = q * NR;
        const ptrdiff_t nr_q = (q == npB - 1) ? (jb - jj0) : NR;
        const TR *Bpanel = &Bp[(size_t)q * pb * NR];
        for (ptrdiff_t r = 0; r < npA; ++r) {
            const ptrdiff_t ii0  = r * MR;
            const ptrdiff_t mr_r = (r == npA - 1) ? (ib - ii0) : MR;
            const TR *Apanel = &Ap[(size_t)r * pb * MR];
            TR *Ctile = &C[(size_t)jj0 * ldc + ii0];
            if (mr_r == MR && nr_q == NR) {
                kernel_2x2(pb, alpha, Apanel, Bpanel, Ctile, ldc);
            } else {
                kernel_edge(mr_r, nr_q, pb, alpha, Apanel, Bpanel, Ctile, ldc);
            }
        }
    }
}

/* ── Small-N path: TA='T' (≡'C'), TB='T' (≡'C') — unblocked plain dot ──
 *
 * C[i,j] = alpha * sum_l A^op[i,l] B^op[l,j], with A^op[i,l] = a[i*lda+l]
 * (stride-1 along the contraction l) and B^op[l,j] = b[l*ldb+j] (strided by
 * ldb along l). A plain single-accumulator dot per (i,j) — NO aligned_alloc,
 * NO packing pass.
 *
 * Used ONLY for small problems (see QGEMM_TT_UNBLOCK_FLOPS in qgemm_serial /
 * qgemm_core). There the blocked packed path's fixed cost — a multi-MB
 * aligned_alloc of Bp plus a packing sweep — cannot amortize over the few
 * flops, and it loses the TT cells to the gfortran reference (par/mig was
 * ~1.04-1.08 at N=64/128). This kernel has neither cost and matches the
 * reference (par/mig ~1.00-1.015), beating ob too.
 *
 * Why a single accumulator and not a 2×2 register tile: the dot here is
 * SHORT (small k), so the soft-float fadd-latency the packed kernel's four
 * chains hide is a non-issue, while a 2×2 tile's extra loads + wider strided-B
 * footprint measurably REGRESS it (par/mig 1.04 vs the plain dot's 1.005 at
 * N=64). Once k grows the four-chain ILP does pay — that is exactly the large-N
 * regime the gate hands back to the packed path. Mirrors ygemm's unblocked
 * tt_plain core: for compute-bound arithmetic with no SIMD, the reference-
 * shaped dot wins at the small sizes where packing is pure overhead.
 *
 * (TN takes qgemm_fast_col instead: there BOTH operands are stride-1 along l,
 * so a single-column dot already streams perfectly — TT cannot, B is strided.) */
void qgemm_tt_unblocked(ptrdiff_t j_start, ptrdiff_t j_end,
                        ptrdiff_t m, ptrdiff_t k, TR alpha,
                        const TR *a, ptrdiff_t lda,
                        const TR *b, ptrdiff_t ldb,
                        TR *c, ptrdiff_t ldc)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        TR *cj = &c[(size_t)j * ldc];
        for (ptrdiff_t i = 0; i < m; ++i) {
            const TR *ai = &a[(size_t)i * lda];
            TR acc = 0.0Q;
            for (ptrdiff_t l = 0; l < k; ++l) acc += ai[l] * b[(size_t)l * ldb + j];
            cj[i] += alpha * acc;
        }
    }
}

/* Gate: use the unblocked plain dot for small TT problems, the blocked packed
 * path otherwise. The packed path's fixed overhead (multi-MB Bp alloc + pack)
 * only amortizes once the flop count is large; below the threshold the plain
 * dot wins (and its single-accumulator latency is hidden by the short k). The
 * threshold sits between the N=128 cube (plain wins) and the N=256 cube
 * (packed wins). */
#define QGEMM_TT_UNBLOCK_FLOPS 8000000.0  /* m*n*k product (not 2*flops) */
bool qgemm_tt_small(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k) {
    return (double)m * (double)n * (double)k < QGEMM_TT_UNBLOCK_FLOPS;
}

/* ── Fast path: TA = 'T' (≡ 'C' for real), TB = 'N', one C-column ── */
void qgemm_fast_col(ptrdiff_t j2, ptrdiff_t m, ptrdiff_t k, TR alpha,
                    const TR *a, ptrdiff_t lda, const TR *b, ptrdiff_t ldb,
                    TR *c, ptrdiff_t ldc)
{
    TR *cj = &c[(size_t)j2 * ldc];
    const TR *bj = &b[(size_t)j2 * ldb];
    for (ptrdiff_t i2 = 0; i2 < m; ++i2) {
        const TR *ai = &a[(size_t)i2 * lda];
        TR acc = 0.0Q;
        for (ptrdiff_t l = 0; l < k; ++l) acc += ai[l] * bj[l];
        cj[i2] += alpha * acc;
    }
}

/* ── Single-thread entry (by-value ptrdiff_t dims) ────────────── */

void qgemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const TR *alpha_,
    const TR *a, ptrdiff_t lda,
    const TR *b, ptrdiff_t ldb,
    const TR *beta_,
    TR *c, ptrdiff_t ldc)
{
    const TR alpha = *alpha_, beta = *beta_;
    const char ta = blas_trans_real(transa);
    const char tb = blas_trans_real(transb);

    if (m <= 0 || n <= 0) return;

    qgemm_beta_prepass(m, n, beta, c, ldc);   /* handles K==0 / alpha==0 */
    if (alpha == 0.0Q || k == 0) return;

    /* TN: unpacked stride-1 dot — for __float128 this beats the blocked packed
     * path at every K and size (packing is pure overhead, no SIMD; see the
     * qgemm_ parallel entry for the measured ratios). */
    if (ta == 'T' && tb == 'N') {
        for (ptrdiff_t j2 = 0; j2 < n; ++j2)
            qgemm_fast_col(j2, m, k, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

    /* TT small problems: unblocked plain dot — no alloc, no packing. Beats the
     * blocked packed path where its alloc+pack fixed cost can't amortize and
     * the gfortran reference wins; large TT falls through to the packed path. */
    if (ta == 'T' && tb == 'T' && qgemm_tt_small(m, n, k)) {
        qgemm_tt_unblocked(0, n, m, k, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

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
                qgemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                for (ptrdiff_t ic = 0; ic < m; ic += MC) {
                    const ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
                    qgemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                    qgemm_macro_kernel(ib, jb, pb, alpha, Ap, Bp,
                                       &c[(size_t)jc * ldc + ic], ldc);
                }
            }
        }
    }
    free(Ap);
    free(Bp);
}
