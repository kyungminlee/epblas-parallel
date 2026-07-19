/*
 * egemm_serial — kind10 (REAL(KIND=10), x86-64 80-bit long double) GEMM,
 * single-thread. This TU owns ALL of the egemm math; egemm_parallel.c
 * only orchestrates threads over these same pieces.
 *
 * Structure: GotoBLAS / OpenBLAS — three-level cache blocking
 * (NC × KC × MC), copy-and-conquer packing (op(A), op(B) absorbed into
 * Ap/Bp), register-blocked MR×NR outer-product micro-kernel, sub-NC
 * chunking by NR to keep the C-tile hot across the K sweep, adaptive
 * MC when K is small.
 *
 * Differences from OpenBLAS dgemm:
 *   - No assembly. Pure C inner kernel; gcc keeps the 4 MR×NR fp80
 *     accumulators on the x87 stack across the K-loop (MR=2, NR=2 is
 *     deliberately small to fit the 8-deep x87 register stack).
 *   - No SIMD on `long double` (x86-64 has no AVX path for 80-bit).
 *   - Edge tiles for the M-tail and N-tail go through a scalar dot path.
 *
 * Packing layouts (panel-packed, OpenBLAS-style):
 *   Ap: tiled by MR rows. For each MR-panel within (ic..ic+ib),
 *       Ap_panel[p*MR + ii] = op(A)[ic + panel_off + ii, pc + p].
 *   Bp: tiled by NR cols. For each NR-panel within (jc..jc+jb),
 *       Bp_panel[p*NR + jj] = op(B)[pc + p, jc + panel_off + jj].
 *   Inner kernel reads MR consecutive A elements and NR consecutive B
 *   elements per p — both stride-1 in the packed layout.
 *
 * Block sizes (fixed compile-time constants):
 *   MC=64   panel rows (grown adaptively at small K toward the L2 budget)
 *   KC=256  panel depth
 *   NC=512  column band per thread
 * Register-tile dims MR=2, NR=2 are compile-time constants (EGEMM_MR/NR).
 *
 * ABI: egemm_serial is the by-value core entry (char/ptrdiff_t by value,
 * alpha/beta by pointer); the public Fortran entry egemm_ lives in
 * egemm_parallel.c behind common/epblas_facade.h. Character args are bare
 * `char *` by design — NO hidden trailing length args anywhere (declaring
 * them corrupts reference-PBLAS caller frames; never re-add them).
 * REAL(KIND=10) ↔ `long double` (x86-64 80-bit extended).
 */

#include "egemm_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include <stdlib.h>
#include <ctype.h>
#include <unistd.h>
#include <stddef.h>

typedef egemm_TR TR;

#define MR EGEMM_MR
#define NR EGEMM_NR

/* ── Block sizes ──────────────────────────────────────────────── */

static ptrdiff_t g_mc = 0, g_kc = 0, g_nc = 0;
static long g_l2_bytes = 0;        /* adaptive-MC cache target (see below) */
static void init_blocks(void) {
    if (g_mc) return;
    g_kc = 256;
    g_nc = 512;
    /* Adaptive-MC cache budget: a packed Ap block (MC*KC) should sit in L2.
     * Detect this core's L2 at runtime rather than hardcoding one machine's
     * size — a wrong target bloats Ap past L2 (e.g. a 768K target on a 256K
     * L2 core blew Ap to 768K at K=256, ~1% slower). */
    long sz = sysconf(_SC_LEVEL2_CACHE_SIZE);
    ptrdiff_t l2_kb = (sz > 0) ? (ptrdiff_t)(sz / 1024) : 256;
    g_l2_bytes = (long)l2_kb * 1024L;
    g_mc = 64;  /* set last: g_mc!=0 is the init flag */
}

/*
 * OpenBLAS-style adaptive MC: when K fits in one panel, grow MC so
 * MC*KC stays roughly L2-sized (rounded to MR). Helps small-K shapes
 * where the default MC under-uses cache.
 */
void egemm_choose_blocks(ptrdiff_t k, ptrdiff_t *MC_out, ptrdiff_t *KC_out, ptrdiff_t *NC_out) {
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

void egemm_beta_prepass(ptrdiff_t m, ptrdiff_t n, TR beta, TR *c, ptrdiff_t ldc) {
    for (ptrdiff_t j = 0; j < n; ++j) {
        TR *cj = &c[(size_t)j * ldc];
        if (beta == 0.0L)      for (ptrdiff_t i = 0; i < m; ++i) cj[i]  = 0.0L;
        else if (beta != 1.0L) for (ptrdiff_t i = 0; i < m; ++i) cj[i] *= beta;
    }
}

/* ── Packers (panel-packed, OpenBLAS-style) ───────────────────── */

/*
 * Pack op(A)(ic..ic+ib, pc..pc+pb) into Ap as a stack of MR-row panels.
 * Panel layout:  Ap[(ii_panel * pb + p) * MR + ii] = op(A)[ic + ii_panel*MR + ii, pc + p].
 *
 * The last panel is zero-padded to MR rows when ib % MR != 0.
 */
void egemm_pack_A(const TR *restrict A, ptrdiff_t lda,
                  ptrdiff_t ic, ptrdiff_t pc, ptrdiff_t ib, ptrdiff_t pb,
                  char ta, TR *restrict Ap)
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
                for (; ii < MR; ++ii) dst[ii] = 0.0L;
            }
        } else {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                TR *dst = &Apanel[(size_t)p * MR];
                ptrdiff_t ii;
                for (ii = 0; ii < rows; ++ii)
                    dst[ii] = A[(size_t)(i0 + ii) * lda + (pc + p)];
                for (; ii < MR; ++ii) dst[ii] = 0.0L;
            }
        }
    }
}

/*
 * Pack op(B)(pc..pc+pb, jc..jc+jb) into Bp as a stack of NR-col panels.
 * Panel layout:  Bp[(jj_panel * pb + p) * NR + jj] = op(B)[pc + p, jc + jj_panel*NR + jj].
 *
 * egemm_pack_B_range packs only panels [q_lo, q_hi) of the npanel total; panels
 * write DISJOINT Bp regions, so the threaded entry can split this `omp for` over
 * the team (kills the serial `omp single` B-pack fraction that Amdahl-capped
 * small-N OMP=4 scaling) while keeping a shared Bp — no redundant pack, no axis
 * change. egemm_pack_B is the whole-panel-range wrapper (serial path). */
void egemm_pack_B_range(const TR *restrict B, ptrdiff_t ldb,
                        ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb,
                        char tb, TR *restrict Bp, ptrdiff_t q_lo, ptrdiff_t q_hi)
{
    const ptrdiff_t npanel = (jb + NR - 1) / NR;
    for (ptrdiff_t q = q_lo; q < q_hi; ++q) {
        const ptrdiff_t j0 = jc + q * NR;
        const ptrdiff_t cols = (q == npanel - 1) ? (jb - q * NR) : NR;
        TR *Bpanel = &Bp[(size_t)q * pb * NR];
        if (tb == 'N') {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                TR *dst = &Bpanel[(size_t)p * NR];
                ptrdiff_t jj;
                for (jj = 0; jj < cols; ++jj)
                    dst[jj] = B[(size_t)(j0 + jj) * ldb + (pc + p)];
                for (; jj < NR; ++jj) dst[jj] = 0.0L;
            }
        } else {
            for (ptrdiff_t p = 0; p < pb; ++p) {
                const TR *src = &B[(size_t)(pc + p) * ldb + j0];
                TR *dst = &Bpanel[(size_t)p * NR];
                ptrdiff_t jj;
                for (jj = 0; jj < cols; ++jj) dst[jj] = src[jj];
                for (; jj < NR; ++jj) dst[jj] = 0.0L;
            }
        }
    }
}

void egemm_pack_B(const TR *restrict B, ptrdiff_t ldb,
                  ptrdiff_t pc, ptrdiff_t jc, ptrdiff_t pb, ptrdiff_t jb,
                  char tb, TR *restrict Bp)
{
    egemm_pack_B_range(B, ldb, pc, jc, pb, jb, tb, Bp, 0, (jb + NR - 1) / NR);
}

/* ── Inner kernel: MR × NR outer-product over K ──────────────── */

/*
 * Full MR × NR tile. C strip is column-major (stride ldc between cols).
 * Four scalar accumulators live on the x87 register stack across the
 * K loop; each iteration loads MR A-elements and NR B-elements from
 * packed buffers and issues MR*NR independent multiply-adds.
 */
static inline void kernel_2x2(ptrdiff_t pb, TR alpha,
                              const TR *restrict Apanel,
                              const TR *restrict Bpanel,
                              TR *restrict C, ptrdiff_t ldc)
{
    TR c00 = 0.0L, c01 = 0.0L, c10 = 0.0L, c11 = 0.0L;
    /* K-loop unrolled by 4 (matches the OpenBLAS-overlay kernel): amortizes
     * the loop-counter/pointer arithmetic over 4 MR×NR MAC-sets and widens
     * gcc's scheduling window for the x87 accumulator chains. */
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
            TR sum = 0.0L;
            for (ptrdiff_t p = 0; p < pb; ++p)
                sum += Apanel[(size_t)p * MR + ii] *
                       Bpanel[(size_t)p * NR + jj];
            cj[ii] += alpha * sum;
        }
    }
}

/* Drive one (ib, jb, pb) macro-tile via MR×NR sub-tiles. */
void egemm_macro_kernel(ptrdiff_t ib, ptrdiff_t jb, ptrdiff_t pb, TR alpha,
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

/* ── Fast path: TA = 'T' (≡ 'C' for real), TB = 'N', one C-column ──
 *
 * With column-major storage the inner k-loop reads A column i and B
 * column j both stride-1 — near peak x87 throughput. Packing adds
 * overhead the blocked path can never recover here; this explicit
 * reference body matches migrated at ~1.0× across all sizes.
 */
void egemm_fast_col(ptrdiff_t j2, ptrdiff_t m, ptrdiff_t k, TR alpha,
                    const TR *a, ptrdiff_t lda, const TR *b, ptrdiff_t ldb,
                    TR *c, ptrdiff_t ldc)
{
    TR *cj = &c[(size_t)j2 * ldc];
    const TR *bj = &b[(size_t)j2 * ldb];
    for (ptrdiff_t i2 = 0; i2 < m; ++i2) {
        const TR *ai = &a[(size_t)i2 * lda];
        TR acc = 0.0L;
        for (ptrdiff_t l = 0; l < k; ++l) acc += ai[l] * bj[l];
        cj[i2] += alpha * acc;
    }
}

/* ── Unpacked register-tiled TN fast path (moderate, L2-resident N) ──
 *
 * TN (ta='T', tb='N') reads A column i and B column j both stride-1 in the
 * contraction index l, so no packing is needed. fast_col above already exploits
 * that — but with a SINGLE fp80 accumulator, whose ~3-cyc x87 `fadd` latency
 * serializes the reduction (~5.1 cyc/MAC); that is why fast_col is gated to
 * skinny/tiny only and the moderate square cube is sent to the blocked path.
 *
 * This kernel keeps the blocked path's FOUR independent MR×NR accumulator
 * chains (so it hides the fadd latency, ~2.9 cyc/MAC) WITHOUT the pack — the
 * pack is pure overhead when the cube is small enough to stay L2-resident, so
 * the blocked path's only edge there is its ILP, which this matches. Net: the
 * small-square TN cells that lost ~1-2% to OpenBLAS's native A^T·B kernel
 * (TN/64, TN/128) close to a win. l-ascending summation is bit-identical to
 * both the blocked path and the netlib oracle. (The egemmtr 2026-06-24 lesson,
 * minus the triangle: egemm writes the full M×N rectangle.) */
static inline void egemm_tn_kernel_2x2(ptrdiff_t k, TR alpha,
        const TR *restrict ai0, const TR *restrict ai1,
        const TR *restrict bj0, const TR *restrict bj1,
        TR *restrict C, ptrdiff_t ldc)
{
    TR c00 = 0.0L, c10 = 0.0L, c01 = 0.0L, c11 = 0.0L;
    for (ptrdiff_t l = 0; l < k; ++l) {
        const TR a0 = ai0[l], a1 = ai1[l], b0 = bj0[l], b1 = bj1[l];
        c00 += a0 * b0; c10 += a1 * b0; c01 += a0 * b1; c11 += a1 * b1;
    }
    C[0]       += alpha * c00; C[1]       += alpha * c10;
    C[ldc]     += alpha * c01; C[ldc + 1] += alpha * c11;
}

void egemm_unpacked_tn(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, TR alpha,
                       const TR *restrict a, ptrdiff_t lda,
                       const TR *restrict b, ptrdiff_t ldb,
                       TR *restrict c, ptrdiff_t ldc)
{
    for (ptrdiff_t j0 = 0; j0 < n; j0 += NR) {
        const ptrdiff_t njr = blas_imin(NR, n - j0);
        TR *restrict Ccol = &c[(size_t)j0 * ldc];
        for (ptrdiff_t i0 = 0; i0 < m; i0 += MR) {
            const ptrdiff_t mir = blas_imin(MR, m - i0);
            if (mir == MR && njr == NR) {
                egemm_tn_kernel_2x2(k, alpha,
                    &a[(size_t)i0 * lda], &a[(size_t)(i0 + 1) * lda],
                    &b[(size_t)j0 * ldb], &b[(size_t)(j0 + 1) * ldb],
                    &Ccol[i0], ldc);
            } else {
                for (ptrdiff_t jj = 0; jj < njr; ++jj) {
                    const TR *restrict bj = &b[(size_t)(j0 + jj) * ldb];
                    TR *restrict cj = &Ccol[(size_t)jj * ldc];
                    for (ptrdiff_t ii = 0; ii < mir; ++ii) {
                        const TR *restrict ai = &a[(size_t)(i0 + ii) * lda];
                        TR s = 0.0L;
                        for (ptrdiff_t l = 0; l < k; ++l) s += ai[l] * bj[l];
                        cj[i0 + ii] += alpha * s;
                    }
                }
            }
        }
    }
}

/* ── Single-thread entry ──────────────────────────────────────── */

void egemm_serial(
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

    egemm_beta_prepass(m, n, beta, c, ldc);   /* handles K==0 / alpha==0 */
    if (alpha == 0.0L || k == 0) return;

    /* TN no-pack fast path only for skinny problems; otherwise the blocked
     * packed path below is faster per FLOP (4-way accumulator ILP). */
    if (ta == 'T' && tb == 'N' && egemm_tn_use_fast(m, n, k)) {
        for (ptrdiff_t j2 = 0; j2 < n; ++j2)
            egemm_fast_col(j2, m, k, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

    /* Moderate-square TN: unpacked 4-chain tile (skips the pack, which buys
     * nothing while L2-resident). Bit-identical to the blocked path. */
    if (ta == 'T' && tb == 'N' && egemm_tn_use_unpacked(m, n, k)) {
        egemm_unpacked_tn(m, n, k, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(k, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * blas_round_up(NC, NR) * sizeof(TR);
    /* Persistent grow-only thread-local pack arena (Ap|Bp in one block): a
     * per-call aligned_alloc+free of these mmap-threshold-sized buffers trips
     * glibc's trim heuristic and re-faults every touched page each call — a
     * pure page-fault tax at small N (see etrsm_serial.c). */
    static __thread TR *g_pack = NULL;
    static __thread size_t g_pack_cap = 0;
    const size_t ap_al = (ap_bytes + 63) & ~(size_t)63;
    const size_t need  = ap_al + ((bp_bytes + 63) & ~(size_t)63);
    if (need > g_pack_cap) {
        free(g_pack);
        size_t cap = need + (need >> 1);            /* 1.5× headroom to amortize regrow */
        cap = (cap + 63) & ~(size_t)63;
        g_pack = aligned_alloc(64, cap);
        g_pack_cap = g_pack ? cap : 0;
    }
    if (g_pack) {
        TR *Ap = g_pack;
        TR *Bp = (TR *)(void *)((char *)g_pack + ap_al);
        for (ptrdiff_t jc = 0; jc < n; jc += NC) {
            const ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
            for (ptrdiff_t pc = 0; pc < k; pc += KC) {
                const ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
                egemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                for (ptrdiff_t ic = 0; ic < m; ic += MC) {
                    const ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
                    egemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                    egemm_macro_kernel(ib, jb, pb, alpha, Ap, Bp,
                                       &c[(size_t)jc * ldc + ic], ldc);
                }
            }
        }
    }
}
