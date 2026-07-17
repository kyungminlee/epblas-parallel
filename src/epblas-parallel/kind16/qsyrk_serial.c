/*
 * qsyrk_serial — kind16 real (__float128) symmetric rank-k update,
 * single-thread. Owns the SYRK-specific math (triangular β pre-pass +
 * diagonal-aware writeback kernel) and the fused serial driver.
 *
 *   C := alpha · A · A^T + beta · C    (trans='N', A is N×K)
 *   C := alpha · A^T · A + beta · C    (trans='T', A is K×N)
 *
 * Only the UPLO triangle of C is read or written.
 *
 * Structure (faithful __float128 port of kind10 esyrk / OpenBLAS DSYRK —
 * interface/syrk.c dispatch, driver/level3/level3_syrk.c blocking nest,
 * driver/level3/syrk_kernel.c diagonal kernel): one packed GEMM whose output
 * is clipped to the UPLO triangle. Both Ap and Bp source from the same input A
 * (A doubles as B); the diagonal kernel splits each MC×NC block around the
 * global diagonal, GEMMing the strict-triangle remainders and merging only the
 * UPLO triangle of each diagonal NR×NR sub-block.
 *
 * Built on the SHARED ob-convention substrate (qtri_gemm_kernel / qtri_ncopy
 * / qtri_tcopy, qtri_kernel.h) because the diagonal kernel indexes the packed
 * buffers at arbitrary (possibly odd) offsets, which is only valid under
 * OpenBLAS's contiguous-odd-tail packing (par's qgemm zero-pads odd tails at
 * stride MR and would mis-read those bytes). The layout-agnostic block policy
 * is shared with qgemm (qgemm_choose_blocks / blas_round_up). Calling only
 * these *serial* primitives (never the threaded entries) keeps qsyrk free of
 * any nested OpenMP team, so it is safe inside another routine's parallel
 * region.
 */

#include "qsyrk_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "qtri_kernel.h"
#include "qgemm_kernel.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>

typedef qsyrk_TR TR;

#define MR QSYRK_MR
#define NR QSYRK_NR

/* ── Triangular β pre-pass ─────────────────────────────────────────────
 * Port of OpenBLAS driver/level3/syrk_k.c's `syrk_beta`. Scales only the
 * UPLO triangle of C; the off-UPLO triangle is left untouched. */
void qsyrk_beta_u(ptrdiff_t n, TR beta, TR *c, ptrdiff_t ldc) {
    if (beta == 1.0Q) return;
    if (beta == 0.0Q) {
        for (ptrdiff_t j = 0; j < n; ++j) {
            TR *cj = c + j * ldc;
            for (ptrdiff_t i = 0; i <= j; ++i) cj[i] = 0.0Q;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            TR *cj = c + j * ldc;
            for (ptrdiff_t i = 0; i <= j; ++i) cj[i] *= beta;
        }
    }
}

void qsyrk_beta_l(ptrdiff_t n, TR beta, TR *c, ptrdiff_t ldc) {
    if (beta == 1.0Q) return;
    if (beta == 0.0Q) {
        for (ptrdiff_t j = 0; j < n; ++j) {
            TR *cj = c + j * ldc;
            for (ptrdiff_t i = j; i < n; ++i) cj[i] = 0.0Q;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            TR *cj = c + j * ldc;
            for (ptrdiff_t i = j; i < n; ++i) cj[i] *= beta;
        }
    }
}

/* ── Diagonal-aware writeback kernel ───────────────────────────────────
 * Faithful port of OpenBLAS driver/level3/syrk_kernel.c, with the GEMM
 * micro-kernel → the shared qtri_gemm_kernel. The strict-triangle rectangular
 * remainders are full GEMMs into C; each diagonal NR×NR sub-block is GEMMed
 * into a zeroed subbuffer (qtri_gemm_kernel accumulates), then only its UPLO
 * triangle is merged into C. Subbuffer sized NR*(NR+1) to match OpenBLAS's
 * safety pad. */
void qsyrk_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, TR alpha,
                    const TR *a, const TR *b,
                    TR *c, ptrdiff_t ldc, ptrdiff_t offset)
{
    TR subbuf[NR * (NR + 1)];

    if (m + offset < 0) {
        qtri_gemm_kernel(m, n, k, alpha, a, b, c, ldc);
        return;
    }
    if (n < offset) {
        return;
    }
    if (offset > 0) {
        b += offset * k;
        c += offset * ldc;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        qtri_gemm_kernel(m, n - m - offset, k, alpha,
                         a, b + (m + offset) * k,
                         c + (m + offset) * ldc, ldc);
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        qtri_gemm_kernel(-offset, n, k, alpha, a, b, c, ldc);
        a -= offset * k;
        c -= offset;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }

    /* Diagonal walk in NR-step blocks. offset == 0, m == n here. */
    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        /* Strict-upper portion: rows 0..loop-1 × cols loop..loop+nn-1. */
        if (loop > 0) {
            qtri_gemm_kernel(loop, nn, k, alpha,
                             a, b + loop * k, c + loop * ldc, ldc);
        }

        /* Diagonal block via subbuffer. */
        for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0.0Q;
        qtri_gemm_kernel(nn, nn, k, alpha,
                         a + loop * k, b + loop * k, subbuf, nn);

        TR *cc = c + loop + loop * ldc;
        const TR *ss = subbuf;
        for (ptrdiff_t j = 0; j < nn; ++j) {
            for (ptrdiff_t i = 0; i <= j; ++i) cc[i] += ss[i];
            ss += nn;
            cc += ldc;
        }
    }
}

void qsyrk_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, TR alpha,
                    const TR *a, const TR *b,
                    TR *c, ptrdiff_t ldc, ptrdiff_t offset)
{
    TR subbuf[NR * (NR + 1)];

    if (m + offset < 0) {
        return;
    }
    if (n < offset) {
        qtri_gemm_kernel(m, n, k, alpha, a, b, c, ldc);
        return;
    }
    if (offset > 0) {
        qtri_gemm_kernel(m, offset, k, alpha, a, b, c, ldc);
        b += offset * k;
        c += offset * ldc;
        n -= offset;
        offset = 0;
        if (n <= 0) return;
    }
    if (n > m + offset) {
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        a -= offset * k;
        c -= offset;
        m += offset;
        offset = 0;
        if (m <= 0) return;
    }
    if (m > n - offset) {
        qtri_gemm_kernel(m - n + offset, n, k, alpha,
                         a + (n - offset) * k, b,
                         c + (n - offset), ldc);
        m = n + offset;
        if (m <= 0) return;
    }

    /* Diagonal walk in NR-step blocks. offset == 0, m == n here. */
    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        /* Diagonal block via subbuffer. */
        for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0.0Q;
        qtri_gemm_kernel(nn, nn, k, alpha,
                         a + loop * k, b + loop * k, subbuf, nn);

        TR *cc = c + loop + loop * ldc;
        const TR *ss = subbuf;
        for (ptrdiff_t j = 0; j < nn; ++j) {
            for (ptrdiff_t i = j; i < nn; ++i) cc[i] += ss[i];
            ss += nn;
            cc += ldc;
        }

        /* Strict-lower portion: rows loop+nn..m-1 × cols loop..loop+nn-1. */
        if (m > mm + nn) {
            qtri_gemm_kernel(m - mm - nn, nn, k, alpha,
                             a + (mm + nn) * k, b + loop * k,
                             c + (mm + nn) + loop * ldc, ldc);
        }
    }
}

/* ── Transpose inner-product (trans='T') ───────────────────────────────
 * C := alpha·A^T·A + beta·C over the UPLO triangle of output column j. A is
 * K×N so both dot operands are unit-stride over the K axis. beta is FUSED
 * into the store (no separate caller prescale — the yherk 9140682 shape):
 * beta·C(i,j) is rounded to fp128 either way and IEEE add commutes, so the
 * fused form is bit-identical to prescale-then-+=; beta∈{0,1} skip the
 * multiply exactly as qsyrk_beta_{u,l} did. The beta case split is resolved
 * to an int ONCE per column — a per-element `beta == 0.0Q` would be a
 * libgcc __eqtf2 call inside the hot loop. Unlike the NoTrans outer product
 * (which RMW-streams C K times per column), each C(i,j) is accumulated in a
 * register and written once — packing has nothing to save here, so the
 * clean unpacked loop matches the reference. */
void qsyrk_trans_col(ptrdiff_t j, char UPLO, ptrdiff_t n, ptrdiff_t k,
                     TR alpha, TR beta,
                     const TR *a, ptrdiff_t lda, TR *c, ptrdiff_t ldc)
{
    const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
    const ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
    const TR *Aj = a + j * lda;
    TR *cj = c + j * ldc;
    const ptrdiff_t bmode = (beta == 0.0Q) ? 0 : (beta == 1.0Q) ? 1 : 2;
    for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
        const TR *Ai = a + i * lda;
        /* Single accumulator: SYRK's dot is one product per l, so the two
         * libquadmath soft-float calls (__multf3 then __addtf3) already serialize
         * on each other — an even/odd accumulator split adds tail logic without
         * any independent chain to overlap, and measured slower (unlike syr2k,
         * whose two distinct products genuinely overlap). */
        TR s = 0.0Q;
        for (ptrdiff_t l = 0; l < k; ++l) s += Ai[l] * Aj[l];
        const TR t = alpha * s;
        if (bmode == 0)      cj[i] = t;
        else if (bmode == 1) cj[i] = t + cj[i];
        else                 cj[i] = t + beta * cj[i];
    }
}

/* ── Single-thread driver ──────────────────────────────────────────────
 * Faithful port of OpenBLAS level3_syrk.c with a single thread spanning the
 * whole output (rows ∈ [0, N]). The js-band UPLO clip bounds the active row
 * range so the kernel only ever writes the requested triangle. */
void qsyrk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TR *alpha_,
    const TR *a, ptrdiff_t lda,
    const TR *beta_,
    TR *c, ptrdiff_t ldc)
{
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO  = blas_up(uplo);
    const char TRANS = blas_up(trans);

    if (n <= 0) return;

    /* Degenerate update: pure triangular beta scale. */
    if (k == 0 || alpha == 0.0Q) {
        if (UPLO == 'U') qsyrk_beta_u(n, beta, c, ldc);
        else             qsyrk_beta_l(n, beta, c, ldc);
        return;
    }

    /* Transpose: netlib-style unpacked inner-product (no packing overhead).
     * beta rides the column store — no separate prescale pass over C. */
    if (TRANS != 'N') {
        for (ptrdiff_t j = 0; j < n; ++j)
            qsyrk_trans_col(j, UPLO, n, k, alpha, beta, a, lda, c, ldc);
        return;
    }

    /* NoTrans packed path: triangular beta pre-pass, then accumulate. */
    if (UPLO == 'U') qsyrk_beta_u(n, beta, c, ldc);
    else             qsyrk_beta_l(n, beta, c, ldc);

    ptrdiff_t MC, KC, NC;
    qgemm_choose_blocks(k, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * (size_t)KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * (size_t)blas_round_up(NC, NR) * sizeof(TR);
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
        for (ptrdiff_t js = 0; js < n; js += NC) {
            const ptrdiff_t jb = (n - js < NC) ? (n - js) : NC;

            /* UPLO clip of the [0, N] row range for this js-band:
             *   UPPER: only rows up to js+jb contribute.
             *   LOWER: only rows from js onwards. */
            ptrdiff_t m_lo_eff = (UPLO == 'L') ? js : 0;
            ptrdiff_t m_hi_eff = (UPLO == 'U' && n > js + jb) ? (js + jb) : n;
            if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);

            for (ptrdiff_t ls = 0; ls < k; ls += KC) {
                const ptrdiff_t pb = (k - ls < KC) ? (k - ls) : KC;

                /* Pack Bp = the same A in OCOPY shape (A doubles as B). NoTrans
                 * only — the Transpose path is handled above, unpacked. */
                qtri_tcopy(pb, jb, &a[(size_t)ls * lda + js], lda, Bp);

                for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                    const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    qtri_tcopy(pb, min_i, &a[(size_t)ls * lda + is], lda, Ap);

                    if (UPLO == 'U')
                        qsyrk_kernel_u(min_i, jb, pb, alpha, Ap, Bp,
                                       &c[(size_t)js * ldc + is], ldc,
                                       (ptrdiff_t)(is - js));
                    else
                        qsyrk_kernel_l(min_i, jb, pb, alpha, Ap, Bp,
                                       &c[(size_t)js * ldc + is], ldc,
                                       (ptrdiff_t)(is - js));
                }
            }
        }
    }
}
