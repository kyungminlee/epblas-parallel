/*
 * qsyr2k_serial — kind16 real (__float128) symmetric rank-2k update,
 * single-thread. Owns the SYR2K-specific diagonal-aware writeback kernel and
 * the fused serial driver.
 *
 *   C := alpha·(A·B^T + B·A^T) + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·(A^T·B + B^T·A) + beta·C   (trans='T', A,B are K×N)
 *
 * Only the UPLO triangle of C is read or written.
 *
 * Structure (faithful __float128 port of kind10 esyr2k / OpenBLAS DSYR2K —
 * interface/syr2k.c dispatch, driver/level3/level3_syr2k.c blocking nest,
 * driver/level3/syr2k_kernel.c diagonal kernel): one packed GEMM nest whose
 * output is clipped to the UPLO triangle, run TWICE per (is,js) tile. Pass 1
 * (Ap=A, Bp=B, flag=1) folds the A·B^T contribution plus both diagonal halves
 * via the symmetric subbuffer merge; pass 2 (Ap=B, Bp=A, flag=0) adds the
 * off-diagonal B·A^T strips.
 *
 * Built on the SHARED ob-convention substrate (qtri_gemm_kernel / qtri_ncopy
 * / qtri_tcopy, qtri_kernel.h) — the same one qsyrk uses — because the diagonal
 * kernel indexes the packed buffers at arbitrary (possibly odd) offsets, valid
 * only under OpenBLAS's contiguous-odd-tail packing. The triangular β pre-pass
 * is the SYRK helper (qsyrk_beta_{u,l}), reused as ob does. The layout-agnostic
 * block policy is shared with qgemm (qgemm_choose_blocks / qgemm_round_up).
 * Calling only these *serial* primitives keeps qsyr2k free of any nested
 * OpenMP team, so it is safe inside another routine's parallel region.
 */

#include "qsyr2k_kernel.h"
#include "../common/blas_char.h"
#include "qsyrk_kernel.h"   /* qsyrk_beta_{u,l} — shared triangular β pre-pass */
#include "qtri_kernel.h"
#include "qgemm_kernel.h"   /* qgemm_choose_blocks / qgemm_round_up */
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>

typedef qsyr2k_T T;

#define MR QSYR2K_MR
#define NR QSYR2K_NR

/* ── Diagonal-aware writeback kernel ───────────────────────────────────
 * Faithful port of OpenBLAS driver/level3/syr2k_kernel.c, with the GEMM
 * micro-kernel → the shared qtri_gemm_kernel. Identical to the SYRK kernel
 * except for the `flag`-gated diagonal block, whose merge adds the symmetric
 * pair (subbuf + subbuf^T) so a single pass 1 covers both A·B^T and B·A^T on
 * the NR×NR diagonal tile. Subbuffer sized NR*(NR+1) to match OpenBLAS's safety
 * pad. */
void qsyr2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, T alpha,
                     const T *a, const T *b,
                     T *c, ptrdiff_t ldc, ptrdiff_t offset, bool flag)
{
    T subbuf[NR * (NR + 1)];

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

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        /* Strict-upper portion: rows 0..loop-1 × cols loop..loop+nn-1. */
        if (loop > 0) {
            qtri_gemm_kernel(loop, nn, k, alpha,
                             a, b + loop * k, c + loop * ldc, ldc);
        }

        if (flag) {
            /* Diagonal NR×NR block: GEMM into subbuf, merge symmetric. */
            for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0.0Q;
            qtri_gemm_kernel(nn, nn, k, alpha,
                             a + loop * k, b + loop * k, subbuf, nn);

            T *cc = c + loop + loop * ldc;
            for (ptrdiff_t j = 0; j < nn; ++j) {
                for (ptrdiff_t i = 0; i <= j; ++i) {
                    cc[i + j * ldc] += subbuf[i + j * nn]
                                     + subbuf[j + i * nn];
                }
            }
        }
    }
}

void qsyr2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, T alpha,
                     const T *a, const T *b,
                     T *c, ptrdiff_t ldc, ptrdiff_t offset, bool flag)
{
    T subbuf[NR * (NR + 1)];

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

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (flag) {
            for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0.0Q;
            qtri_gemm_kernel(nn, nn, k, alpha,
                             a + loop * k, b + loop * k, subbuf, nn);

            T *cc = c + loop + loop * ldc;
            for (ptrdiff_t j = 0; j < nn; ++j) {
                for (ptrdiff_t i = j; i < nn; ++i) {
                    cc[i + j * ldc] += subbuf[i + j * nn]
                                     + subbuf[j + i * nn];
                }
            }
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
 * C := alpha·(A^T·B + B^T·A) + C over the UPLO triangle of output column j.
 * A,B are K×N so all dot operands are unit-stride over the K axis. β is already
 * applied. Each C(i,j) is accumulated in a register and written once — packing
 * has nothing to save here, so the clean unpacked loop matches the reference. */
void qsyr2k_trans_col(ptrdiff_t j, char uplo, ptrdiff_t n, ptrdiff_t k,
                      T alpha, const T *a, ptrdiff_t lda,
                      const T *b, ptrdiff_t ldb, T *c, ptrdiff_t ldc)
{
    const ptrdiff_t i_lo = (uplo == 'L') ? j : 0;
    const ptrdiff_t i_hi = (uplo == 'L') ? n : j + 1;
    const T *Aj = a + j * lda;
    const T *Bj = b + j * ldb;
    T *cj = c + j * ldc;
    for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
        const T *Ai = a + i * lda;
        const T *Bi = b + i * ldb;
        /* Two independent accumulators (netlib's temp1/temp2): the A·B and
         * B·A dot chains have no mutual dependency, so the out-of-order core
         * overlaps consecutive libquadmath __addtf3 calls — fp128 has no FMA
         * and each add is a serial soft-float call, so a single fused `s`
         * chain is latency-bound; splitting hides it (~4-8% on Transpose). */
        T s1 = 0.0Q, s2 = 0.0Q;
        for (ptrdiff_t l = 0; l < k; ++l) {
            s1 += Ai[l] * Bj[l];
            s2 += Bi[l] * Aj[l];
        }
        cj[i] += alpha * (s1 + s2);
    }
}

/* ── Single-thread driver ──────────────────────────────────────────────
 * Faithful port of OpenBLAS level3_syr2k.c with a single thread spanning the
 * whole output (rows ∈ [0, N]). The js-band UPLO clip bounds the active row
 * range so the kernel only ever writes the requested triangle. Two B-packs
 * (A and B in OCOPY shape) and two A-packs (ditto in ICOPY shape) per tile;
 * pass 1 = (Ap_A, Bp_B, flag=1), pass 2 = (Ap_B, Bp_A, flag=0). */
void qsyr2k_serial(
    char uplo_c, char trans_c,
    ptrdiff_t n, ptrdiff_t k,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    const T *beta_,
    T *c, ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const char uplo  = blas_up(uplo_c);
    char trans = blas_up(trans_c);
    if (trans == 'C') trans = 'T';

    if (n <= 0) return;

    if (uplo == 'U') qsyrk_beta_u(n, beta, c, ldc);
    else             qsyrk_beta_l(n, beta, c, ldc);

    if (k == 0 || alpha == 0.0Q) return;

    /* Transpose: netlib-style unpacked inner-product (no packing overhead). */
    if (trans != 'N') {
        for (ptrdiff_t j = 0; j < n; ++j)
            qsyr2k_trans_col(j, uplo, n, k, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

    ptrdiff_t MC, KC, NC;
    qgemm_choose_blocks(k, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)qgemm_round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)qgemm_round_up(NC, NR) * sizeof(T);
    T *Ap_A = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    T *Ap_B = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    T *Bp_A = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    T *Bp_B = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (Ap_A && Ap_B && Bp_A && Bp_B) {
        for (ptrdiff_t js = 0; js < n; js += NC) {
            const ptrdiff_t jb = (n - js < NC) ? (n - js) : NC;

            /* UPLO clip of the [0, N] row range for this js-band:
             *   UPPER: only rows up to js+jb contribute.
             *   LOWER: only rows from js onwards. */
            ptrdiff_t m_lo_eff = (uplo == 'L') ? js : 0;
            ptrdiff_t m_hi_eff = (uplo == 'U' && n > js + jb) ? (js + jb) : n;
            if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);

            for (ptrdiff_t ls = 0; ls < k; ls += KC) {
                const ptrdiff_t pb = (k - ls < KC) ? (k - ls) : KC;

                /* Pack both B-side panels (A and B in OCOPY shape). NoTrans
                 * only — the Transpose path is handled above, unpacked. */
                qtri_tcopy(pb, jb, &a[(size_t)ls * lda + js], lda, Bp_A);
                qtri_tcopy(pb, jb, &b[(size_t)ls * ldb + js], ldb, Bp_B);

                for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                    const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    qtri_tcopy(pb, min_i, &a[(size_t)ls * lda + is], lda, Ap_A);
                    qtri_tcopy(pb, min_i, &b[(size_t)ls * ldb + is], ldb, Ap_B);

                    T *cij = &c[(size_t)js * ldc + is];
                    const ptrdiff_t off = (ptrdiff_t)(is - js);

                    /* Pass 1: alpha·A·B^T + symmetric diagonal merge. */
                    if (uplo == 'U')
                        qsyr2k_kernel_u(min_i, jb, pb, alpha, Ap_A, Bp_B, cij, ldc, off, 1);
                    else
                        qsyr2k_kernel_l(min_i, jb, pb, alpha, Ap_A, Bp_B, cij, ldc, off, 1);

                    /* Pass 2: alpha·B·A^T into the off-diagonal strips only. */
                    if (uplo == 'U')
                        qsyr2k_kernel_u(min_i, jb, pb, alpha, Ap_B, Bp_A, cij, ldc, off, 0);
                    else
                        qsyr2k_kernel_l(min_i, jb, pb, alpha, Ap_B, Bp_A, cij, ldc, off, 0);
                }
            }
        }
    }
    free(Ap_A);
    free(Ap_B);
    free(Bp_A);
    free(Bp_B);
}
