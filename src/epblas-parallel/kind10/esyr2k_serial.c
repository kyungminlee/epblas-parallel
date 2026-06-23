/*
 * esyr2k_serial — kind10 real (long double) symmetric rank-2k update,
 * single-thread. Owns the SYR2K-specific diagonal-aware writeback kernel and
 * the fused serial driver.
 *
 *   C := alpha·(A·B^T + B·A^T) + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·(A^T·B + B^T·A) + beta·C   (trans='T', A,B are K×N)
 *
 * Only the UPLO triangle of C is read or written.
 *
 * Structure (faithful port of OpenBLAS DSYR2K — interface/syr2k.c dispatch,
 * driver/level3/level3_syr2k.c blocking nest, driver/level3/syr2k_kernel.c
 * diagonal kernel): one packed GEMM nest whose output is clipped to the UPLO
 * triangle, run TWICE per (is,js) tile. Pass 1 (Ap=A, Bp=B, flag=1) folds the
 * A·B^T contribution plus both diagonal halves via the symmetric subbuffer
 * merge; pass 2 (Ap=B, Bp=A, flag=0) adds the off-diagonal B·A^T strips.
 *
 * Built on the SHARED ob-convention substrate (etri_gemm_kernel / etri_ncopy
 * / etri_tcopy, etri_kernel.h) — the same one etrsm/etrmm/esyrk use — because
 * the diagonal kernel indexes the packed buffers at arbitrary (possibly odd)
 * offsets, valid only under OpenBLAS's contiguous-odd-tail packing. The
 * triangular β pre-pass is the SYRK helper (esyrk_beta_{u,l}), reused as ob
 * does. The layout-agnostic block policy is shared with egemm
 * (egemm_choose_blocks / blas_round_up). Calling only these *serial*
 * primitives keeps esyr2k free of any nested OpenMP team, so it is safe inside
 * another routine's parallel region (the libgomp barrier-wedge guard, memory
 * project-etrsm-omp4-wedge).
 */

#include "esyr2k_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "esyrk_kernel.h"   /* esyrk_beta_{u,l} — shared triangular β pre-pass */
#include "etri_kernel.h"
#include "egemm_kernel.h"   /* egemm_choose_blocks / blas_round_up */
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>

typedef esyr2k_TR TR;

#define MR ESYR2K_MR
#define NR ESYR2K_NR

/* ── Diagonal-aware writeback kernel ───────────────────────────────────
 * Faithful port of OpenBLAS driver/level3/syr2k_kernel.c (the openblas
 * overlay's eblas_esyr2k_kernel_{u,l}), with eblas_egemm_kernel → the shared
 * etri_gemm_kernel. Identical to the SYRK kernel except for the `flag`-gated
 * diagonal block, whose merge adds the symmetric pair (subbuf + subbuf^T) so a
 * single pass 1 covers both A·B^T and B·A^T on the NR×NR diagonal tile.
 * Subbuffer sized NR*(NR+1) to match OpenBLAS's safety pad. */
void esyr2k_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, TR alpha,
                     const TR *a, const TR *b,
                     TR *c, ptrdiff_t ldc, ptrdiff_t offset, ptrdiff_t flag)
{
    TR subbuf[NR * (NR + 1)];

    if (m + offset < 0) {
        etri_gemm_kernel(m, n, k, alpha, a, b, c, ldc);
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
        etri_gemm_kernel(m, n - m - offset, k, alpha,
                         a, b + (m + offset) * k,
                         c + (m + offset) * ldc, ldc);
        n = m + offset;
        if (n <= 0) return;
    }
    if (offset < 0) {
        etri_gemm_kernel(-offset, n, k, alpha, a, b, c, ldc);
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
            etri_gemm_kernel(loop, nn, k, alpha,
                             a, b + loop * k, c + loop * ldc, ldc);
        }

        if (flag) {
            /* Diagonal NR×NR block: GEMM into subbuf, merge symmetric. */
            for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0.0L;
            etri_gemm_kernel(nn, nn, k, alpha,
                             a + loop * k, b + loop * k, subbuf, nn);

            TR *cc = c + loop + loop * ldc;
            for (ptrdiff_t j = 0; j < nn; ++j) {
                for (ptrdiff_t i = 0; i <= j; ++i) {
                    cc[i + j * ldc] += subbuf[i + j * nn]
                                     + subbuf[j + i * nn];
                }
            }
        }
    }
}

void esyr2k_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, TR alpha,
                     const TR *a, const TR *b,
                     TR *c, ptrdiff_t ldc, ptrdiff_t offset, ptrdiff_t flag)
{
    TR subbuf[NR * (NR + 1)];

    if (m + offset < 0) {
        return;
    }
    if (n < offset) {
        etri_gemm_kernel(m, n, k, alpha, a, b, c, ldc);
        return;
    }
    if (offset > 0) {
        etri_gemm_kernel(m, offset, k, alpha, a, b, c, ldc);
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
        etri_gemm_kernel(m - n + offset, n, k, alpha,
                         a + (n - offset) * k, b,
                         c + (n - offset), ldc);
        m = n + offset;
        if (m <= 0) return;
    }

    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        if (flag) {
            for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0.0L;
            etri_gemm_kernel(nn, nn, k, alpha,
                             a + loop * k, b + loop * k, subbuf, nn);

            TR *cc = c + loop + loop * ldc;
            for (ptrdiff_t j = 0; j < nn; ++j) {
                for (ptrdiff_t i = j; i < nn; ++i) {
                    cc[i + j * ldc] += subbuf[i + j * nn]
                                     + subbuf[j + i * nn];
                }
            }
        }

        /* Strict-lower portion: rows loop+nn..m-1 × cols loop..loop+nn-1. */
        if (m > mm + nn) {
            etri_gemm_kernel(m - mm - nn, nn, k, alpha,
                             a + (mm + nn) * k, b + loop * k,
                             c + (mm + nn) + loop * ldc, ldc);
        }
    }
}

/* ── Single-thread driver ──────────────────────────────────────────────
 * Faithful port of OpenBLAS level3_syr2k.c with a single thread spanning the
 * whole output (rows ∈ [0, N]). The js-band UPLO clip bounds the active row
 * range so the kernel only ever writes the requested triangle. Two B-packs
 * (A and B in OCOPY shape) and two A-packs (ditto in ICOPY shape) per tile;
 * pass 1 = (Ap_A, Bp_B, flag=1), pass 2 = (Ap_B, Bp_A, flag=0). */
void esyr2k_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TR *alpha_,
    const TR *a, ptrdiff_t lda,
    const TR *b, ptrdiff_t ldb,
    const TR *beta_,
    TR *c, ptrdiff_t ldc)
{
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO  = blas_up(uplo);
    const char TRANS = blas_up(trans);

    if (n <= 0) return;

    if (UPLO == 'U') esyrk_beta_u((ptrdiff_t)n, beta, c, (ptrdiff_t)ldc);
    else             esyrk_beta_l((ptrdiff_t)n, beta, c, (ptrdiff_t)ldc);

    if (k == 0 || alpha == 0.0L) return;

    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(k, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * (size_t)KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * (size_t)blas_round_up(NC, NR) * sizeof(TR);
    TR *Ap_A = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    TR *Ap_B = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    TR *Bp_A = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    TR *Bp_B = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (Ap_A && Ap_B && Bp_A && Bp_B) {
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

                /* Pack both B-side panels (A and B in OCOPY shape). */
                if (TRANS == 'N') {
                    etri_tcopy(pb, jb, &a[(size_t)ls * lda + js], lda, Bp_A);
                    etri_tcopy(pb, jb, &b[(size_t)ls * ldb + js], ldb, Bp_B);
                } else {
                    etri_ncopy(pb, jb, &a[(size_t)js * lda + ls], lda, Bp_A);
                    etri_ncopy(pb, jb, &b[(size_t)js * ldb + ls], ldb, Bp_B);
                }

                for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                    const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (TRANS == 'N') {
                        etri_tcopy(pb, min_i, &a[(size_t)ls * lda + is], lda, Ap_A);
                        etri_tcopy(pb, min_i, &b[(size_t)ls * ldb + is], ldb, Ap_B);
                    } else {
                        etri_ncopy(pb, min_i, &a[(size_t)is * lda + ls], lda, Ap_A);
                        etri_ncopy(pb, min_i, &b[(size_t)is * ldb + ls], ldb, Ap_B);
                    }

                    TR *cij = &c[(size_t)js * ldc + is];
                    const ptrdiff_t off = (ptrdiff_t)(is - js);

                    /* Pass 1: alpha·A·B^T + symmetric diagonal merge. */
                    if (UPLO == 'U')
                        esyr2k_kernel_u(min_i, jb, pb, alpha, Ap_A, Bp_B, cij, ldc, off, 1);
                    else
                        esyr2k_kernel_l(min_i, jb, pb, alpha, Ap_A, Bp_B, cij, ldc, off, 1);

                    /* Pass 2: alpha·B·A^T into the off-diagonal strips only. */
                    if (UPLO == 'U')
                        esyr2k_kernel_u(min_i, jb, pb, alpha, Ap_B, Bp_A, cij, ldc, off, 0);
                    else
                        esyr2k_kernel_l(min_i, jb, pb, alpha, Ap_B, Bp_A, cij, ldc, off, 0);
                }
            }
        }
    }
    free(Ap_A);
    free(Ap_B);
    free(Bp_A);
    free(Bp_B);
}
