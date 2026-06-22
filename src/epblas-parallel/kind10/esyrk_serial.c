/*
 * esyrk_serial — kind10 real (long double) symmetric rank-k update,
 * single-thread. Owns the SYRK-specific math (triangular β pre-pass +
 * diagonal-aware writeback kernel) and the fused serial driver.
 *
 *   C := alpha · A · A^T + beta · C    (trans='N', A is N×K)
 *   C := alpha · A^T · A + beta · C    (trans='T', A is K×N)
 *
 * Only the UPLO triangle of C is read or written.
 *
 * Structure (faithful port of OpenBLAS DSYRK — interface/syrk.c dispatch,
 * driver/level3/level3_syrk.c blocking nest, driver/level3/syrk_kernel.c
 * diagonal kernel): one packed GEMM whose output is clipped to the UPLO
 * triangle. Both Ap and Bp source from the same input A (A doubles as B);
 * the diagonal kernel splits each MC×NC block around the global diagonal,
 * GEMMing the strict-triangle remainders and merging only the UPLO triangle
 * of each diagonal NR×NR sub-block.
 *
 * Built on the SHARED ob-convention substrate (etri_gemm_kernel / etri_ncopy
 * / etri_tcopy, etri_kernel.h) — the same one etrsm/etrmm use — because the
 * diagonal kernel indexes the packed buffers at arbitrary (possibly odd)
 * offsets, which is only valid under OpenBLAS's contiguous-odd-tail packing
 * (par's egemm zero-pads odd tails at stride MR and would mis-read those
 * bytes). The layout-agnostic block policy is shared with egemm
 * (egemm_choose_blocks / egemm_round_up). Calling only these *serial*
 * primitives (never the threaded entries) keeps esyrk free of any nested
 * OpenMP team, so it is safe inside another routine's parallel region (the
 * libgomp barrier-wedge guard).
 */

#include "esyrk_kernel.h"
#include "../common/blas_char.h"
#include "etri_kernel.h"
#include "egemm_kernel.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>

typedef esyrk_T T;

#define MR ESYRK_MR
#define NR ESYRK_NR

/* ── Triangular β pre-pass ─────────────────────────────────────────────
 * Port of OpenBLAS driver/level3/syrk_k.c's `syrk_beta`. Scales only the
 * UPLO triangle of C; the off-UPLO triangle is left untouched. */
void esyrk_beta_u(ptrdiff_t n, T beta, T *c, ptrdiff_t ldc) {
    if (beta == 1.0L) return;
    if (beta == 0.0L) {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T *cj = c + j * ldc;
            for (ptrdiff_t i = 0; i <= j; ++i) cj[i] = 0.0L;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T *cj = c + j * ldc;
            for (ptrdiff_t i = 0; i <= j; ++i) cj[i] *= beta;
        }
    }
}

void esyrk_beta_l(ptrdiff_t n, T beta, T *c, ptrdiff_t ldc) {
    if (beta == 1.0L) return;
    if (beta == 0.0L) {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T *cj = c + j * ldc;
            for (ptrdiff_t i = j; i < n; ++i) cj[i] = 0.0L;
        }
    } else {
        for (ptrdiff_t j = 0; j < n; ++j) {
            T *cj = c + j * ldc;
            for (ptrdiff_t i = j; i < n; ++i) cj[i] *= beta;
        }
    }
}

/* ── Diagonal-aware writeback kernel ───────────────────────────────────
 * Faithful port of OpenBLAS driver/level3/syrk_kernel.c (the openblas
 * overlay's eblas_esyrk_kernel_{u,l}), with eblas_egemm_kernel → the shared
 * etri_gemm_kernel. The strict-triangle rectangular remainders are full
 * GEMMs into C; each diagonal NR×NR sub-block is GEMMed into a zeroed
 * subbuffer (etri_gemm_kernel accumulates), then only its UPLO triangle is
 * merged into C. Subbuffer sized NR*(NR+1) to match OpenBLAS's safety pad. */
void esyrk_kernel_u(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, T alpha,
                    const T *a, const T *b,
                    T *c, ptrdiff_t ldc, ptrdiff_t offset)
{
    T subbuf[NR * (NR + 1)];

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

    /* Diagonal walk in NR-step blocks. offset == 0, m == n here. */
    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        /* Strict-upper portion: rows 0..loop-1 × cols loop..loop+nn-1. */
        if (loop > 0) {
            etri_gemm_kernel(loop, nn, k, alpha,
                             a, b + loop * k, c + loop * ldc, ldc);
        }

        /* Diagonal block via subbuffer. */
        for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0.0L;
        etri_gemm_kernel(nn, nn, k, alpha,
                         a + loop * k, b + loop * k, subbuf, nn);

        T *cc = c + loop + loop * ldc;
        const T *ss = subbuf;
        for (ptrdiff_t j = 0; j < nn; ++j) {
            for (ptrdiff_t i = 0; i <= j; ++i) cc[i] += ss[i];
            ss += nn;
            cc += ldc;
        }
    }
}

void esyrk_kernel_l(ptrdiff_t m, ptrdiff_t n, ptrdiff_t k, T alpha,
                    const T *a, const T *b,
                    T *c, ptrdiff_t ldc, ptrdiff_t offset)
{
    T subbuf[NR * (NR + 1)];

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

    /* Diagonal walk in NR-step blocks. offset == 0, m == n here. */
    for (ptrdiff_t loop = 0; loop < n; loop += NR) {
        ptrdiff_t mm = loop;
        ptrdiff_t nn = (n - loop < (ptrdiff_t)NR) ? (n - loop) : (ptrdiff_t)NR;

        /* Diagonal block via subbuffer. */
        for (ptrdiff_t z = 0; z < nn * nn; ++z) subbuf[z] = 0.0L;
        etri_gemm_kernel(nn, nn, k, alpha,
                         a + loop * k, b + loop * k, subbuf, nn);

        T *cc = c + loop + loop * ldc;
        const T *ss = subbuf;
        for (ptrdiff_t j = 0; j < nn; ++j) {
            for (ptrdiff_t i = j; i < nn; ++i) cc[i] += ss[i];
            ss += nn;
            cc += ldc;
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
 * Faithful port of OpenBLAS level3_syrk.c with a single thread spanning the
 * whole output (rows ∈ [0, N]). The js-band UPLO clip bounds the active row
 * range so the kernel only ever writes the requested triangle. */
void esyrk_serial(
    char uplo, char trans,
    ptrdiff_t N, ptrdiff_t K,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    const T *beta_,
    T *c, ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO  = blas_up(uplo);
    const char TRANS = blas_up(trans);

    if (N <= 0) return;

    if (UPLO == 'U') esyrk_beta_u((ptrdiff_t)N, beta, c, (ptrdiff_t)ldc);
    else             esyrk_beta_l((ptrdiff_t)N, beta, c, (ptrdiff_t)ldc);

    if (K == 0 || alpha == 0.0L) return;

    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(K, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)egemm_round_up(NC, NR) * sizeof(T);
    T *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (Ap && Bp) {
        for (ptrdiff_t js = 0; js < N; js += NC) {
            const ptrdiff_t jb = (N - js < NC) ? (N - js) : NC;

            /* UPLO clip of the [0, N] row range for this js-band:
             *   UPPER: only rows up to js+jb contribute.
             *   LOWER: only rows from js onwards. */
            ptrdiff_t m_lo_eff = (UPLO == 'L') ? js : 0;
            ptrdiff_t m_hi_eff = (UPLO == 'U' && N > js + jb) ? (js + jb) : N;
            if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);

            for (ptrdiff_t ls = 0; ls < K; ls += KC) {
                const ptrdiff_t pb = (K - ls < KC) ? (K - ls) : KC;

                /* Pack Bp = the same A in OCOPY shape (A doubles as B). */
                if (TRANS == 'N')
                    etri_tcopy(pb, jb, &a[(size_t)ls * lda + js], lda, Bp);
                else
                    etri_ncopy(pb, jb, &a[(size_t)js * lda + ls], lda, Bp);

                for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                    const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (TRANS == 'N')
                        etri_tcopy(pb, min_i, &a[(size_t)ls * lda + is], lda, Ap);
                    else
                        etri_ncopy(pb, min_i, &a[(size_t)is * lda + ls], lda, Ap);

                    if (UPLO == 'U')
                        esyrk_kernel_u(min_i, jb, pb, alpha, Ap, Bp,
                                       &c[(size_t)js * ldc + is], ldc,
                                       (ptrdiff_t)(is - js));
                    else
                        esyrk_kernel_l(min_i, jb, pb, alpha, Ap, Bp,
                                       &c[(size_t)js * ldc + is], ldc,
                                       (ptrdiff_t)(is - js));
                }
            }
        }
    }
    free(Ap);
    free(Bp);
}
