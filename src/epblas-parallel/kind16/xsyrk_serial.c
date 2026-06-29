/*
 * xsyrk_serial.c — kind16 complex (__complex128) symmetric rank-k update,
 * single-thread. The fused serial packed-GEMM driver plus the internal serial
 * entry `xsyrk_serial`, called by xsyrk_ as its OOM fallback / nesting
 * delegate.
 *
 *   C := alpha·A·Aᵀ + beta·C   (trans='N', A is N×K)
 *   C := alpha·Aᵀ·A + beta·C   (trans='T', A is K×N)
 *
 * Only the UPLO triangle of C is read or written.
 *
 * Faithful port of OpenBLAS ZSYRK, as the SINGLE-PRODUCT special case of the
 * xsyr2k packed-GEMM nest: one packed GEMM whose output is clipped to the UPLO
 * triangle, run ONCE per (is,js) tile (Bp packed from the same A). The
 * diagonal-aware kernel (qblas_xsyrk_kernel_{u,l}) merges the diagonal NR×NR
 * block SINGLY — a single product A·Aᵀ is already symmetric on the diagonal, so
 * adding the transpose half (as syr2k does) would double it. The triangular β
 * pre-pass (qblas_xsyrk_beta_{u,l}) and the diagonal kernel both live in the
 * shared complex L3 substrate (xl3_complex.h); no conjugation, so the packers
 * are called with conj = 0.
 *
 * Arrays are interleaved (re,im) __float128; ld-args, k, n and offset are in
 * COMPLEX elements, so every pointer step is ×2. Calling only the *serial*
 * substrate primitives keeps this path free of any nested OpenMP team, so it is
 * safe inside another routine's parallel region.
 */

#include "xsyrk_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "xl3_complex.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>

typedef xsyrk_TC TC;
typedef xsyrk_TR TR;

#define MR QBLAS_XGEMM_MR
#define NR QBLAS_XGEMM_NR


void xsyrk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TC *alpha_c,
    const TC *a_c, ptrdiff_t lda,
    const TC *beta_c,
    TC *c_c, ptrdiff_t ldc)
{
    /* Reinterpret the complex ABI as interleaved (re,im) __float128 storage. */
    const TR *alpha_ = (const TR *)alpha_c;
    const TR *a = (const TR *)a_c;
    const TR *beta_ = (const TR *)beta_c;
    TR *c = (TR *)c_c;
    const TR alphar = alpha_[0], alphai = alpha_[1];
    const TR beta_r = beta_[0],  beta_i = beta_[1];
    const char UPLO  = blas_up(uplo);
    const char TRANS = blas_up(trans);

    if (n <= 0) return;

    if (UPLO == 'U') qblas_xsyrk_beta_u(n, beta_r, beta_i, c, ldc);
    else             qblas_xsyrk_beta_l(n, beta_r, beta_i, c, ldc);

    if (k == 0 || (alphar == 0.0Q && alphai == 0.0Q)) return;

    ptrdiff_t MC0, KC0, NC0;
    qblas_xgemm_blocks(&MC0, &KC0, &NC0);
    ptrdiff_t MC = MC0, KC = KC0, NC = NC0;

    /* Grow MC toward an L2-sized panel when K is small (complex doubles the
     * per-element footprint), capped at 4×MC0 and rounded to MR. */
    if (k <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;
        long target_mc = L2_TARGET_BYTES / ((long)k * 2L * (long)sizeof(TR));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = blas_round_up((ptrdiff_t)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * (size_t)KC * 2 * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * (size_t)blas_round_up(NC, NR) * 2 * sizeof(TR);
    TR *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    TR *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (Ap && Bp) {
        for (ptrdiff_t js = 0; js < n; js += NC) {
            const ptrdiff_t jb = (n - js < NC) ? (n - js) : NC;

            /* UPLO clip of the [0, N] row range for this js-band. */
            ptrdiff_t m_lo_eff = (UPLO == 'L') ? js : 0;
            ptrdiff_t m_hi_eff = (UPLO == 'U' && n > js + jb) ? (js + jb) : n;
            if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);

            for (ptrdiff_t ls = 0; ls < k; ls += KC) {
                const ptrdiff_t pb = (k - ls < KC) ? (k - ls) : KC;

                if (TRANS == 'N')
                    qblas_xgemm_tcopy(pb, jb, 0, &a[((size_t)ls * lda + js) * 2], lda, Bp);
                else
                    qblas_xgemm_ncopy(pb, jb, 0, &a[((size_t)js * lda + ls) * 2], lda, Bp);

                for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                    const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (TRANS == 'N')
                        qblas_xgemm_tcopy(pb, min_i, 0, &a[((size_t)ls * lda + is) * 2], lda, Ap);
                    else
                        qblas_xgemm_ncopy(pb, min_i, 0, &a[((size_t)is * lda + ls) * 2], lda, Ap);

                    TR *cij = &c[((size_t)js * ldc + is) * 2];
                    const ptrdiff_t off = is - js;

                    /* Single pass: alpha·A·Aᵀ + single-add diagonal merge. */
                    if (UPLO == 'U')
                        qblas_xsyrk_kernel_u(min_i, jb, pb, alphar, alphai, Ap, Bp, cij, ldc, off);
                    else
                        qblas_xsyrk_kernel_l(min_i, jb, pb, alphar, alphai, Ap, Bp, cij, ldc, off);
                }
            }
        }
    }
    free(Ap);
    free(Bp);
}
