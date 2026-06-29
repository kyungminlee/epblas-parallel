/*
 * xherk_serial.c — kind16 complex (__complex128) Hermitian rank-k update,
 * single-thread. The fused serial packed-GEMM driver plus the internal serial
 * entry `xherk_serial`, called by xherk_ as its OOM fallback / nesting
 * delegate.
 *
 *   C := alpha·A·Aᴴ + beta·C   (trans='N', A is N×K)
 *   C := alpha·Aᴴ·A + beta·C   (trans='C', A is K×N)
 *
 * alpha and beta are REAL, C is HERMITIAN — only the UPLO triangle is touched
 * and the diagonal stays real on output.
 *
 * Faithful port of OpenBLAS ZHERK, as the SINGLE-PRODUCT special case of the
 * xher2k packed-GEMM nest: one packed GEMM whose output is clipped to the UPLO
 * triangle, run ONCE per (is,js) tile (Bp packed from the same A). The
 * diagonal-aware kernel (qblas_xherk_kernel_{u,l}) merges the diagonal NR×NR
 * block singly and realifies the true diagonal element (Hermitian contract).
 * Conjugation is absorbed at pack time: TRANS='N' conjugates the Bp side,
 * TRANS='C' conjugates the Ap side. The triangular β pre-pass
 * (qblas_xherk_beta_{u,l}, real beta) and the diagonal kernel both live in the
 * shared complex L3 substrate (xl3_complex.h). alpha is real, so there is no
 * conj(alpha) second pass.
 *
 * Arrays are interleaved (re,im) __float128; ld-args, k, n and offset are in
 * COMPLEX elements, so every pointer step is ×2. Calling only the *serial*
 * substrate primitives keeps this path free of any nested OpenMP team, so it is
 * safe inside another routine's parallel region.
 */

#include "xherk_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "xl3_complex.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>

typedef xherk_TC TC;
typedef xherk_TR TR;

#define MR QBLAS_XGEMM_MR
#define NR QBLAS_XGEMM_NR


void xherk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TR *alpha_,
    const TC *a_c, ptrdiff_t lda,
    const TR *beta_,
    TC *c_c, ptrdiff_t ldc)
{
    /* Reinterpret the complex ABI as interleaved (re,im) __float128 storage. */
    const TR *a = (const TR *)a_c;
    TR *c = (TR *)c_c;
    const TR alphar = *alpha_;       /* real alpha; imag is identically 0 */
    const TR beta_r = *beta_;
    const char UPLO  = blas_up(uplo);
    const char TRANS = blas_up(trans);

    if (n <= 0) return;

    if (UPLO == 'U') qblas_xherk_beta_u(n, beta_r, c, ldc);
    else             qblas_xherk_beta_l(n, beta_r, c, ldc);

    if (k == 0 || alphar == 0.0Q) return;

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

    /* Conjugation absorbed at pack time (upstream GEMM_KERNEL_R/_L choice):
     *   TRANS='N' → conjugate Bp;  TRANS='C' → conjugate Ap. */
    const bool conj_a_pack = (TRANS == 'C') ? 1 : 0;
    const bool conj_b_pack = (TRANS == 'N') ? 1 : 0;

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
                    qblas_xgemm_tcopy(pb, jb, conj_b_pack, &a[((size_t)ls * lda + js) * 2], lda, Bp);
                else
                    qblas_xgemm_ncopy(pb, jb, conj_b_pack, &a[((size_t)js * lda + ls) * 2], lda, Bp);

                for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                    const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (TRANS == 'N')
                        qblas_xgemm_tcopy(pb, min_i, conj_a_pack, &a[((size_t)ls * lda + is) * 2], lda, Ap);
                    else
                        qblas_xgemm_ncopy(pb, min_i, conj_a_pack, &a[((size_t)is * lda + ls) * 2], lda, Ap);

                    TR *cij = &c[((size_t)js * ldc + is) * 2];
                    const ptrdiff_t off = is - js;

                    /* Single pass: alpha·A·Aᴴ + single-add Hermitian merge. */
                    if (UPLO == 'U')
                        qblas_xherk_kernel_u(min_i, jb, pb, alphar, 0.0Q, Ap, Bp, cij, ldc, off);
                    else
                        qblas_xherk_kernel_l(min_i, jb, pb, alphar, 0.0Q, Ap, Bp, cij, ldc, off);
                }
            }
        }
    }
    free(Ap);
    free(Bp);
}
