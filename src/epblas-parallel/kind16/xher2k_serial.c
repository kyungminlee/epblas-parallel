/*
 * xher2k_serial.c — kind16 complex (__complex128) Hermitian rank-2k update,
 * single-thread. The fused serial packed-GEMM driver plus the internal serial
 * entry `xher2k_serial`, called by xher2k_ as its OOM fallback / nesting
 * delegate.
 *
 *   C := alpha·A·Bᴴ + conj(alpha)·B·Aᴴ + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·Aᴴ·B + conj(alpha)·Bᴴ·A + beta·C   (trans='C', A,B are K×N)
 *
 * alpha is COMPLEX, beta is REAL, C is HERMITIAN — only the UPLO triangle is
 * touched and the diagonal stays real on output.
 *
 * Faithful port of OpenBLAS ZHER2K: one packed GEMM nest whose output is
 * clipped to the UPLO triangle, run TWICE per (is,js) tile. Pass 1
 * (Ap=A, Bp=B, alpha, flag=1) folds A·Bᴴ plus both diagonal halves via the
 * Hermitian subbuffer merge; pass 2 (Ap=B, Bp=A, conj(alpha) via −alphai,
 * flag=0) adds the off-diagonal B·Aᴴ strips. Conjugation is absorbed at pack
 * time: TRANS='N' conjugates the Bp side, TRANS='C' conjugates the Ap side
 * (matching upstream's GEMM_KERNEL_R / _L selection). The triangular β
 * pre-pass (qblas_yherk_beta_{u,l}, real beta, unconditionally clears diag
 * imag) and the diagonal-aware kernel (qblas_yher2k_kernel_{u,l}) both live in
 * the shared complex L3 substrate (xl3_complex.h).
 *
 * Arrays are interleaved (re,im) __float128; ld-args, k, n and offset are in COMPLEX
 * elements, so every pointer step is ×2. Calling only the *serial* substrate
 * primitives keeps this path free of any nested OpenMP team, so it is safe
 * inside another routine's parallel region.
 */

#include "xher2k_kernel.h"
#include "../common/blas_char.h"
#include "xl3_complex.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>

typedef __float128 T;

#define MR QBLAS_YGEMM_MR
#define NR QBLAS_YGEMM_NR

static ptrdiff_t round_up(ptrdiff_t v, ptrdiff_t m) { return ((v + m - 1) / m) * m; }

void xher2k_serial(
    char uplo_c, char trans_c,
    ptrdiff_t n, ptrdiff_t k,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    const T *beta_,
    T *c, ptrdiff_t ldc)
{
    const T alphar = alpha_[0], alphai = alpha_[1];
    const T beta_r = beta_[0];
    const char uplo  = blas_up(uplo_c);
    const char trans = blas_up(trans_c);

    if (n <= 0) return;

    if (uplo == 'U') qblas_yherk_beta_u(n, beta_r, c, ldc);
    else             qblas_yherk_beta_l(n, beta_r, c, ldc);

    if (k == 0 || (alphar == 0.0Q && alphai == 0.0Q)) return;

    ptrdiff_t MC0, KC0, NC0;
    qblas_ygemm_blocks(&MC0, &KC0, &NC0);
    ptrdiff_t MC = MC0, KC = KC0, NC = NC0;

    /* Grow MC toward an L2-sized panel when K is small (complex doubles the
     * per-element footprint), capped at 4×MC0 and rounded to MR. */
    if (k <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;
        long target_mc = L2_TARGET_BYTES / ((long)k * 2L * (long)sizeof(T));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((ptrdiff_t)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    /* Conjugation absorbed at pack time (upstream GEMM_KERNEL_R/_L choice):
     *   TRANS='N' → conjugate Bp;  TRANS='C' → conjugate Ap. */
    const bool conj_a_pack = (trans == 'C') ? 1 : 0;
    const bool conj_b_pack = (trans == 'N') ? 1 : 0;

    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * 2 * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * 2 * sizeof(T);
    T *Ap_A = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    T *Ap_B = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    T *Bp_A = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    T *Bp_B = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (Ap_A && Ap_B && Bp_A && Bp_B) {
        for (ptrdiff_t js = 0; js < n; js += NC) {
            const ptrdiff_t jb = (n - js < NC) ? (n - js) : NC;

            /* UPLO clip of the [0, N] row range for this js-band. */
            ptrdiff_t m_lo_eff = (uplo == 'L') ? js : 0;
            ptrdiff_t m_hi_eff = (uplo == 'U' && n > js + jb) ? (js + jb) : n;
            if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);

            for (ptrdiff_t ls = 0; ls < k; ls += KC) {
                const ptrdiff_t pb = (k - ls < KC) ? (k - ls) : KC;

                if (trans == 'N') {
                    qblas_ygemm_tcopy(pb, jb, conj_b_pack, &a[((size_t)ls * lda + js) * 2], lda, Bp_A);
                    qblas_ygemm_tcopy(pb, jb, conj_b_pack, &b[((size_t)ls * ldb + js) * 2], ldb, Bp_B);
                } else {
                    qblas_ygemm_ncopy(pb, jb, conj_b_pack, &a[((size_t)js * lda + ls) * 2], lda, Bp_A);
                    qblas_ygemm_ncopy(pb, jb, conj_b_pack, &b[((size_t)js * ldb + ls) * 2], ldb, Bp_B);
                }

                for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                    const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (trans == 'N') {
                        qblas_ygemm_tcopy(pb, min_i, conj_a_pack, &a[((size_t)ls * lda + is) * 2], lda, Ap_A);
                        qblas_ygemm_tcopy(pb, min_i, conj_a_pack, &b[((size_t)ls * ldb + is) * 2], ldb, Ap_B);
                    } else {
                        qblas_ygemm_ncopy(pb, min_i, conj_a_pack, &a[((size_t)is * lda + ls) * 2], lda, Ap_A);
                        qblas_ygemm_ncopy(pb, min_i, conj_a_pack, &b[((size_t)is * ldb + ls) * 2], ldb, Ap_B);
                    }

                    T *cij = &c[((size_t)js * ldc + is) * 2];
                    const ptrdiff_t off = is - js;

                    /* Pass 1: alpha·A·Bᴴ + Hermitian diagonal merge. */
                    if (uplo == 'U')
                        qblas_yher2k_kernel_u(min_i, jb, pb, alphar, alphai, Ap_A, Bp_B, cij, ldc, off, 1);
                    else
                        qblas_yher2k_kernel_l(min_i, jb, pb, alphar, alphai, Ap_A, Bp_B, cij, ldc, off, 1);

                    /* Pass 2: conj(alpha)·B·Aᴴ into the off-diagonal strips. */
                    if (uplo == 'U')
                        qblas_yher2k_kernel_u(min_i, jb, pb, alphar, -alphai, Ap_B, Bp_A, cij, ldc, off, 0);
                    else
                        qblas_yher2k_kernel_l(min_i, jb, pb, alphar, -alphai, Ap_B, Bp_A, cij, ldc, off, 0);
                }
            }
        }
    }
    free(Ap_A);
    free(Ap_B);
    free(Bp_A);
    free(Bp_B);
}
