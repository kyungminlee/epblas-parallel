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
 * pre-pass (qblas_xherk_beta_{u,l}, real beta, unconditionally clears diag
 * imag) and the diagonal-aware kernel (qblas_xher2k_kernel_{u,l}) both live in
 * the shared complex L3 substrate (xl3_complex.h).
 *
 * Arrays are interleaved (re,im) __float128; ld-args, k, n and offset are in COMPLEX
 * elements, so every pointer step is ×2. Calling only the *serial* substrate
 * primitives keeps this path free of any nested OpenMP team, so it is safe
 * inside another routine's parallel region.
 */

#include "xher2k_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "xl3_complex.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>

typedef xher2k_TC TC;
typedef xher2k_TR TR;

#define MR QBLAS_XGEMM_MR
#define NR QBLAS_XGEMM_NR


void xher2k_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TC *alpha_c,
    const TC *a_c, ptrdiff_t lda,
    const TC *b_c, ptrdiff_t ldb,
    const TR *beta_,
    TC *c_c, ptrdiff_t ldc)
{
    /* Reinterpret the complex ABI as interleaved (re,im) __float128 storage. */
    const TR *alpha_ = (const TR *)alpha_c;
    const TR *a = (const TR *)a_c;
    const TR *b = (const TR *)b_c;
    TR *c = (TR *)c_c;
    const TR alphar = alpha_[0], alphai = alpha_[1];
    const TR beta_r = beta_[0];
    const char UPLO  = blas_up(uplo);
    const char TRANS = blas_up(trans);

    if (n <= 0) return;

    if (UPLO == 'U') qblas_xherk_beta_u(n, beta_r, c, ldc);
    else             qblas_xherk_beta_l(n, beta_r, c, ldc);

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

    /* Conjugation absorbed at pack time (upstream GEMM_KERNEL_R/_L choice):
     *   TRANS='N' → conjugate Bp;  TRANS='C' → conjugate Ap. */
    const bool conj_a_pack = (TRANS == 'C') ? 1 : 0;
    const bool conj_b_pack = (TRANS == 'N') ? 1 : 0;

    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * (size_t)KC * 2 * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * (size_t)blas_round_up(NC, NR) * 2 * sizeof(TR);
    /* Persistent grow-only thread-local pack arena (Ap_A|Ap_B|Bp_A|Bp_B in one
     * block): a per-call aligned_alloc+free of these mmap-threshold-sized
     * buffers trips glibc's trim heuristic and re-faults every touched page
     * each call — a pure page-fault tax at small N (see etrsm_serial.c). */
    static __thread TR *g_pack = NULL;
    static __thread size_t g_pack_cap = 0;
    const size_t ap_al = (ap_bytes + 63) & ~(size_t)63;
    const size_t bp_al = (bp_bytes + 63) & ~(size_t)63;
    const size_t need  = 2 * ap_al + 2 * bp_al;
    if (need > g_pack_cap) {
        free(g_pack);
        size_t cap = need + (need >> 1);            /* 1.5× headroom to amortize regrow */
        cap = (cap + 63) & ~(size_t)63;
        g_pack = aligned_alloc(64, cap);
        g_pack_cap = g_pack ? cap : 0;
    }
    if (g_pack) {
        TR *Ap_A = g_pack;
        TR *Ap_B = (TR *)(void *)((char *)g_pack + ap_al);
        TR *Bp_A = (TR *)(void *)((char *)g_pack + 2 * ap_al);
        TR *Bp_B = (TR *)(void *)((char *)g_pack + 2 * ap_al + bp_al);
        for (ptrdiff_t js = 0; js < n; js += NC) {
            const ptrdiff_t jb = (n - js < NC) ? (n - js) : NC;

            /* UPLO clip of the [0, N] row range for this js-band. */
            ptrdiff_t m_lo_eff = (UPLO == 'L') ? js : 0;
            ptrdiff_t m_hi_eff = (UPLO == 'U' && n > js + jb) ? (js + jb) : n;
            if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);

            for (ptrdiff_t ls = 0; ls < k; ls += KC) {
                const ptrdiff_t pb = (k - ls < KC) ? (k - ls) : KC;

                if (TRANS == 'N') {
                    qblas_xgemm_tcopy(pb, jb, conj_b_pack, &a[((size_t)ls * lda + js) * 2], lda, Bp_A);
                    qblas_xgemm_tcopy(pb, jb, conj_b_pack, &b[((size_t)ls * ldb + js) * 2], ldb, Bp_B);
                } else {
                    qblas_xgemm_ncopy(pb, jb, conj_b_pack, &a[((size_t)js * lda + ls) * 2], lda, Bp_A);
                    qblas_xgemm_ncopy(pb, jb, conj_b_pack, &b[((size_t)js * ldb + ls) * 2], ldb, Bp_B);
                }

                for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                    const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (TRANS == 'N') {
                        qblas_xgemm_tcopy(pb, min_i, conj_a_pack, &a[((size_t)ls * lda + is) * 2], lda, Ap_A);
                        qblas_xgemm_tcopy(pb, min_i, conj_a_pack, &b[((size_t)ls * ldb + is) * 2], ldb, Ap_B);
                    } else {
                        qblas_xgemm_ncopy(pb, min_i, conj_a_pack, &a[((size_t)is * lda + ls) * 2], lda, Ap_A);
                        qblas_xgemm_ncopy(pb, min_i, conj_a_pack, &b[((size_t)is * ldb + ls) * 2], ldb, Ap_B);
                    }

                    TR *cij = &c[((size_t)js * ldc + is) * 2];
                    const ptrdiff_t off = is - js;

                    /* Pass 1: alpha·A·Bᴴ + Hermitian diagonal merge. */
                    if (UPLO == 'U')
                        qblas_xher2k_kernel_u(min_i, jb, pb, alphar, alphai, Ap_A, Bp_B, cij, ldc, off, 1);
                    else
                        qblas_xher2k_kernel_l(min_i, jb, pb, alphar, alphai, Ap_A, Bp_B, cij, ldc, off, 1);

                    /* Pass 2: conj(alpha)·B·Aᴴ into the off-diagonal strips. */
                    if (UPLO == 'U')
                        qblas_xher2k_kernel_u(min_i, jb, pb, alphar, -alphai, Ap_B, Bp_A, cij, ldc, off, 0);
                    else
                        qblas_xher2k_kernel_l(min_i, jb, pb, alphar, -alphai, Ap_B, Bp_A, cij, ldc, off, 0);
                }
            }
        }
    }
}
