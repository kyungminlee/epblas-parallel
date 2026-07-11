/*
 * yherk — kind10 (COMPLEX(KIND=10)) Hermitian rank-k update.
 *
 *   C := alpha * A * A^H + beta * C    (trans='N', A is N×K)
 *   C := alpha * A^H * A + beta * C    (trans='C', A is K×N)
 *
 * alpha and beta are REAL(KIND=10) scalars. C is Hermitian: only the
 * UPLO triangle is read or written, and the diagonal stays real on
 * output (Hermitian C contract).
 *
 * Port source: OpenBLAS.
 *   - interface/syrk.c            (Z-variant, built with HEMM macro:
 *                                  ERROR_NAME "ZHERK ", TRANS ∈ {N,C},
 *                                  same arg-validation as ZSYRK)
 *   - driver/level3/zherk_k.c     (level3_syrk wrapper — same blocking
 *                                  nest as syrk_k.c with the kernel
 *                                  swapped for zherk_kernel: diag-imag
 *                                  forced to zero)
 *   - driver/level3/zherk_kernel.c
 *                                → eblas_yherk_kernel_{u,l}
 *                                  (in common/eblas_l3_complex.c)
 *   - driver/level3/zherk_beta.c → eblas_yherk_beta_{u,l}
 *                                  (in common/eblas_l3_complex.c)
 *
 * Conjugation: upstream zherk_kernel.c picks GEMM_KERNEL_R (conjugate
 * Bp) for TRANS='N' and GEMM_KERNEL_L (conjugate Ap) for TRANS='C'.
 * Our shared eblas_ygemm_kernel is NN-only (matching ygemm/ysymm —
 * conjugation absorbed at pack time), so we set `conj` on the side the
 * upstream kernel would have conjugated:
 *   - TRANS='N': Bp packer uses conj=1 (caller of yherk path passes A
 *                in both ICOPY and OCOPY roles, so the Bp-side pack of
 *                A turns into A^H during multiply)
 *   - TRANS='C': Ap packer uses conj=1
 *
 * Both packers in level3_syrk.c use the same default ICOPY/OCOPY
 * macros from common.h (no HERK-specific override) — the only HERK
 * specialization is the kernel and the beta function.
 *
 * Fortran ABI:
 *   subroutine yherk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
 *   - alpha, beta are REAL(KIND=10): one long double each
 *   - a, c are COMPLEX(KIND=10) arrays (interleaved re,im)
 *   - lda, ldc are in COMPLEX(KIND=10) elements
 */

#include "eblas_l3_complex.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#define MR EBLAS_YGEMM_MR
#define NR EBLAS_YGEMM_NR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


void yherk_(
    const char *uplo_p, const char *trans_p,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_)
{
    const int N = *n_, K = *k_;
    const T alphar = *alpha_;
    const T beta_r = *beta_;
    const int lda = *lda_, ldc = *ldc_;
    const int uplo  = (char)toupper((unsigned char)*uplo_p);
    const int trans = (char)toupper((unsigned char)*trans_p);

    if (N <= 0) return;

    if (uplo == 'U') eblas_yherk_beta_u((ptrdiff_t)N, beta_r,
                                        c, (ptrdiff_t)ldc);
    else             eblas_yherk_beta_l((ptrdiff_t)N, beta_r,
                                        c, (ptrdiff_t)ldc);

    if (K == 0 || alphar == 0.0L) return;

    int MC0, KC, NC;
    eblas_ygemm_blocks(&MC0, &KC, &NC);

    int MC = MC0;
    if (K <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;
        long target_mc = L2_TARGET_BYTES / ((long)K * 2L * (long)sizeof(T));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((int)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * 2 * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * 2 * sizeof(T);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
#endif

    long nnk = (long)N * (long)N * (long)K;
    if (nnk < 64L * 64L * 64L) nthreads = 1;

    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) return;

    T **Ap_arr = calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_arr) { free(Bp); return; }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
        free(Ap_arr); free(Bp);
        return;
    }

    const int conj_a = (trans == 'C') ? 1 : 0;
    const int conj_b = (trans == 'N') ? 1 : 0;

#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
#else
        int tid = 0, nth = 1;
#endif
        T *Ap = Ap_arr[tid];

        int m_chunk = round_up((N + nth - 1) / nth, MR);
        int m_lo = tid * m_chunk;
        int m_hi = m_lo + m_chunk;
        if (m_hi > N) m_hi = N;

        for (int js = 0; js < N; js += NC) {
            int jb = (N - js < NC) ? (N - js) : NC;

            int m_lo_eff = (uplo == 'L' && m_lo < js) ? js : m_lo;
            int m_hi_eff = (uplo == 'U' && m_hi > js + jb) ? (js + jb) : m_hi;
            if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);
            if (m_lo_eff < m_lo) m_lo_eff = m_lo;

            for (int ls = 0; ls < K; ls += KC) {
                int pb = (K - ls < KC) ? (K - ls) : KC;

#ifdef _OPENMP
                #pragma omp barrier
                #pragma omp single
#endif
                {
                    /* B-pack source is always A (HERK uses A twice).
                     * Same layout choice as SYRK: TRANS='N' → tcopy
                     * (A at (row=js, col=ls)); TRANS='C' → ncopy
                     * (A at (row=ls, col=js)). Conjugation set per
                     * the GEMM_KERNEL_R/L swap explained at the top. */
                    if (trans == 'N') {
                        eblas_ygemm_tcopy(pb, jb, conj_b,
                            &a[((size_t)ls * lda + js) * 2], lda, Bp);
                    } else {
                        eblas_ygemm_ncopy(pb, jb, conj_b,
                            &a[((size_t)js * lda + ls) * 2], lda, Bp);
                    }
                }

                for (int is = m_lo_eff; is < m_hi_eff; is += MC) {
                    int min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (trans == 'N') {
                        eblas_ygemm_tcopy(pb, min_i, conj_a,
                            &a[((size_t)ls * lda + is) * 2], lda, Ap);
                    } else {
                        eblas_ygemm_ncopy(pb, min_i, conj_a,
                            &a[((size_t)is * lda + ls) * 2], lda, Ap);
                    }

                    if (uplo == 'U') {
                        eblas_yherk_kernel_u(min_i, jb, pb, alphar,
                            Ap, Bp,
                            &c[((size_t)js * ldc + is) * 2], ldc,
                            (ptrdiff_t)(is - js));
                    } else {
                        eblas_yherk_kernel_l(min_i, jb, pb, alphar,
                            Ap, Bp,
                            &c[((size_t)js * ldc + is) * 2], ldc,
                            (ptrdiff_t)(is - js));
                    }
                }
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}
