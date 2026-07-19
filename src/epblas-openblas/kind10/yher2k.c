/*
 * yher2k — kind10 (COMPLEX(KIND=10)) Hermitian rank-2k update.
 *
 *   C := alpha*A*B^H + conj(alpha)*B*A^H + beta*C   (trans='N', A,B are N×K)
 *   C := alpha*A^H*B + conj(alpha)*B^H*A + beta*C   (trans='C', A,B are K×N)
 *
 * alpha is COMPLEX(KIND=10); beta is REAL(KIND=10). C is Hermitian:
 * only the UPLO triangle is read/written, and the diagonal stays real
 * on output.
 *
 * Port source: OpenBLAS.
 *   - interface/syr2k.c (built with HEMM macro:
 *                        ERROR_NAME "ZHER2K", TRANS ∈ {N,C},
 *                        real beta, complex alpha)
 *   - driver/level3/zher2k_k.c       (level3_syr2k wrapper —
 *                                     KERNEL_OPERATION_C negates
 *                                     alpha[1] for the second pass so
 *                                     the symmetric mirror picks up
 *                                     conj(alpha)·B·A^H)
 *   - driver/level3/zher2k_kernel.c → eblas_yher2k_kernel_{u,l}
 *                                     (in common/eblas_l3_complex.c)
 *   - driver/level3/syrk_beta (HER2K variant): same shape as
 *                                     zherk_beta.c (real beta,
 *                                     unconditional diag imag = 0),
 *                                     so we reuse eblas_yherk_beta_{u,l}
 *                                     directly.
 *
 * Two-pass per (jc, pc) tile (per level3_syr2k.c):
 *   pass 1 packs (Ap=A, Bp=B), kernel with alpha       , flag=1 on diag
 *   pass 2 packs (Ap=B, Bp=A), kernel with conj(alpha) , flag=0 on diag
 *
 * The diagonal mirror lives inside eblas_yher2k_kernel — it computes
 * alpha·A·B^H once into a subbuf and reconstructs the matching
 * conj(alpha)·B·A^H via subbuf[j,i] flipped, which is why pass 2 sets
 * flag=0 (would otherwise double-count the diagonal).
 *
 * Conjugation: upstream zher2k_kernel.c picks GEMM_KERNEL_R (conj Bp)
 * for TRANS='N' and GEMM_KERNEL_L (conj Ap) for TRANS='C'. Our shared
 * NN kernel absorbs conjugation at pack time:
 *   - TRANS='N': pack the Bp side (which holds B in pass 1, A in pass 2)
 *                with conj=1; Ap side with conj=0.
 *   - TRANS='C': pack the Ap side (A in pass 1, B in pass 2) with conj=1;
 *                Bp side with conj=0.
 *
 * Fortran ABI:
 *   subroutine yher2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
 *   - alpha is COMPLEX(KIND=10): 2 long doubles (re, im)
 *   - beta  is REAL(KIND=10): 1 long double
 *   - a, b, c are COMPLEX(KIND=10) arrays (interleaved re, im)
 *   - lda, ldb, ldc are in COMPLEX(KIND=10) elements
 */

#include "eblas_l3_complex.h"
#include "eblas_tuning.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#define MR EBLAS_YGEMM_MR
#define NR EBLAS_YGEMM_NR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


void yher2k_(
    const char *uplo_p, const char *trans_p,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_)
{
    const int N = *n_, K = *k_;
    const T alphar = alpha_[0], alphai = alpha_[1];
    const T beta_r = *beta_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const int uplo  = (char)toupper((unsigned char)*uplo_p);
    const int trans = (char)toupper((unsigned char)*trans_p);

    if (N <= 0) return;

    /* K==0 / alpha==0: the triangular beta pass on the UPLO triangle
     * of C is the entire operation. */
    if (K == 0 || (alphar == 0.0L && alphai == 0.0L)) {
        if (uplo == 'U') eblas_yherk_beta_u((ptrdiff_t)N, beta_r,
                                            c, (ptrdiff_t)ldc);
        else             eblas_yherk_beta_l((ptrdiff_t)N, beta_r,
                                            c, (ptrdiff_t)ldc);
        return;
    }

    int MC0, KC, NC;
    eblas_ygemm_blocks(&MC0, &KC, &NC);

    int MC = MC0;
    if (K <= KC) {
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

    /* All pack allocations happen BEFORE the in-place beta pass on C so
     * a failure can never leave C scaled but not updated; benchmark-
     * reference policy is to abort() loudly on failure.
     *
     * Two pre-packed Bp panels per (jc, pc) tile: Bp_B holds B (used in
     * pass 1 as the Bp role) and Bp_A holds A (used in pass 2 as the Bp
     * role). Each thread keeps its own pair of Ap workspaces — Ap_A
     * (used in pass 1) and Ap_B (used in pass 2). */
    T *Bp_A = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    T *Bp_B = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp_A || !Bp_B) {
        free(Bp_A); free(Bp_B);
        fprintf(stderr, "yher2k: pack buffer allocation failed\n");
        abort();
    }

    T **Ap_A_arr = calloc((size_t)nthreads, sizeof(T *));
    T **Ap_B_arr = calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_A_arr || !Ap_B_arr) {
        free(Ap_A_arr); free(Ap_B_arr); free(Bp_A); free(Bp_B);
        fprintf(stderr, "yher2k: pack buffer allocation failed\n");
        abort();
    }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_A_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        Ap_B_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (!Ap_A_arr[t] || !Ap_B_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) {
            free(Ap_A_arr[t]); free(Ap_B_arr[t]);
        }
        free(Ap_A_arr); free(Ap_B_arr); free(Bp_A); free(Bp_B);
        fprintf(stderr, "yher2k: pack buffer allocation failed\n");
        abort();
    }

    /* Triangular beta pre-pass on the UPLO triangle of C only —
     * after every allocation has succeeded. */
    if (uplo == 'U') eblas_yherk_beta_u((ptrdiff_t)N, beta_r,
                                        c, (ptrdiff_t)ldc);
    else             eblas_yherk_beta_l((ptrdiff_t)N, beta_r,
                                        c, (ptrdiff_t)ldc);

    /* Per upstream's KERNEL_R/L choice (see header for derivation):
     *   TRANS='N' (!CONJ) → GEMM_KERNEL_R → conjugate Bp
     *   TRANS='C' ( CONJ) → GEMM_KERNEL_L → conjugate Ap
     * Same conj flag for both passes; only the A/B input swap differs. */
    const int conj_a_pack = (trans == 'C') ? 1 : 0;
    const int conj_b_pack = (trans == 'N') ? 1 : 0;

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
        T *Ap_A = Ap_A_arr[tid];
        T *Ap_B = Ap_B_arr[tid];

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
                    /* Pack both Bp panels (one from A, one from B) up
                     * front so both passes can reuse them. Same layout
                     * choice as SYR2K: TRANS='N' → tcopy; TRANS='C' →
                     * ncopy. Conj per the per-trans assignment above. */
                    if (trans == 'N') {
                        eblas_ygemm_tcopy(pb, jb, conj_b_pack,
                            &a[((size_t)ls * lda + js) * 2], lda, Bp_A);
                        eblas_ygemm_tcopy(pb, jb, conj_b_pack,
                            &b[((size_t)ls * ldb + js) * 2], ldb, Bp_B);
                    } else {
                        eblas_ygemm_ncopy(pb, jb, conj_b_pack,
                            &a[((size_t)js * lda + ls) * 2], lda, Bp_A);
                        eblas_ygemm_ncopy(pb, jb, conj_b_pack,
                            &b[((size_t)js * ldb + ls) * 2], ldb, Bp_B);
                    }
                }

                for (int is = m_lo_eff; is < m_hi_eff; is += MC) {
                    int min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (trans == 'N') {
                        eblas_ygemm_tcopy(pb, min_i, conj_a_pack,
                            &a[((size_t)ls * lda + is) * 2], lda, Ap_A);
                        eblas_ygemm_tcopy(pb, min_i, conj_a_pack,
                            &b[((size_t)ls * ldb + is) * 2], ldb, Ap_B);
                    } else {
                        eblas_ygemm_ncopy(pb, min_i, conj_a_pack,
                            &a[((size_t)is * lda + ls) * 2], lda, Ap_A);
                        eblas_ygemm_ncopy(pb, min_i, conj_a_pack,
                            &b[((size_t)is * ldb + ls) * 2], ldb, Ap_B);
                    }

                    if (uplo == 'U') {
                        eblas_yher2k_kernel_u(min_i, jb, pb, alphar, alphai,
                            Ap_A, Bp_B,
                            &c[((size_t)js * ldc + is) * 2], ldc,
                            (ptrdiff_t)(is - js), 1);
                        eblas_yher2k_kernel_u(min_i, jb, pb, alphar, -alphai,
                            Ap_B, Bp_A,
                            &c[((size_t)js * ldc + is) * 2], ldc,
                            (ptrdiff_t)(is - js), 0);
                    } else {
                        eblas_yher2k_kernel_l(min_i, jb, pb, alphar, alphai,
                            Ap_A, Bp_B,
                            &c[((size_t)js * ldc + is) * 2], ldc,
                            (ptrdiff_t)(is - js), 1);
                        eblas_yher2k_kernel_l(min_i, jb, pb, alphar, -alphai,
                            Ap_B, Bp_A,
                            &c[((size_t)js * ldc + is) * 2], ldc,
                            (ptrdiff_t)(is - js), 0);
                    }
                }
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) {
        free(Ap_A_arr[t]); free(Ap_B_arr[t]);
    }
    free(Ap_A_arr); free(Ap_B_arr);
    free(Bp_A); free(Bp_B);
}
