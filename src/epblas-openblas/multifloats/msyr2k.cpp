/*
 * msyr2k — kind10 (REAL(KIND=10) / 80-bit multifloats::float64x2) port of OpenBLAS DSYR2K.
 *
 *   C := alpha * (A*B^T + B*A^T) + beta * C    (trans='N', A,B are N×K)
 *   C := alpha * (A^T*B + B^T*A) + beta * C    (trans='T', A,B are K×N)
 *
 * Only the UPLO triangle of C is read or written; the off-UPLO triangle
 * is left untouched (verified by the fuzz sentinel check).
 *
 * Port source: OpenBLAS.
 *   - interface/syr2k.c                (UPLO/TRANS dispatch; xerbla/info)
 *   - driver/level3/syr2k_k.c          (triangular-beta + level3_syr2k
 *                                       inclusion)
 *   - driver/level3/level3_syr2k.c     (blocking nest; two-pass diagonal
 *                                       walk per (is, js) tile)
 *   - driver/level3/syr2k_kernel.c     (mblas_esyr2k_kernel_{u,l} in
 *                                       common/mblas_l3_real.c)
 *
 * Reuses the shared microkernel + packers from common/. The per-tile
 * loop does TWO kernel calls: pass 1 with (Ap=A-pack, Bp=B-pack) and
 * flag=1 (diagonal subbuf merge enabled — covers both A*B^T and B*A^T
 * contributions on the NR×NR diagonal block via the symmetric mirror);
 * pass 2 with (Ap=B-pack, Bp=A-pack) and flag=0 (only the kept-triangle
 * strips, accumulating B*A^T into the same C locations).
 *
 * Triangular beta reuses the SYRK helpers (mblas_esyrk_beta_{u,l}) —
 * the off-UPLO triangle is left untouched.
 *
 * Fortran ABI:
 *   subroutine msyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
 *   - character args with trailing hidden size_t lengths (gfortran)
 *   - all scalars by pointer; REAL(KIND=10) ↔ multifloats::float64x2
 */

#include "mblas_l3_real.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#define MR MBLAS_EGEMM_MR
#define NR MBLAS_EGEMM_NR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


extern "C" void msyr2k_(
    const char *uplo_p, const char *trans_p,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const T alpha = *alpha_, beta = *beta_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const int uplo  = (char)toupper((unsigned char)*uplo_p);
    const int trans = (char)toupper((unsigned char)*trans_p);

    if (N <= 0) return;

    if (uplo == 'U') mblas_esyrk_beta_u((ptrdiff_t)N, beta, c, (ptrdiff_t)ldc);
    else             mblas_esyrk_beta_l((ptrdiff_t)N, beta, c, (ptrdiff_t)ldc);

    if (K == 0 || alpha == 0.0) return;

    int MC0, KC, NC;
    mblas_egemm_blocks(&MC0, &KC, &NC);

    int MC = MC0;
    if (K <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;
        long target_mc = L2_TARGET_BYTES / ((long)K * (long)sizeof(T));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((int)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * sizeof(T);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
#endif

    /* SYR2K does ~ N^2 * K flops; tiny-cutoff sized like egemm. */
    long nnk = (long)N * (long)N * (long)K;
    if (nnk < 64L * 64L * 64L) nthreads = 1;

    /* Two B-side packs (A in B-shape, B in B-shape) shared across all
     * M-strips of this (js, ls) tile. */
    T *Bp_A = (T *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    T *Bp_B = (T *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp_A || !Bp_B) { free(Bp_A); free(Bp_B); return; }

    /* Two A-side packs per thread (one for A, one for B). */
    T **Ap_A_arr = (T **)calloc((size_t)nthreads, sizeof(T *));
    T **Ap_B_arr = (T **)calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_A_arr || !Ap_B_arr) {
        free(Ap_A_arr); free(Ap_B_arr); free(Bp_A); free(Bp_B); return;
    }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_A_arr[t] = (T *)aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        Ap_B_arr[t] = (T *)aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (!Ap_A_arr[t] || !Ap_B_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) {
            free(Ap_A_arr[t]); free(Ap_B_arr[t]);
        }
        free(Ap_A_arr); free(Ap_B_arr); free(Bp_A); free(Bp_B); return;
    }

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
                    if (trans == 'N') {
                        mblas_egemm_tcopy(pb, jb,
                            &a[(size_t)ls * lda + js], lda, Bp_A);
                        mblas_egemm_tcopy(pb, jb,
                            &b[(size_t)ls * ldb + js], ldb, Bp_B);
                    } else {
                        mblas_egemm_ncopy(pb, jb,
                            &a[(size_t)js * lda + ls], lda, Bp_A);
                        mblas_egemm_ncopy(pb, jb,
                            &b[(size_t)js * ldb + ls], ldb, Bp_B);
                    }
                }

                for (int is = m_lo_eff; is < m_hi_eff; is += MC) {
                    int min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (trans == 'N') {
                        mblas_egemm_tcopy(pb, min_i,
                            &a[(size_t)ls * lda + is], lda, Ap_A);
                        mblas_egemm_tcopy(pb, min_i,
                            &b[(size_t)ls * ldb + is], ldb, Ap_B);
                    } else {
                        mblas_egemm_ncopy(pb, min_i,
                            &a[(size_t)is * lda + ls], lda, Ap_A);
                        mblas_egemm_ncopy(pb, min_i,
                            &b[(size_t)is * ldb + ls], ldb, Ap_B);
                    }

                    /* Pass 1: alpha * A * B^T into UPLO triangle (incl.
                     * symmetric diagonal merge). */
                    if (uplo == 'U') {
                        mblas_esyr2k_kernel_u(min_i, jb, pb, alpha,
                            Ap_A, Bp_B,
                            &c[(size_t)js * ldc + is], ldc,
                            (ptrdiff_t)(is - js), 1);
                    } else {
                        mblas_esyr2k_kernel_l(min_i, jb, pb, alpha,
                            Ap_A, Bp_B,
                            &c[(size_t)js * ldc + is], ldc,
                            (ptrdiff_t)(is - js), 1);
                    }

                    /* Pass 2: alpha * B * A^T into kept-triangle strips
                     * only — diagonal block already covered by pass 1's
                     * symmetric merge. */
                    if (uplo == 'U') {
                        mblas_esyr2k_kernel_u(min_i, jb, pb, alpha,
                            Ap_B, Bp_A,
                            &c[(size_t)js * ldc + is], ldc,
                            (ptrdiff_t)(is - js), 0);
                    } else {
                        mblas_esyr2k_kernel_l(min_i, jb, pb, alpha,
                            Ap_B, Bp_A,
                            &c[(size_t)js * ldc + is], ldc,
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
