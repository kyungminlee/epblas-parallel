/*
 * esymm — kind10 (REAL(KIND=10) / 80-bit long double) port of OpenBLAS DSYMM.
 *
 *   C := alpha * A * B + beta * C    (SIDE='L', A symmetric M×M)
 *   C := alpha * B * A + beta * C    (SIDE='R', A symmetric N×N)
 *
 * Port source: OpenBLAS.
 *   - interface/symm.c            (SIDE/UPLO dispatch + the SIDE=R side-
 *                                  swap that moves the symmetric matrix
 *                                  into B's slot internally; xerbla / info)
 *   - driver/level3/symm_k.c      (selects SYMM_I{U,L}TCOPY for the A
 *                                  packer when SIDE=L, or SYMM_O{U,L}TCOPY
 *                                  for the B packer when SIDE=R; otherwise
 *                                  delegates to driver/level3/level3.c)
 *   - kernel/generic/
 *       symm_ucopy_2.c           → eblas_esymm_ucopy  (shared via common/)
 *       symm_lcopy_2.c           → eblas_esymm_lcopy  (shared via common/)
 *
 * The microkernel + standard GEMM packers (eblas_egemm_kernel /
 * eblas_egemm_ncopy / eblas_egemm_tcopy / eblas_egemm_beta) are reused
 * from the shared common/eblas_l3_real.c — SYMM only changes WHICH
 * pack function fills Ap (SIDE=L) or Bp (SIDE=R).
 *
 * Differences from upstream DSYMM:
 *   - No SIMD (same reason as egemm).
 *   - Single-level OpenMP over the M axis with per-thread Ap, shared Bp
 *     packed once per (jc, pc) under `omp single` (mirrors egemm.c
 *     exactly — see that file for the barrier ordering).
 *
 * Fortran ABI:
 *   subroutine esymm(side, uplo, m, n, alpha, a, lda, b, ldb,
 *                    beta, c, ldc)
 *   - character args with trailing hidden size_t lengths (gfortran)
 *   - all scalars by pointer; REAL(KIND=10) ↔ long double
 *   - A is M×M when side='L', N×N when side='R'; only the UPLO-indicated
 *     triangle is read.
 */

#include "eblas_l3_real.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#define MR EBLAS_EGEMM_MR
#define NR EBLAS_EGEMM_NR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


void esymm_(
    const char *side_p, const char *uplo_p,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_)
{
    const int M = *m_, N = *n_;
    const T alpha = *alpha_, beta = *beta_;
    const int side = (char)toupper((unsigned char)*side_p);
    const int uplo = (char)toupper((unsigned char)*uplo_p);

    if (M <= 0 || N <= 0) return;

    /* Beta pre-pass. */
    eblas_egemm_beta((ptrdiff_t)M, (ptrdiff_t)N, beta, c, (ptrdiff_t)(*ldc_));

    if (alpha == 0.0L) return;

    /* OpenBLAS's interface/symm.c side-swap: when SIDE=R, the symmetric
     * matrix logically lives in the B slot of the internal GEMM. This
     * lets the level3 driver use OCOPY (= SYMM packer for B) instead of
     * ICOPY, while still iterating M / N / K identically. The "K" of
     * the underlying GEMM is the side-dim of the symmetric matrix. */
    const T *A_eff = (side == 'L') ? a : b;
    const T *B_eff = (side == 'L') ? b : a;
    const int lda_eff = (side == 'L') ? *lda_ : *ldb_;
    const int ldb_eff = (side == 'L') ? *ldb_ : *lda_;
    const int ldc    = *ldc_;
    const int K = (side == 'L') ? M : N;

    if (K == 0) return;

    /* Block sizes (env-overridable). */
    int MC0, KC, NC;
    eblas_egemm_blocks(&MC0, &KC, &NC);

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

    long mnk = (long)M * (long)N * (long)K;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

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

        int m_chunk = round_up((M + nth - 1) / nth, MR);
        int m_lo = tid * m_chunk;
        int m_hi = m_lo + m_chunk;
        if (m_hi > M) m_hi = M;

        for (int js = 0; js < N; js += NC) {
            int jb = (N - js < NC) ? (N - js) : NC;
            for (int ls = 0; ls < K; ls += KC) {
                int pb = (K - ls < KC) ? (K - ls) : KC;

                /* OCOPY(B). When SIDE=L, B is the regular matrix → standard
                 * NCOPY. When SIDE=R, the symmetric matrix is now in the
                 * B slot → use the SYMM packer. The (posX, posY) mapping
                 * follows OpenBLAS's level3.c + symm_k.c macro chain:
                 *
                 *   OCOPY_OPERATION(M=pb, N=jb, B, ldb, X=ls, Y=js, buf)
                 *     -> SYMM_O*TCOPY(M=pb, N=jb, B, ldb,
                 *                     posX=js, posY=ls, buf)
                 */
#ifdef _OPENMP
                #pragma omp barrier
                #pragma omp single
#endif
                {
                    if (side == 'L') {
                        eblas_egemm_ncopy(pb, jb,
                            &B_eff[(size_t)js * ldb_eff + ls], ldb_eff, Bp);
                    } else {
                        if (uplo == 'U') {
                            eblas_esymm_ucopy(pb, jb,
                                B_eff, ldb_eff, js, ls, Bp);
                        } else {
                            eblas_esymm_lcopy(pb, jb,
                                B_eff, ldb_eff, js, ls, Bp);
                        }
                    }
                }

                for (int is = m_lo; is < m_hi; is += MC) {
                    int min_i = (m_hi - is < MC) ? (m_hi - is) : MC;

                    /* ICOPY(A). When SIDE=L, A is the symmetric matrix
                     * → SYMM packer; when SIDE=R, A is the regular matrix
                     * → standard TCOPY (normal-A path of GEMM).
                     *
                     *   ICOPY_OPERATION(M=pb, N=min_i, A, lda, X=ls, Y=is, buf)
                     *     -> SYMM_I*TCOPY(M=pb, N=min_i, A, lda,
                     *                     posX=is, posY=ls, buf)
                     */
                    if (side == 'L') {
                        if (uplo == 'U') {
                            eblas_esymm_ucopy(pb, min_i,
                                A_eff, lda_eff, is, ls, Ap);
                        } else {
                            eblas_esymm_lcopy(pb, min_i,
                                A_eff, lda_eff, is, ls, Ap);
                        }
                    } else {
                        eblas_egemm_tcopy(pb, min_i,
                            &A_eff[(size_t)ls * lda_eff + is], lda_eff, Ap);
                    }

                    eblas_egemm_kernel(min_i, jb, pb, alpha,
                                       Ap, Bp,
                                       &c[(size_t)js * ldc + is], ldc);
                }
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}
