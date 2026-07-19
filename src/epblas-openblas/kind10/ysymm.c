/*
 * ysymm — kind10 (COMPLEX(KIND=10)) port of OpenBLAS ZSYMM.
 *
 *   C := alpha * A * B + beta * C    (SIDE='L', A complex-symmetric M×M)
 *   C := alpha * B * A + beta * C    (SIDE='R', A complex-symmetric N×N)
 *
 * NOTE: ZSYMM operates on a complex SYMMETRIC matrix (A == A^T), NOT a
 * Hermitian matrix. There is no conjugation involved — the symmetric
 * reflection just copies (re, im) across the diagonal as-is. Hermitian
 * HEMM is a separate routine (yhemm — task #64).
 *
 * Port source: OpenBLAS.
 *   - interface/symm.c           (Z-variant: side-swap, xerbla, etc.)
 *   - driver/level3/symm_k.c     (same level3 wrapper as DSYMM, just
 *                                 with COMPSIZE=2)
 *   - kernel/generic/
 *       zsymm_ucopy_2.c          → eblas_ysymm_ucopy  (shared via common/)
 *       zsymm_lcopy_2.c          → eblas_ysymm_lcopy  (shared via common/)
 *
 * The microkernel + standard GEMM packers (eblas_ygemm_kernel /
 * eblas_ygemm_ncopy / eblas_ygemm_tcopy / eblas_ygemm_beta) are reused
 * from the shared common/eblas_l3_complex.c. SYMM packers do NOT
 * negate the imag float (no conjugation).
 *
 * Fortran ABI:
 *   subroutine ysymm(side, uplo, m, n, alpha, a, lda, b, ldb,
 *                    beta, c, ldc)
 *   - character args are passed as bare char* (hidden gfortran length
 *     args deliberately omitted — never re-add them)
 *   - alpha, beta are COMPLEX(KIND=10): 2 long doubles each (re, im)
 *   - a, b, c are COMPLEX(KIND=10) arrays (interleaved re,im)
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


void ysymm_(
    const char *side_p, const char *uplo_p,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_)
{
    const int M = *m_, N = *n_;
    const T alphar = alpha_[0], alphai = alpha_[1];
    const T beta_r = beta_[0],  beta_i = beta_[1];
    const int side = (char)toupper((unsigned char)*side_p);
    const int uplo = (char)toupper((unsigned char)*uplo_p);

    if (M <= 0 || N <= 0) return;

    /* alpha==0: the beta pass on C is the entire operation. */
    if (alphar == 0.0L && alphai == 0.0L) {
        eblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, beta_r, beta_i,
                         c, (ptrdiff_t)(*ldc_));
        return;
    }

    const T *A_eff = (side == 'L') ? a : b;
    const T *B_eff = (side == 'L') ? b : a;
    const int lda_eff = (side == 'L') ? *lda_ : *ldb_;
    const int ldb_eff = (side == 'L') ? *ldb_ : *lda_;
    const int ldc    = *ldc_;
    const int K = (side == 'L') ? M : N;

    if (K == 0) {
        eblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, beta_r, beta_i,
                         c, (ptrdiff_t)ldc);
        return;
    }

    int MC0, KC, NC;
    eblas_ygemm_blocks(&MC0, &KC, &NC);

    int MC = MC0;
    if (K <= KC) {
        long target_mc = L2_TARGET_BYTES / ((long)K * (long)(2 * sizeof(T)));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((int)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    /* Complex elements → 2 long doubles per element. */
    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * 2 * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * 2 * sizeof(T);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
#endif

    long mnk = (long)M * (long)N * (long)K;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

    /* All pack allocations happen BEFORE the in-place beta pass on C so
     * a failure can never leave C scaled but not updated; benchmark-
     * reference policy is to abort() loudly on failure. */
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) {
        fprintf(stderr, "ysymm: pack buffer allocation failed\n");
        abort();
    }

    T **Ap_arr = calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_arr) {
        free(Bp);
        fprintf(stderr, "ysymm: pack buffer allocation failed\n");
        abort();
    }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
        free(Ap_arr); free(Bp);
        fprintf(stderr, "ysymm: pack buffer allocation failed\n");
        abort();
    }

    /* Beta pre-pass — after every allocation has succeeded. */
    eblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, beta_r, beta_i,
                     c, (ptrdiff_t)ldc);

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

#ifdef _OPENMP
                #pragma omp barrier
                #pragma omp single
#endif
                {
                    if (side == 'L') {
                        /* SIDE=L: B is regular; standard normal-B NCOPY,
                         * conj=0 (no conjugation in SYMM). */
                        eblas_ygemm_ncopy(pb, jb, 0,
                            &B_eff[((size_t)js * ldb_eff + ls) * 2],
                            ldb_eff, Bp);
                    } else {
                        if (uplo == 'U') {
                            eblas_ysymm_ucopy(pb, jb,
                                B_eff, ldb_eff, js, ls, Bp);
                        } else {
                            eblas_ysymm_lcopy(pb, jb,
                                B_eff, ldb_eff, js, ls, Bp);
                        }
                    }
                }

                for (int is = m_lo; is < m_hi; is += MC) {
                    int min_i = (m_hi - is < MC) ? (m_hi - is) : MC;

                    if (side == 'L') {
                        if (uplo == 'U') {
                            eblas_ysymm_ucopy(pb, min_i,
                                A_eff, lda_eff, is, ls, Ap);
                        } else {
                            eblas_ysymm_lcopy(pb, min_i,
                                A_eff, lda_eff, is, ls, Ap);
                        }
                    } else {
                        /* SIDE=R: A is regular; standard normal-A TCOPY,
                         * conj=0. */
                        eblas_ygemm_tcopy(pb, min_i, 0,
                            &A_eff[((size_t)ls * lda_eff + is) * 2],
                            lda_eff, Ap);
                    }

                    eblas_ygemm_kernel(min_i, jb, pb, alphar, alphai,
                                       Ap, Bp,
                                       &c[((size_t)js * ldc + is) * 2], ldc);
                }
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}
