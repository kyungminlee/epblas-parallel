/*
 * xsyr2k — kind16 (COMPLEX(KIND=16)) port of OpenBLAS ZSYR2K.
 *
 *   C := alpha * (A*B^T + B*A^T) + beta * C    (trans='N', A,B are N×K)
 *   C := alpha * (A^T*B + B^T*A) + beta * C    (trans='T', A,B are K×N)
 *
 * Complex SYMMETRIC variant — no conjugation, A == A^T (NOT Hermitian).
 * HER2K is a separate routine (xher2k). Only the UPLO
 * triangle of C is read or written.
 *
 * Port source: OpenBLAS.
 *   - interface/syr2k.c              (Z-variant: UPLO/TRANS dispatch)
 *   - driver/level3/syr2k_k.c        (triangular-beta + level3_syr2k hook)
 *   - driver/level3/level3_syr2k.c   (blocking nest; two-pass diagonal walk)
 *   - driver/level3/syr2k_kernel.c   (qblas_ysyr2k_kernel_{u,l} in
 *                                     common/qblas_l3_complex.c)
 *
 * Reuses qblas_ygemm_kernel + qblas_ygemm_ncopy / qblas_ygemm_tcopy from common/.
 * No conjugation, so packers are called with `conj = 0`.
 *
 * Fortran ABI:
 *   subroutine xsyr2k(uplo, trans, n, k, alpha, a, lda, b, ldb, beta, c, ldc)
 *   - alpha, beta are COMPLEX(KIND=16): 2 __float128s each (re, im)
 *   - a, b, c are COMPLEX(KIND=16) arrays (interleaved re,im)
 *   - lda, ldb, ldc are in COMPLEX(KIND=16) elements
 */

#include "qblas_l3_complex.h"
#include "qblas_tuning.h"
#include <quadmath.h>
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __float128 T;

#define MR QBLAS_YGEMM_MR
#define NR QBLAS_YGEMM_NR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


void xsyr2k_(
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
    const T beta_r = beta_[0],  beta_i = beta_[1];
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const int uplo  = (char)toupper((unsigned char)*uplo_p);
    const int trans = (char)toupper((unsigned char)*trans_p);

    if (N <= 0) return;

    if (uplo == 'U') qblas_ysyrk_beta_u((ptrdiff_t)N, beta_r, beta_i,
                                        c, (ptrdiff_t)ldc);
    else             qblas_ysyrk_beta_l((ptrdiff_t)N, beta_r, beta_i,
                                        c, (ptrdiff_t)ldc);

    if (K == 0 || (alphar == 0.0Q && alphai == 0.0Q)) return;

    int MC0, KC, NC;
    qblas_ygemm_blocks(&MC0, &KC, &NC);

    int MC = MC0;
    if (K <= KC) {
        const long L2_TARGET_BYTES = QBLAS_L2_TARGET_BYTES;
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

    T *Bp_A = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    T *Bp_B = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp_A || !Bp_B) { free(Bp_A); free(Bp_B); return; }

    T **Ap_A_arr = calloc((size_t)nthreads, sizeof(T *));
    T **Ap_B_arr = calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_A_arr || !Ap_B_arr) {
        free(Ap_A_arr); free(Ap_B_arr); free(Bp_A); free(Bp_B); return;
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
                        qblas_ygemm_tcopy(pb, jb, 0,
                            &a[((size_t)ls * lda + js) * 2], lda, Bp_A);
                        qblas_ygemm_tcopy(pb, jb, 0,
                            &b[((size_t)ls * ldb + js) * 2], ldb, Bp_B);
                    } else {
                        qblas_ygemm_ncopy(pb, jb, 0,
                            &a[((size_t)js * lda + ls) * 2], lda, Bp_A);
                        qblas_ygemm_ncopy(pb, jb, 0,
                            &b[((size_t)js * ldb + ls) * 2], ldb, Bp_B);
                    }
                }

                for (int is = m_lo_eff; is < m_hi_eff; is += MC) {
                    int min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (trans == 'N') {
                        qblas_ygemm_tcopy(pb, min_i, 0,
                            &a[((size_t)ls * lda + is) * 2], lda, Ap_A);
                        qblas_ygemm_tcopy(pb, min_i, 0,
                            &b[((size_t)ls * ldb + is) * 2], ldb, Ap_B);
                    } else {
                        qblas_ygemm_ncopy(pb, min_i, 0,
                            &a[((size_t)is * lda + ls) * 2], lda, Ap_A);
                        qblas_ygemm_ncopy(pb, min_i, 0,
                            &b[((size_t)is * ldb + ls) * 2], ldb, Ap_B);
                    }

                    if (uplo == 'U') {
                        qblas_ysyr2k_kernel_u(min_i, jb, pb, alphar, alphai,
                            Ap_A, Bp_B,
                            &c[((size_t)js * ldc + is) * 2], ldc,
                            (ptrdiff_t)(is - js), 1);
                        qblas_ysyr2k_kernel_u(min_i, jb, pb, alphar, alphai,
                            Ap_B, Bp_A,
                            &c[((size_t)js * ldc + is) * 2], ldc,
                            (ptrdiff_t)(is - js), 0);
                    } else {
                        qblas_ysyr2k_kernel_l(min_i, jb, pb, alphar, alphai,
                            Ap_A, Bp_B,
                            &c[((size_t)js * ldc + is) * 2], ldc,
                            (ptrdiff_t)(is - js), 1);
                        qblas_ysyr2k_kernel_l(min_i, jb, pb, alphar, alphai,
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
