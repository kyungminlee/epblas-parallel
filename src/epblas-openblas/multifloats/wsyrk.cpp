/*
 * wsyrk — kind10 (COMPLEX(KIND=10)) port of OpenBLAS ZSYRK.
 *
 *   C := alpha * A * A^T + beta * C    (trans='N', A is N×K)
 *   C := alpha * A^T * A + beta * C    (trans='T', A is K×N)
 *
 * Complex SYMMETRIC variant — no conjugation, A == A^T (NOT Hermitian).
 * HERK is a separate routine (yherk — task #65). Only the UPLO triangle
 * of C is read or written.
 *
 * Port source: OpenBLAS.
 *   - interface/syrk.c              (Z-variant: UPLO/TRANS dispatch)
 *   - driver/level3/syrk_k.c        (triangular-beta + level3_syrk hook)
 *   - driver/level3/level3_syrk.c   (blocking nest with UPLO clip)
 *   - driver/level3/syrk_kernel.c   (mblas_ysyrk_kernel_{u,l} in
 *                                    common/mblas_l3_complex.c)
 *
 * Reuses mblas_ygemm_kernel + ygemm_ncopy / ygemm_tcopy from common/.
 * No conjugation, so packers are called with `conj = 0`.
 *
 * Fortran ABI:
 *   subroutine wsyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
 *   - alpha, beta are COMPLEX(KIND=10): 2 multifloats::float64x2s each (re, im)
 *   - a, c are COMPLEX(KIND=10) arrays (interleaved re,im)
 *   - lda, ldc are in COMPLEX(KIND=10) elements
 */

#include "mblas_l3_complex.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#define MR MBLAS_YGEMM_MR
#define NR MBLAS_YGEMM_NR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


extern "C" void wsyrk_(
    const char *uplo_p, const char *trans_p,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const T alphar = alpha_[0], alphai = alpha_[1];
    const T beta_r = beta_[0],  beta_i = beta_[1];
    const int lda = *lda_, ldc = *ldc_;
    const int uplo  = (char)toupper((unsigned char)*uplo_p);
    const int trans = (char)toupper((unsigned char)*trans_p);

    if (N <= 0) return;

    if (uplo == 'U') mblas_ysyrk_beta_u((ptrdiff_t)N, beta_r, beta_i,
                                        c, (ptrdiff_t)ldc);
    else             mblas_ysyrk_beta_l((ptrdiff_t)N, beta_r, beta_i,
                                        c, (ptrdiff_t)ldc);

    if (K == 0 || (alphar == 0.0 && alphai == 0.0)) return;

    int MC0, KC, NC;
    mblas_ygemm_blocks(&MC0, &KC, &NC);

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

    /* Complex packing: 2 multifloats::float64x2s per element. */
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

    T *Bp = (T *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) return;

    T **Ap_arr = (T **)calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_arr) { free(Bp); return; }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = (T *)aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
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
                        /* B-pack source is A read at (row=js, col=ls);
                         * trans-pack shape (same as egemm tb='T'). */
                        mblas_ygemm_tcopy(pb, jb, 0,
                            &a[((size_t)ls * lda + js) * 2], lda, Bp);
                    } else {
                        mblas_ygemm_ncopy(pb, jb, 0,
                            &a[((size_t)js * lda + ls) * 2], lda, Bp);
                    }
                }

                for (int is = m_lo_eff; is < m_hi_eff; is += MC) {
                    int min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    if (trans == 'N') {
                        mblas_ygemm_tcopy(pb, min_i, 0,
                            &a[((size_t)ls * lda + is) * 2], lda, Ap);
                    } else {
                        mblas_ygemm_ncopy(pb, min_i, 0,
                            &a[((size_t)is * lda + ls) * 2], lda, Ap);
                    }

                    if (uplo == 'U') {
                        mblas_ysyrk_kernel_u(min_i, jb, pb, alphar, alphai,
                            Ap, Bp,
                            &c[((size_t)js * ldc + is) * 2], ldc,
                            (ptrdiff_t)(is - js));
                    } else {
                        mblas_ysyrk_kernel_l(min_i, jb, pb, alphar, alphai,
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
