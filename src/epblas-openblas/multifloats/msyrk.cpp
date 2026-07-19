/*
 * msyrk — multifloats DD (float64x2, 128-bit double-double) port of OpenBLAS DSYRK.
 *
 * ADAPTATION vs the verbatim kind10 source: the element type here is
 * multifloats::float64x2 — a double-double of two binary64 limbs
 * (128 bits), not the 80-bit x87 REAL(KIND=10) of the kind10 leg.
 * Structure, loop order, blocking and thresholds are the kind10
 * port's; only the element type and its arithmetic differ.
 *
 *   C := alpha * A * A^T + beta * C    (trans='N', A is N×K)
 *   C := alpha * A^T * A + beta * C    (trans='T', A is K×N)
 *
 * Only the UPLO triangle of C is read or written; the off-UPLO triangle
 * is left untouched (verified by the fuzz sentinel check).
 *
 * Port source: OpenBLAS.
 *   - interface/syrk.c                 (UPLO/TRANS dispatch; xerbla/info)
 *   - driver/level3/syrk_k.c           (triangular-beta + level3_syrk
 *                                       inclusion)
 *   - driver/level3/level3_syrk.c      (blocking nest with triangle-aware
 *                                       m_start/m_end clipping per N-band)
 *   - driver/level3/syrk_kernel.c      (mblas_esyrk_kernel_{u,l} in
 *                                       common/mblas_l3_real.c)
 *
 * Reuses the shared microkernel, packers, and beta from common/. SYRK
 * adds: triangular β pre-pass + diagonal-aware kernel + UPLO-clipped
 * m-loop bounds. Both Ap and Bp source from the same input A — for
 * trans='N' Bp holds A again in trans pack-shape, for trans='T' Ap
 * uses the trans pack-shape and Bp holds A in normal pack-shape.
 *
 * Fortran ABI:
 *   subroutine msyrk(uplo, trans, n, k, alpha, a, lda, beta, c, ldc)
 *   - character args are plain char* — NO trailing hidden length args
 *     (declaring them caused the v0.9.1 frame corruption; never re-add)
 *   - all scalars by pointer; REAL(KIND=10) ↔ multifloats::float64x2
 */

#include "mblas_l3_real.h"
#include "mblas_tuning.h"
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


extern "C" void msyrk_(
    const char *uplo_p, const char *trans_p,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_)
{
    const int N = *n_, K = *k_;
    const T alpha = *alpha_, beta = *beta_;
    const int lda = *lda_, ldc = *ldc_;
    const int uplo  = (char)toupper((unsigned char)*uplo_p);
    const int trans = (char)toupper((unsigned char)*trans_p);

    if (N <= 0) return;

    /* Triangular beta pre-pass on the UPLO triangle of C only. */
    if (uplo == 'U') mblas_esyrk_beta_u((ptrdiff_t)N, beta, c, (ptrdiff_t)ldc);
    else             mblas_esyrk_beta_l((ptrdiff_t)N, beta, c, (ptrdiff_t)ldc);

    if (K == 0 || alpha == 0.0) return;

    int MC0, KC, NC;
    mblas_egemm_blocks(&MC0, &KC, &NC);

    int MC = MC0;
    if (K <= KC) {
        const long L2_TARGET_BYTES = MBLAS_L2_TARGET_BYTES;
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

    /* SYRK does ~ N^2 * K / 2 flops; tiny-cutoff sized to match egemm. */
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

        /* M-axis partition (= N-axis of SYRK output). The per-(js)
         * UPLO clip will further reduce each thread's actual work,
         * but every thread still encounters every `omp single` /
         * `omp barrier`. */
        int m_chunk = round_up((N + nth - 1) / nth, MR);
        int m_lo = tid * m_chunk;
        int m_hi = m_lo + m_chunk;
        if (m_hi > N) m_hi = N;

        for (int js = 0; js < N; js += NC) {
            int jb = (N - js < NC) ? (N - js) : NC;

            /* UPLO clip for this js-band:
             *   UPPER: only rows up to js+jb contribute (rows below
             *          would write strictly into the lower triangle,
             *          which the kernel skips anyway).
             *   LOWER: only rows from js onwards. */
            int m_lo_eff = (uplo == 'L' && m_lo < js) ? js : m_lo;
            int m_hi_eff = (uplo == 'U' && m_hi > js + jb) ? (js + jb) : m_hi;
            /* Realign to MR so packer slabs stay aligned. */
            if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);
            if (m_lo_eff < m_lo) m_lo_eff = m_lo;

            for (int ls = 0; ls < K; ls += KC) {
                int pb = (K - ls < KC) ? (K - ls) : KC;

                /* OCOPY(B) — pack source A again, but in the OCOPY
                 * pack shape. For trans='N' we want the same A read
                 * at (row=js..js+jb, col=ls..ls+pb); for trans='T'
                 * the read is at (row=ls..ls+pb, col=js..js+jb). */
#ifdef _OPENMP
                #pragma omp barrier
                #pragma omp single
#endif
                {
                    if (trans == 'N') {
                        mblas_egemm_tcopy(pb, jb,
                            &a[(size_t)ls * lda + js], lda, Bp);
                    } else {
                        mblas_egemm_ncopy(pb, jb,
                            &a[(size_t)js * lda + ls], lda, Bp);
                    }
                }

                for (int is = m_lo_eff; is < m_hi_eff; is += MC) {
                    int min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                    /* ICOPY(A) — into Ap. Mirrors egemm's ICOPY
                     * pattern but reads the same source matrix as
                     * OCOPY (A doubles as B in SYRK). */
                    if (trans == 'N') {
                        mblas_egemm_tcopy(pb, min_i,
                            &a[(size_t)ls * lda + is], lda, Ap);
                    } else {
                        mblas_egemm_ncopy(pb, min_i,
                            &a[(size_t)is * lda + ls], lda, Ap);
                    }

                    /* Diagonal-aware kernel. offset = X - Y = is - js. */
                    if (uplo == 'U') {
                        mblas_esyrk_kernel_u(min_i, jb, pb, alpha,
                            Ap, Bp,
                            &c[(size_t)js * ldc + is], ldc,
                            (ptrdiff_t)(is - js));
                    } else {
                        mblas_esyrk_kernel_l(min_i, jb, pb, alpha,
                            Ap, Bp,
                            &c[(size_t)js * ldc + is], ldc,
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
