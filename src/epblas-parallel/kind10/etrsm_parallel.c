/*
 * etrsm_ — kind10 (REAL(KIND=10) / long double) triangular solve: the
 * public Fortran entry and threading-orchestration half of the etrsm
 * overlay (all the math lives in etrsm_serial.c / etrsm_kernel.c /
 * etrsm_pack.c). Faithful port of OpenBLAS interface/trsm.c dispatch
 * (gemm_thread_n for SIDE='L', gemm_thread_m for SIDE='R').
 *
 * One `omp parallel` per solve. Each thread takes a contiguous slice of
 * the free axis — B's columns (SIDE='L') or rows (SIDE='R') — and runs
 * the full L3 band driver on its slice with private Ap/Bp scratch, so
 * there is no cross-thread synchronization inside the solve.
 *
 * Nesting guard: when etrsm_ is called from inside another routine's
 * parallel region, it delegates to etrsm_serial and opens no team of its
 * own — a nested team trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "etrsm_kernel.h"
#include "egemm_kernel.h"   /* egemm_choose_blocks / egemm_beta_prepass / egemm_round_up */

typedef etrsm_T T;

#define MR 2
#define NR 2

void etrsm_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        etrsm_serial(side, uplo, transa, diag, m_, n_, alpha_, a, lda_,
                     b, ldb_, side_len, uplo_len, transa_len, diag_len);
        return;
    }
#endif
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;

    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;

    const int lside = (toupper((unsigned char)*side)   == 'L');
    const int upper = (toupper((unsigned char)*uplo)   == 'U');
    const char trc  = (char)toupper((unsigned char)*transa);
    const int trans = (trc == 'T' || trc == 'C');   /* real: 'C' ≡ 'T' */
    const int unit  = (toupper((unsigned char)*diag) == 'U');

    if (M == 0 || N == 0) return;

    /* α pre-scale of B (mirrors trsm_{L,R}.c GEMM_BETA pass). */
    if (alpha != 1.0L) egemm_beta_prepass(M, N, alpha, b, ldb);
    if (alpha == 0.0L) return;

    const int K_eff = lside ? M : N;
    int MC, KC, NC;
    egemm_choose_blocks(K_eff, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)egemm_round_up(NC, NR) * sizeof(T);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
#endif

    /* Small problems: one thread (the parallel-region setup isn't worth
     * it). Mirrors OpenBLAS's GEMM_MULTITHREAD_THRESHOLD gating. */
    long mnk = (long)M * (long)N * (long)K_eff;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

    const int partition_axis = lside ? N : M;
    if (nthreads > partition_axis) nthreads = partition_axis;
    if (nthreads < 1) nthreads = 1;

    /* Per-thread Ap/Bp scratch, allocated BEFORE the region: an in-region
     * alloc failure that skips a thread's loop body would deadlock the
     * others at no barrier here (there is none), but pre-allocating keeps
     * the failure path simple and race-free. */
    T **Ap_arr = calloc((size_t)nthreads, sizeof(T *));
    T **Bp_arr = calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_arr || !Bp_arr) { free(Ap_arr); free(Bp_arr); return; }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        Bp_arr[t] = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t] || !Bp_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) {
            if (Ap_arr) free(Ap_arr[t]);
            if (Bp_arr) free(Bp_arr[t]);
        }
        free(Ap_arr); free(Bp_arr);
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
        T *Bp = Bp_arr[tid];

        if (lside) {
            int chunk = egemm_round_up((N + nth - 1) / nth, NR);
            int js0 = tid * chunk;
            int js1 = js0 + chunk;
            if (js0 > N) js0 = N;
            if (js1 > N) js1 = N;
            if (js0 < js1)
                etrsm_L_band(upper, trans, unit, M, js0, js1,
                             MC, KC, NC, a, lda, b, ldb, Ap, Bp);
        } else {
            int chunk = egemm_round_up((M + nth - 1) / nth, MR);
            int m_lo = tid * chunk;
            int m_hi = m_lo + chunk;
            if (m_lo > M) m_lo = M;
            if (m_hi > M) m_hi = M;
            if (m_lo < m_hi)
                etrsm_R_band(upper, trans, unit, N, m_lo, m_hi,
                             MC, KC, NC, a, lda, b, ldb, Ap, Bp);
        }
    }

    for (int t = 0; t < nthreads; ++t) {
        free(Ap_arr[t]);
        free(Bp_arr[t]);
    }
    free(Ap_arr);
    free(Bp_arr);
}
