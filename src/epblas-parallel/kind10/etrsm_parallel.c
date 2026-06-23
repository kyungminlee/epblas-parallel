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
#include <stdbool.h>
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "etrsm_kernel.h"
#include "egemm_kernel.h"   /* egemm_choose_blocks / egemm_beta_prepass / blas_round_up */
#include "../common/epblas_facade.h"

typedef etrsm_TR TR;

#define MR 2
#define NR 2

static void etrsm_core(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const TR *alpha_,
    const TR *a, ptrdiff_t lda,
    TR *b, ptrdiff_t ldb)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        etrsm_serial(side, uplo, transa, diag, m, n, alpha_, a, lda, b, ldb);
        return;
    }
#endif
    const TR alpha = *alpha_;

    const bool lside = (blas_up(side)   == 'L');
    const bool upper = (blas_up(uplo)   == 'U');
    const char TRANS  = blas_up(transa);
    const bool trans = (TRANS == 'T' || TRANS == 'C');   /* real: 'C' ≡ 'T' */
    const bool unit  = (blas_up(diag) == 'U');

    if (m == 0 || n == 0) return;

    /* α pre-scale of B (mirrors trsm_{L,R}.c GEMM_BETA pass). */
    if (alpha != 1.0L) egemm_beta_prepass(m, n, alpha, b, ldb);
    if (alpha == 0.0L) return;

    const ptrdiff_t K_eff = lside ? m : n;
    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(K_eff, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * (size_t)KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * (size_t)blas_round_up(NC, NR) * sizeof(TR);

#ifdef _OPENMP
    ptrdiff_t nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif

    /* Small problems: one thread (the parallel-region setup isn't worth
     * it). Mirrors OpenBLAS's GEMM_MULTITHREAD_THRESHOLD gating. */
    long mnk = (long)m * (long)n * (long)K_eff;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

    const ptrdiff_t partition_axis = lside ? n : m;
    if (nthreads > partition_axis) nthreads = partition_axis;
    if (nthreads < 1) nthreads = 1;

    /* Per-thread Ap/Bp scratch, allocated BEFORE the region: an in-region
     * alloc failure that skips a thread's loop body would deadlock the
     * others at no barrier here (there is none), but pre-allocating keeps
     * the failure path simple and race-free. */
    TR **Ap_arr = calloc((size_t)nthreads, sizeof(TR *));
    TR **Bp_arr = calloc((size_t)nthreads, sizeof(TR *));
    if (!Ap_arr || !Bp_arr) { free(Ap_arr); free(Bp_arr); return; }
    ptrdiff_t alloc_ok = 1;
    for (ptrdiff_t t = 0; t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        Bp_arr[t] = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t] || !Bp_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (ptrdiff_t t = 0; t < nthreads; ++t) {
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
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
#else
        ptrdiff_t tid = 0, nth = 1;
#endif
        TR *Ap = Ap_arr[tid];
        TR *Bp = Bp_arr[tid];

        if (lside) {
            ptrdiff_t chunk = blas_round_up((n + nth - 1) / nth, NR);
            ptrdiff_t js0 = tid * chunk;
            ptrdiff_t js1 = js0 + chunk;
            if (js0 > n) js0 = n;
            if (js1 > n) js1 = n;
            if (js0 < js1)
                etrsm_L_band(upper, trans, unit, m, js0, js1,
                             MC, KC, NC, a, lda, b, ldb, Ap, Bp);
        } else {
            ptrdiff_t chunk = blas_round_up((m + nth - 1) / nth, MR);
            ptrdiff_t m_lo = tid * chunk;
            ptrdiff_t m_hi = m_lo + chunk;
            if (m_lo > m) m_lo = m;
            if (m_hi > m) m_hi = m;
            if (m_lo < m_hi)
                etrsm_R_band(upper, trans, unit, n, m_lo, m_hi,
                             MC, KC, NC, a, lda, b, ldb, Ap, Bp);
        }
    }

    for (ptrdiff_t t = 0; t < nthreads; ++t) {
        free(Ap_arr[t]);
        free(Bp_arr[t]);
    }
    free(Ap_arr);
    free(Bp_arr);
}

EPBLAS_FACADE_TRMM(etrsm, TR)
