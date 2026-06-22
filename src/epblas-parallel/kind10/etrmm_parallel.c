/*
 * etrmm_ — kind10 (REAL(KIND=10) / long double) triangular matrix-multiply:
 * the public Fortran entry and threading-orchestration half of the etrmm
 * overlay (all the math lives in etrmm_serial.c / etrmm_kernel.c /
 * etrmm_pack.c). Faithful port of OpenBLAS interface/trsm.c dispatch
 * (gemm_thread_n for SIDE='L', gemm_thread_m for SIDE='R').
 *
 * One `omp parallel` per multiply. Each thread takes a contiguous slice of
 * the free axis — B's columns (SIDE='L') or rows (SIDE='R') — and runs the
 * full L3 band driver on its slice with private Ap/Bp scratch, so there is
 * no cross-thread synchronization inside the multiply (A is read-only; the
 * slices of B written are disjoint).
 *
 * Nesting guard: when etrmm_ is called from inside another routine's
 * parallel region, it delegates to etrmm_serial and opens no team of its
 * own — a nested team trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "etrmm_kernel.h"
#include "egemm_kernel.h"   /* egemm_choose_blocks / egemm_beta_prepass / egemm_round_up */
#include "../common/epblas_facade.h"

typedef etrmm_T T;

#define MR 2
#define NR 2

static void etrmm_core(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        etrmm_serial(side, uplo, transa, diag, m, n, alpha_, a, lda, b, ldb);
        return;
    }
#endif
    const T alpha = *alpha_;

    const bool lside = (blas_up(side)   == 'L');
    const bool upper = (blas_up(uplo)   == 'U');
    const char TRANS  = blas_up(transa);
    const bool trans = (TRANS == 'T' || TRANS == 'C');   /* real: 'C' ≡ 'T' */
    const bool unit  = (blas_up(diag) == 'U');

    if (m == 0 || n == 0) return;

    /* α pre-scale of B in place, then the nest runs kernel-alpha = 1
     * (mirrors trmm_{L,R}.c GEMM_BETA pass; alpha == 0 → B := 0). */
    if (alpha != 1.0L) egemm_beta_prepass(m, n, alpha, b, ldb);
    if (alpha == 0.0L) return;

    const ptrdiff_t K_eff = lside ? m : n;
    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(K_eff, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)egemm_round_up(NC, NR) * sizeof(T);

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

    /* Per-thread Ap/Bp scratch, allocated BEFORE the region (no barrier in
     * the region, so a per-thread alloc failure that skips a slice cannot
     * deadlock the others; pre-allocating keeps the failure path simple). */
    T **Ap_arr = calloc((size_t)nthreads, sizeof(T *));
    T **Bp_arr = calloc((size_t)nthreads, sizeof(T *));
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
        T *Ap = Ap_arr[tid];
        T *Bp = Bp_arr[tid];

        if (lside) {
            ptrdiff_t chunk = egemm_round_up((n + nth - 1) / nth, NR);
            ptrdiff_t js0 = tid * chunk;
            ptrdiff_t js1 = js0 + chunk;
            if (js0 > n) js0 = n;
            if (js1 > n) js1 = n;
            if (js0 < js1)
                etrmm_L_band(upper, trans, unit, m, js0, js1,
                             MC, KC, NC, a, lda, b, ldb, Ap, Bp);
        } else {
            ptrdiff_t chunk = egemm_round_up((m + nth - 1) / nth, MR);
            ptrdiff_t m_lo = tid * chunk;
            ptrdiff_t m_hi = m_lo + chunk;
            if (m_lo > m) m_lo = m;
            if (m_hi > m) m_hi = m;
            if (m_lo < m_hi)
                etrmm_R_band(upper, trans, unit, n, m_lo, m_hi,
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

EPBLAS_FACADE_TRMM(etrmm, T)
