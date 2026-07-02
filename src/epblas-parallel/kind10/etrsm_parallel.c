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
#include "etri_kernel.h"    /* etri_pack_guard_{poison,check} */
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
    const bool lside = (blas_up(side)   == 'L');
    const bool upper = (blas_up(uplo)   == 'U');
    const char TRANS  = blas_up(transa);
    const bool trans = (TRANS == 'T' || TRANS == 'C');   /* real: 'C' ≡ 'T' */
    const bool unit  = (blas_up(diag) == 'U');

    if (m == 0 || n == 0) return;

    const ptrdiff_t K_eff = lside ? m : n;

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

    /* Serial: run the single-band worker directly, opening no parallel
     * region. The num_threads(1) team fork/join is pure per-call overhead
     * here — uniform ~5% at N=64, shrinking with N — and ob (no OpenMP) pays
     * none of it. etrsm_serial does its own α pre-scale, so delegate the
     * whole call (α included) before we touch B, else B would be scaled
     * twice. */
    if (nthreads == 1) {
        etrsm_serial(side, uplo, transa, diag, m, n, alpha_, a, lda, b, ldb);
        return;
    }

    const TR alpha = *alpha_;
    /* α pre-scale of B (mirrors trsm_{L,R}.c GEMM_BETA pass). */
    if (alpha != 1.0L) egemm_beta_prepass(m, n, alpha, b, ldb);
    if (alpha == 0.0L) return;

    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(K_eff, &MC, &KC, &NC);

    /* Bound the per-thread pack buffers to what one band actually packs, not
     * the full cache-block params — every block is capped by the remaining
     * extent, so a thread never packs more than min(block, dim). The per-axis
     * dims are the SAME for both sides: Ap's MR-panel row axis is m (SIDE='L'
     * packs A tiles of ≤min(MC,m) rows; SIDE='R' packs B-row strips of
     * ≤min(MC,m_band) rows — bounded by m, NOT by K_eff: with m≫n the adaptive
     * MC grown for small K exceeds n and a K_eff bound overflows Ap), the KC
     * axis is the triangular dim K_eff, and the NC sweep walks B's columns n
     * on both sides. Bound by whole dims, never the per-thread chunk:
     * num_threads() is a request and a short team makes each band wider than
     * chunk. At N=64 the unbounded sizes were ~1MB Ap + ~2MB Bp (both past
     * glibc's 128KB mmap threshold → a per-call mmap/munmap + page-zeroing,
     * ×nthreads) for a 64×64 corner; bounding keeps them in the malloc arena
     * and is a no-op once the dims exceed the blocks. */
    const ptrdiff_t mc_eff = (MC < m)     ? MC : m;
    const ptrdiff_t kc_eff = (KC < K_eff) ? KC : K_eff;
    const ptrdiff_t nc_eff = (NC < n)     ? NC : n;
    const size_t ap_bytes = (size_t)blas_round_up(mc_eff, MR) * (size_t)kc_eff * sizeof(TR);
    const size_t bp_bytes = (size_t)kc_eff * (size_t)blas_round_up(nc_eff, NR) * sizeof(TR);

    /* Per-thread Ap/Bp scratch, allocated BEFORE the region: an in-region
     * alloc failure that skips a thread's loop body would deadlock the
     * others at no barrier here (there is none), but pre-allocating keeps
     * the failure path simple and race-free. */
    TR **Ap_arr = calloc((size_t)nthreads, sizeof(TR *));
    TR **Bp_arr = calloc((size_t)nthreads, sizeof(TR *));
    if (!Ap_arr || !Bp_arr) { free(Ap_arr); free(Bp_arr); return; }
    const size_t ap_al = ((ap_bytes + 63) & ~(size_t)63) + ETRI_PACK_GUARD;
    const size_t bp_al = ((bp_bytes + 63) & ~(size_t)63) + ETRI_PACK_GUARD;
    ptrdiff_t alloc_ok = 1;
    for (ptrdiff_t t = 0; t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, ap_al);
        Bp_arr[t] = aligned_alloc(64, bp_al);
        if (!Ap_arr[t] || !Bp_arr[t]) { alloc_ok = 0; break; }
        etri_pack_guard_poison(Ap_arr[t], ap_bytes, ap_al);
        etri_pack_guard_poison(Bp_arr[t], bp_bytes, bp_al);
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
        etri_pack_guard_check(Ap_arr[t], ap_bytes, ap_al, "etrsm_parallel Ap");
        etri_pack_guard_check(Bp_arr[t], bp_bytes, bp_al, "etrsm_parallel Bp");
        free(Ap_arr[t]);
        free(Bp_arr[t]);
    }
    free(Ap_arr);
    free(Bp_arr);
}

EPBLAS_FACADE_TRMM(etrsm, TR)
