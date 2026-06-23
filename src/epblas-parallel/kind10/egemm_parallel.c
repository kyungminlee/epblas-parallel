/*
 * egemm_ — kind10 (REAL(KIND=10) / 80-bit long double) GEMM, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * egemm_serial.c (packers, micro-kernel, beta pre-pass, block policy),
 * shared through egemm_kernel.h.
 *
 *   C := alpha * op(A) * op(B) + beta * C
 *
 * Two regimes:
 *   - Called from inside another OpenMP team (the L3 family — etrsm,
 *     etrmm, egemmtr, esyrk, esymm, esyr2k — runs egemm trailing updates
 *     inside its own `omp parallel`): open NO nested region. Run the
 *     single-thread kernel in the calling thread. This is the cure for
 *     the libgomp barrier wedge:
 *     the old code opened a nested team-of-1 GOMP_parallel per trailing
 *     block, and that create/destroy churn under libgomp's default
 *     barrier spin window tripped a lost-wakeup race that livelocked
 *     ~16% of etrsm OMP=4 runs.
 *   - Called at top level: fan the M-axis across an OpenMP team. Bp is
 *     packed once per (jc, pc) under `omp single` (implicit barrier) and
 *     shared; each thread keeps a private Ap and takes an `omp for`
 *     slice of the ic loop.
 *
 * Fortran ABI:
 *   - subroutine name lowercased + trailing underscore: `egemm_`
 *   - scalars passed by pointer
 *   - character args followed by hidden trailing `size_t` lengths
 *   - REAL(KIND=10) ↔ `long double` (x86-64 80-bit extended)
 */

#include "egemm_kernel.h"
#include "../common/blas_char.h"
#include "../common/epblas_facade.h"
#include "../common/blas_math.h"
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#include <stddef.h>
#endif

typedef egemm_TR TR;

#define MR EGEMM_MR
#define NR EGEMM_NR

static void egemm_core(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const TR *alpha_,
    const TR *a, ptrdiff_t lda,
    const TR *b, ptrdiff_t ldb,
    const TR *beta_,
    TR *c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Already inside a team → run serially in this thread, no nested
     * region. Removing the team-of-1 GOMP_parallel churn is what cures
     * the wedge; it is also slightly faster (no per-call team setup) and
     * hardens egemm against oversubscription if OMP_NESTED is ever set. */
    if (omp_in_parallel()) {
        egemm_serial(transa, transb, m, n, k, alpha_, a, lda,
                     b, ldb, beta_, c, ldc);
        return;
    }
#endif

    const TR alpha = *alpha_, beta = *beta_;
    const char ta = blas_trans_real(transa);
    const char tb = blas_trans_real(transb);

    if (m <= 0 || n <= 0) return;

    egemm_beta_prepass(m, n, beta, c, ldc);   /* handles K==0 / alpha==0 */
    if (alpha == 0.0L || k == 0) return;

    /* Fast path: TA='T' (≡'C'), TB='N'. Stride-1 dot, no packing — but only
     * for skinny problems (see egemm_tn_use_fast). For non-trivial K the
     * blocked packed path below is faster per FLOP (the single fp80 dot
     * accumulator in fast_col serializes the fadd latency the blocked
     * kernel's 4 chains hide) and threads over M just as well. */
    if (ta == 'T' && tb == 'N' && egemm_tn_use_fast(m, n, k)) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t j2 = 0; j2 < n; ++j2)
            egemm_fast_col(j2, m, k, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(k, &MC, &KC, &NC);
#ifdef _OPENMP
    /* This entry partitions the ic loop across the team via `omp for`, so the
     * number of ic-blocks (ceil(M/MC)) must be >= the team size or threads
     * sit idle. egemm_choose_blocks' adaptive-MC growth can collapse a
     * small-M problem to a single block (e.g. N=K=128 grew MC to M -> one
     * block -> no threading, par4 ~= par1). Cap MC so M splits into at least
     * `nthr` blocks. The cap is local to this threaded entry; the shared
     * policy (used by the serial path and the L3 routines, which partition
     * other axes) is untouched. Only shrinks MC; stays a multiple of MR. */
    const ptrdiff_t nthr = omp_get_max_threads();
    if (nthr > 1) {
        ptrdiff_t cap = blas_round_up((m + nthr - 1) / nthr, MR);
        if (cap < MR) cap = MR;
        if (MC > cap) MC = cap;
    }
#endif

    /*
     * Threading: single outer `omp parallel`, shared Bp packed once per
     * (jc, pc) via `omp single` (implicit barrier), then `omp for` over
     * the ic loop. Each thread keeps a private Ap.
     *
     * Splitting along ic (the M axis) keeps parallelism even when
     * N ≤ NC (only one jc band); keeping Bp shared avoids the per-tile
     * re-packing a naive collapse(2) would force. Effective parallelism
     * is bounded by (M / MC) per jc-band — ample for square problems.
     */
    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * blas_round_up(NC, NR) * sizeof(TR);
    TR *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) return;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        TR *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (Ap) {
            for (ptrdiff_t jc = 0; jc < n; jc += NC) {
                const ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
                for (ptrdiff_t pc = 0; pc < k; pc += KC) {
                    const ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
#ifdef _OPENMP
                    #pragma omp single
#endif
                    egemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                    /* implicit barrier at end of `single` makes Bp safe to
                     * read in the for below. */
#ifdef _OPENMP
                    #pragma omp for schedule(static)
#endif
                    for (ptrdiff_t ic = 0; ic < m; ic += MC) {
                        const ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
                        egemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                        egemm_macro_kernel(ib, jb, pb, alpha, Ap, Bp,
                                           &c[(size_t)jc * ldc + ic], ldc);
                    }
                    /* implicit barrier at end of `for` keeps Bp stable
                     * for the next (jc, pc) iteration. */
                }
            }
        }
        free(Ap);
    }
    free(Bp);
}

EPBLAS_FACADE_GEMM(egemm, TR)
