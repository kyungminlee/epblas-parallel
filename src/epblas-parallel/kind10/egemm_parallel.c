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
 *     the libgomp barrier wedge (see memory project-etrsm-omp4-wedge):
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
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef egemm_T T;

#define MR EGEMM_MR
#define NR EGEMM_NR

void egemm_(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t transa_len, size_t transb_len)
{
#ifdef _OPENMP
    /* Already inside a team → run serially in this thread, no nested
     * region. Removing the team-of-1 GOMP_parallel churn is what cures
     * the wedge; it is also slightly faster (no per-call team setup) and
     * hardens egemm against oversubscription if OMP_NESTED is ever set. */
    if (omp_in_parallel()) {
        egemm_serial(transa, transb, m_, n_, k_, alpha_, a, lda_,
                     b, ldb_, beta_, c, ldc_, transa_len, transb_len);
        return;
    }
#endif

    const int M = *m_, N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const int ta = egemm_trans_code(transa, transa_len);
    const int tb = egemm_trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    egemm_beta_prepass(M, N, beta, c, ldc);   /* handles K==0 / alpha==0 */
    if (alpha == 0.0L || K == 0) return;

    /* Fast path: TA='T' (≡'C'), TB='N'. Stride-1 dot, no packing — but only
     * for skinny problems (see egemm_tn_use_fast). For non-trivial K the
     * blocked packed path below is faster per FLOP (the single fp80 dot
     * accumulator in fast_col serializes the fadd latency the blocked
     * kernel's 4 chains hide) and threads over M just as well. */
    if (ta == 'T' && tb == 'N' && egemm_tn_use_fast(M, N, K)) {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (int j2 = 0; j2 < N; ++j2)
            egemm_fast_col(j2, M, K, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

    int MC, KC, NC;
    egemm_choose_blocks(K, &MC, &KC, &NC);
#ifdef _OPENMP
    /* This entry partitions the ic loop across the team via `omp for`, so the
     * number of ic-blocks (ceil(M/MC)) must be >= the team size or threads
     * sit idle. egemm_choose_blocks' adaptive-MC growth can collapse a
     * small-M problem to a single block (e.g. N=K=128 grew MC to M -> one
     * block -> no threading, par4 ~= par1). Cap MC so M splits into at least
     * `nthr` blocks. The cap is local to this threaded entry; the shared
     * policy (used by the serial path and the L3 routines, which partition
     * other axes) is untouched. Only shrinks MC; stays a multiple of MR. */
    const int nthr = omp_get_max_threads();
    if (nthr > 1) {
        int cap = egemm_round_up((M + nthr - 1) / nthr, MR);
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
    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * egemm_round_up(NC, NR) * sizeof(T);
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) return;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        T *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (Ap) {
            for (int jc = 0; jc < N; jc += NC) {
                const int jb = (N - jc < NC) ? (N - jc) : NC;
                for (int pc = 0; pc < K; pc += KC) {
                    const int pb = (K - pc < KC) ? (K - pc) : KC;
#ifdef _OPENMP
                    #pragma omp single
#endif
                    egemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                    /* implicit barrier at end of `single` makes Bp safe to
                     * read in the for below. */
#ifdef _OPENMP
                    #pragma omp for schedule(static)
#endif
                    for (int ic = 0; ic < M; ic += MC) {
                        const int ib = (M - ic < MC) ? (M - ic) : MC;
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
