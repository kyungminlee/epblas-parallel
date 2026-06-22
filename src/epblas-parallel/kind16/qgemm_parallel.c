/*
 * qgemm_ — kind16 (REAL(KIND=16) / __float128) GEMM, public Fortran entry.
 * THREADING ORCHESTRATION ONLY: all the math lives in qgemm_serial.c (the
 * packers, MR×NR micro-kernel, beta pre-pass, block policy), shared through
 * qgemm_kernel.h.
 *
 *   C := alpha * op(A) * op(B) + beta * C
 *
 * Two regimes (mirrors the kind10 egemm overlay):
 *   - Called from inside another OpenMP team (the L3 family — qtrsm runs
 *     qgemm trailing updates inside its own `omp parallel`): open NO nested
 *     region. Run the single-thread kernel in the calling thread.
 *   - Called at top level: fan the M-axis across an OpenMP team. Bp is
 *     packed once per (jc, pc) under `omp single` (implicit barrier) and
 *     shared; each thread keeps a private Ap and takes an `omp for` slice
 *     of the ic loop.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths; REAL(KIND=16)
 * ↔ __float128.
 */

#include "qgemm_kernel.h"
#include "../common/epblas_facade.h"
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#include <stddef.h>
#include "../common/blas_omp.h"
#endif

typedef qgemm_T T;

#define MR QGEMM_MR
#define NR QGEMM_NR

static void qgemm_core(
    char transa, char transb,
    ptrdiff_t M, ptrdiff_t N, ptrdiff_t K,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    const T *beta_,
    T *c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Already inside a team → run serially in this thread, no nested region.
     * qgemm_serial shares the by-value core ABI, so forward the args. */
    if (omp_in_parallel()) {
        qgemm_serial(transa, transb, M, N, K, alpha_, a, lda,
                     b, ldb, beta_, c, ldc);
        return;
    }
#endif

    const T alpha = *alpha_, beta = *beta_;
    const char ta = qgemm_trans_code(transa);
    const char tb = qgemm_trans_code(transb);

    if (M <= 0 || N <= 0) return;

    qgemm_beta_prepass(M, N, beta, c, ldc);   /* handles K==0 / alpha==0 */
    if (alpha == 0.0Q || K == 0) return;

    /* TA='T' (≡'C'), TB='N': unpacked stride-1 dot, threaded over columns.
     * For __float128 this beats the blocked packed path at every measured
     * K and size, serial AND threaded — A^T already gives stride-1 access in
     * the contraction index for both operands, so packing is pure overhead
     * (no SIMD to feed, and the MR×NR 4-accumulator ILP does not outrun the
     * simpler dot). par/mig 1.12->1.00 serial; par/ob 1.00->0.89 omp4. */
    if (ta == 'T' && tb == 'N') {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (ptrdiff_t j2 = 0; j2 < N; ++j2)
            qgemm_fast_col(j2, M, K, alpha, a, lda, b, ldb, c, ldc);
        return;
    }

    ptrdiff_t MC, KC, NC;
    qgemm_choose_blocks(K, &MC, &KC, &NC);
#ifdef _OPENMP
    /* The ic loop is partitioned across the team via `omp for`, so the number
     * of ic-blocks (ceil(M/MC)) must be >= the team size or threads sit idle.
     * Cap MC so M splits into at least `nthr` blocks (only shrinks MC; stays
     * a multiple of MR). The cap is local to this threaded entry. */
    const ptrdiff_t nthr = blas_omp_max_threads();
    if (nthr > 1) {
        ptrdiff_t cap = qgemm_round_up((M + nthr - 1) / nthr, MR);
        if (cap < MR) cap = MR;
        if (MC > cap) MC = cap;
    }
#endif

    /* Single outer `omp parallel`, shared Bp packed once per (jc, pc) via
     * `omp single` (implicit barrier), then `omp for` over the ic loop. */
    const size_t ap_bytes = (size_t)qgemm_round_up(MC, MR) * KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * qgemm_round_up(NC, NR) * sizeof(T);
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) return;
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        T *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (Ap) {
            for (ptrdiff_t jc = 0; jc < N; jc += NC) {
                const ptrdiff_t jb = (N - jc < NC) ? (N - jc) : NC;
                for (ptrdiff_t pc = 0; pc < K; pc += KC) {
                    const ptrdiff_t pb = (K - pc < KC) ? (K - pc) : KC;
#ifdef _OPENMP
                    #pragma omp single
#endif
                    qgemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                    /* implicit barrier at end of `single` makes Bp safe to
                     * read in the for below. */
#ifdef _OPENMP
                    #pragma omp for schedule(static)
#endif
                    for (ptrdiff_t ic = 0; ic < M; ic += MC) {
                        const ptrdiff_t ib = (M - ic < MC) ? (M - ic) : MC;
                        qgemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                        qgemm_macro_kernel(ib, jb, pb, alpha, Ap, Bp,
                                           &c[(size_t)jc * ldc + ic], ldc);
                    }
                    /* implicit barrier at end of `for` keeps Bp stable for
                     * the next (jc, pc) iteration. */
                }
            }
        }
        free(Ap);
    }
    free(Bp);
}

EPBLAS_FACADE_GEMM(qgemm, T)
