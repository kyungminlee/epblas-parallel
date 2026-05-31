/*
 * esymm_ — kind10 real (long double) symmetric matrix-multiply, the public
 * Fortran entry and threading half of the esymm overlay (see esymm_kernel.h;
 * the packers + serial driver live in esymm_serial.c).
 *
 * Threading: one outer `omp parallel`. The right operand (Bp) is packed once
 * per (jc, pc) under `omp single` and shared; each thread owns a CONTIGUOUS
 * slice of the M axis (m_chunk = ceil(M/nth) rounded to MR) and runs the
 * MC-blocked ic loop within it. Partitioning M into per-thread chunks — not
 * by MC-block count — keeps all threads busy even when M is small (an
 * omp-for over ic-by-MC starves when ceil(M/MC) < nth). This mirrors the
 * OpenBLAS-overlay esymm and is what makes it robust on small/thin shapes.
 *
 * Nesting guard: when esymm_ is called from inside another routine's parallel
 * region, delegate to esymm_serial and open no team of our own — calling only
 * the *serial* egemm kernel primitives (never egemm_) means no nested team
 * either way, the libgomp barrier-wedge cure (memory project-etrsm-omp4-wedge).
 */

#include "esymm_kernel.h"
#include "egemm_kernel.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef esymm_T T;

#define MR EGEMM_MR
#define NR EGEMM_NR

void esymm_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict b, const int *ldb_,
    const T *beta_,
    T *restrict c, const int *ldc_,
    size_t side_len, size_t uplo_len)
{
#ifdef _OPENMP
    /* Inside another team → run serial, open no region of our own. */
    if (omp_in_parallel()) {
        esymm_serial(side, uplo, m_, n_, alpha_, a, lda_, b, ldb_, beta_,
                     c, ldc_, side_len, uplo_len);
        return;
    }
#endif
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = (char)toupper((unsigned char)*side);
    const char UPLO = (char)toupper((unsigned char)*uplo);

    if (M <= 0 || N <= 0) return;

    /* alpha == 0 ⇒ pure C := beta*C, no GEMM. Rare; not worth a team. */
    if (alpha == 0.0L) { egemm_beta_prepass(M, N, beta, c, ldc); return; }

    /* The C := beta*C pre-pass is NOT done here: each thread applies it to
     * its own M-row slice inside the region (see below), so it scales with
     * the team instead of being a serial Amdahl tax. A thread only ever
     * touches its own rows — for both the beta pass and the kernel writes —
     * so no barrier is needed between them. */

    const int K = (SIDE == 'L') ? M : N;

    int MC, KC, NC;
    egemm_choose_blocks(K, &MC, &KC, &NC);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
#endif
    /* Tiny problems: the team setup + Bp barrier cost outweighs the split. */
    if ((long)M * (long)N * (long)K < 64L * 64L * 64L) nthreads = 1;

    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * egemm_round_up(NC, NR) * sizeof(T);

    /* Pre-allocate the shared Bp and one private Ap per thread BEFORE the
     * region: a thread that skipped the loop on a failed in-region alloc
     * would deadlock the others at the Bp barrier. */
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    T **Ap_arr = Bp ? calloc((size_t)nthreads, sizeof(T *)) : NULL;
    int alloc_ok = (Bp && Ap_arr);
    for (int t = 0; alloc_ok && t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t]) alloc_ok = 0;
    }
    if (alloc_ok) {
#ifdef _OPENMP
        #pragma omp parallel num_threads(nthreads)
#endif
        {
#ifdef _OPENMP
            const int tid = omp_get_thread_num();
            const int nth = omp_get_num_threads();
#else
            const int tid = 0, nth = 1;
#endif
            T *Ap = Ap_arr[tid];

            const int m_chunk = egemm_round_up((M + nth - 1) / nth, MR);
            const int m_lo = tid * m_chunk;
            int m_hi = m_lo + m_chunk;
            if (m_hi > M) m_hi = M;

            /* C := beta*C over this thread's rows only (handles beta 0/1). */
            if (beta != 1.0L && m_lo < m_hi) {
                for (int j = 0; j < N; ++j) {
                    T *cj = &c[(size_t)j * ldc];
                    if (beta == 0.0L) for (int i = m_lo; i < m_hi; ++i) cj[i]  = 0.0L;
                    else              for (int i = m_lo; i < m_hi; ++i) cj[i] *= beta;
                }
            }

            for (int jc = 0; jc < N; jc += NC) {
                const int jb = (N - jc < NC) ? (N - jc) : NC;
                for (int pc = 0; pc < K; pc += KC) {
                    const int pb = (K - pc < KC) ? (K - pc) : KC;
#ifdef _OPENMP
                    #pragma omp barrier
                    #pragma omp single
#endif
                    {
                        if (SIDE == 'L')
                            egemm_pack_B(b, ldb, pc, jc, pb, jb, 'N', Bp);
                        else
                            esymm_pack_b_sym(a, lda, pc, jc, pb, jb, UPLO, Bp);
                    }
                    /* implicit barrier at end of `single` → Bp safe to read */

                    for (int ic = m_lo; ic < m_hi; ic += MC) {
                        const int ib = (m_hi - ic < MC) ? (m_hi - ic) : MC;
                        if (SIDE == 'L')
                            esymm_pack_a_sym(a, lda, ic, pc, ib, pb, UPLO, Ap);
                        else
                            egemm_pack_A(b, ldb, ic, pc, ib, pb, 'N', Ap);

                        egemm_macro_kernel(ib, jb, pb, alpha, Ap, Bp,
                                           &c[(size_t)jc * ldc + ic], ldc);
                    }
                }
            }
        }
    }

    for (int t = 0; t < nthreads && Ap_arr; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}
