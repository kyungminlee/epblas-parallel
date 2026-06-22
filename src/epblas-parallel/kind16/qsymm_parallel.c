/*
 * qsymm_ — kind16 real (__float128) symmetric matrix-multiply, the public
 * Fortran entry and threading half of the qsymm overlay (see qsymm_kernel.h;
 * the packers + serial driver live in qsymm_serial.c).
 *
 * Threading: one outer `omp parallel`. The right operand (Bp) is packed once
 * per (jc, pc) under `omp single` and shared; each thread owns a CONTIGUOUS
 * slice of the M axis (m_chunk = ceil(M/nth) rounded to MR) and runs the
 * MC-blocked ic loop within it. Partitioning M into per-thread chunks — not
 * by MC-block count — keeps all threads busy even when M is small. Mirrors
 * the kind10 esymm overlay.
 *
 * Nesting guard: when qsymm_ is called from inside another routine's parallel
 * region, delegate to qsymm_serial_ and open no team of our own.
 */

#include "qsymm_kernel.h"
#include "../common/blas_char.h"
#include "qgemm_kernel.h"
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef qsymm_T T;

#define MR QGEMM_MR
#define NR QGEMM_NR

static void qsymm_core(
    char side_c, char uplo_c,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *restrict a, ptrdiff_t lda,
    const T *restrict b, ptrdiff_t ldb,
    const T *beta_,
    T *restrict c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Inside another team → run serial, open no region of our own. */
    if (omp_in_parallel()) {
        qsymm_serial(side_c, uplo_c, M, N, alpha_, a, lda, b, ldb, beta_,
                     c, ldc);
        return;
    }
#endif
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = blas_up(side_c);
    const char UPLO = blas_up(uplo_c);

    if (M <= 0 || N <= 0) return;

    /* alpha == 0 ⇒ pure C := beta*C, no GEMM. Rare; not worth a team. */
    if (alpha == 0.0Q) { qgemm_beta_prepass(M, N, beta, c, ldc); return; }

    /* The C := beta*C pre-pass is NOT done here: each thread applies it to
     * its own M-row slice inside the region, so it scales with the team. A
     * thread only ever touches its own rows — for both the beta pass and the
     * kernel writes — so no barrier is needed between them. */

    const ptrdiff_t K = (SIDE == 'L') ? M : N;

    ptrdiff_t MC, KC, NC;
    qgemm_choose_blocks(K, &MC, &KC, &NC);

#ifdef _OPENMP
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif
    /* Tiny problems: the team setup + Bp barrier cost outweighs the split. */
    if ((long)M * (long)N * (long)K < 64L * 64L * 64L) nthreads = 1;

    const size_t ap_bytes = (size_t)qgemm_round_up(MC, MR) * KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * qgemm_round_up(NC, NR) * sizeof(T);

    /* Pre-allocate the shared Bp and one private Ap per thread BEFORE the
     * region: a thread that skipped the loop on a failed in-region alloc
     * would deadlock the others at the Bp barrier. */
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    T **Ap_arr = Bp ? calloc((size_t)nthreads, sizeof(T *)) : NULL;
    bool alloc_ok = (Bp && Ap_arr);
    for (ptrdiff_t t = 0; alloc_ok && t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t]) alloc_ok = 0;
    }
    if (alloc_ok) {
#ifdef _OPENMP
        #pragma omp parallel num_threads(nthreads)
#endif
        {
#ifdef _OPENMP
            const ptrdiff_t tid = omp_get_thread_num();
            const ptrdiff_t nth = omp_get_num_threads();
#else
            const ptrdiff_t tid = 0, nth = 1;
#endif
            T *Ap = Ap_arr[tid];

            const ptrdiff_t m_chunk = qgemm_round_up((M + nth - 1) / nth, MR);
            const ptrdiff_t m_lo = tid * m_chunk;
            ptrdiff_t m_hi = m_lo + m_chunk;
            if (m_hi > M) m_hi = M;

            /* C := beta*C over this thread's rows only (handles beta 0/1). */
            if (beta != 1.0Q && m_lo < m_hi) {
                for (ptrdiff_t j = 0; j < N; ++j) {
                    T *cj = &c[(size_t)j * ldc];
                    if (beta == 0.0Q) for (ptrdiff_t i = m_lo; i < m_hi; ++i) cj[i]  = 0.0Q;
                    else              for (ptrdiff_t i = m_lo; i < m_hi; ++i) cj[i] *= beta;
                }
            }

            for (ptrdiff_t jc = 0; jc < N; jc += NC) {
                const ptrdiff_t jb = (N - jc < NC) ? (N - jc) : NC;
                for (ptrdiff_t pc = 0; pc < K; pc += KC) {
                    const ptrdiff_t pb = (K - pc < KC) ? (K - pc) : KC;
#ifdef _OPENMP
                    #pragma omp barrier
                    #pragma omp single
#endif
                    {
                        if (SIDE == 'L')
                            qgemm_pack_B(b, ldb, pc, jc, pb, jb, 'N', Bp);
                        else
                            qsymm_pack_b_sym(a, lda, pc, jc, pb, jb, UPLO, Bp);
                    }
                    /* implicit barrier at end of `single` → Bp safe to read */

                    for (ptrdiff_t ic = m_lo; ic < m_hi; ic += MC) {
                        const ptrdiff_t ib = (m_hi - ic < MC) ? (m_hi - ic) : MC;
                        if (SIDE == 'L')
                            qsymm_pack_a_sym(a, lda, ic, pc, ib, pb, UPLO, Ap);
                        else
                            qgemm_pack_A(b, ldb, ic, pc, ib, pb, 'N', Ap);

                        qgemm_macro_kernel(ib, jb, pb, alpha, Ap, Bp,
                                           &c[(size_t)jc * ldc + ic], ldc);
                    }
                }
            }
        }
    }

    for (ptrdiff_t t = 0; t < nthreads && Ap_arr; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}

EPBLAS_FACADE_SYMM(qsymm, T)
