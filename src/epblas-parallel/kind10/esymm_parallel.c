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
 * either way, the libgomp barrier-wedge cure.
 */

#include "esymm_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "egemm_kernel.h"
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef esymm_TR TR;

#define MR EGEMM_MR
#define NR EGEMM_NR

static void esymm_core(
    char side, char uplo,
    ptrdiff_t m, ptrdiff_t n,
    const TR *alpha_,
    const TR *restrict a, ptrdiff_t lda,
    const TR *restrict b, ptrdiff_t ldb,
    const TR *beta_,
    TR *restrict c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Inside another team → run serial, open no region of our own. */
    if (omp_in_parallel()) {
        esymm_serial(side, uplo, m, n, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const TR alpha = *alpha_, beta = *beta_;
    const char SIDE = blas_up(side);
    const char UPLO = blas_up(uplo);

    if (m <= 0 || n <= 0) return;

    /* alpha == 0 ⇒ pure C := beta*C, no GEMM. Rare; not worth a team. */
    if (alpha == 0.0L) { egemm_beta_prepass(m, n, beta, c, ldc); return; }

    /* The C := beta*C pre-pass is NOT done here: each thread applies it to
     * its own M-row slice inside the region (see below), so it scales with
     * the team instead of being a serial Amdahl tax. A thread only ever
     * touches its own rows — for both the beta pass and the kernel writes —
     * so no barrier is needed between them. */

    const ptrdiff_t k = (SIDE == 'L') ? m : n;

    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(k, &MC, &KC, &NC);

#ifdef _OPENMP
    ptrdiff_t nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif
    /* Tiny problems: the team setup + Bp barrier cost outweighs the split. */
    if ((long)m * (long)n * (long)k < 64L * 64L * 64L) nthreads = 1;

    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * blas_round_up(NC, NR) * sizeof(TR);

    /* Shared Bp + one private Ap slot per thread, carved BEFORE the region
     * from a persistent grow-only thread-local arena on the calling thread
     * (a per-call aligned_alloc+free of these mmap-threshold-sized buffers
     * trips glibc's trim heuristic and re-faults every touched page each
     * call — see etrsm_serial.c); a thread that skipped the loop on a failed
     * in-region alloc would deadlock the others at the Bp barrier. Only the
     * small Ap_arr pointer array stays per-call. */
    static __thread TR *g_pack = NULL;
    static __thread size_t g_pack_cap = 0;
    const size_t ap_al = (ap_bytes + 63) & ~(size_t)63;
    const size_t bp_al = (bp_bytes + 63) & ~(size_t)63;
    const size_t need  = bp_al + (size_t)nthreads * ap_al;
    if (need > g_pack_cap) {
        free(g_pack);
        size_t cap = need + (need >> 1);            /* 1.5× headroom to amortize regrow */
        cap = (cap + 63) & ~(size_t)63;
        g_pack = aligned_alloc(64, cap);
        g_pack_cap = g_pack ? cap : 0;
    }
    TR *Bp = g_pack;
    TR **Ap_arr = g_pack ? calloc((size_t)nthreads, sizeof(TR *)) : NULL;
    ptrdiff_t alloc_ok = (g_pack && Ap_arr);
    for (ptrdiff_t t = 0; alloc_ok && t < nthreads; ++t)
        Ap_arr[t] = (TR *)(void *)((char *)g_pack + bp_al + (size_t)t * ap_al);
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
            TR *Ap = Ap_arr[tid];

            const ptrdiff_t m_chunk = blas_round_up((m + nth - 1) / nth, MR);
            const ptrdiff_t m_lo = tid * m_chunk;
            ptrdiff_t m_hi = m_lo + m_chunk;
            if (m_hi > m) m_hi = m;

            /* C := beta*C over this thread's rows only (handles beta 0/1). */
            if (beta != 1.0L && m_lo < m_hi) {
                for (ptrdiff_t j = 0; j < n; ++j) {
                    TR *cj = &c[(size_t)j * ldc];
                    if (beta == 0.0L) for (ptrdiff_t i = m_lo; i < m_hi; ++i) cj[i]  = 0.0L;
                    else              for (ptrdiff_t i = m_lo; i < m_hi; ++i) cj[i] *= beta;
                }
            }

            for (ptrdiff_t jc = 0; jc < n; jc += NC) {
                const ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
                for (ptrdiff_t pc = 0; pc < k; pc += KC) {
                    const ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
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

                    for (ptrdiff_t ic = m_lo; ic < m_hi; ic += MC) {
                        const ptrdiff_t ib = (m_hi - ic < MC) ? (m_hi - ic) : MC;
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

    free(Ap_arr);
}

EPBLAS_FACADE_SYMM(esymm, TR)
