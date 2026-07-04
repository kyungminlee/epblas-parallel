/*
 * qsyr2k_ — kind16 (REAL(KIND=16) / __float128) symmetric rank-2k update:
 * the public Fortran entry and threading-orchestration half of the qsyr2k
 * overlay (all the math lives in qsyr2k_serial.c / qsyr2k_kernel.h). Faithful
 * __float128 port of kind10 esyr2k / OpenBLAS interface/syr2k.c →
 * driver/level3/level3_syr2k.c threading.
 *
 *   C := alpha·(A·B^T + B·A^T) + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·(A^T·B + B^T·A) + beta·C   (trans='T', A,B are K×N)
 *
 * Threading: one outer `omp parallel`. The two right operands (Bp_A = A in
 * B-shape, Bp_B = B in B-shape) are packed once per (js, ls) band under
 * `omp single` and shared; each thread owns a CONTIGUOUS slice of the M axis
 * (the N output rows), sized by triangular AREA (qtri_row_bounds) so every
 * thread carries equal work — an equal-row split caps the speedup at ~16/7
 * because the thread owning the fat end of the triangle hogs 7/16 of the work.
 * It then runs the MC-blocked is loop within its range, packing its own
 * Ap_A/Ap_B and doing the two kernel passes. The per-band UPLO clip (m_lo_eff
 * / m_hi_eff) trims each thread's row range, but every thread still executes
 * every js/ls iteration so the `omp single`/`omp barrier` pair stays
 * collective. Partitioning the M axis into per-thread chunks (not by MC-block
 * count) keeps threads busy on small/thin shapes.
 *
 * Nesting guard: when qsyr2k_ is called from inside another routine's parallel
 * region, delegate to qsyr2k_serial_ and open no team of our own — calling only
 * the *serial* kernel primitives means no nested team either way.
 */

#include "qsyr2k_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "qsyrk_kernel.h"   /* qsyrk_beta_{u,l} — shared triangular β pre-pass */
#include "qtri_kernel.h"
#include "qgemm_kernel.h"   /* qgemm_choose_blocks / blas_round_up */
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef qsyr2k_TR TR;

#define MR QSYR2K_MR
#define NR QSYR2K_NR

static void qsyr2k_core(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TR *alpha_,
    const TR *a, ptrdiff_t lda,
    const TR *b, ptrdiff_t ldb,
    const TR *beta_,
    TR *c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Inside another team → run serial, open no region of our own. */
    if (omp_in_parallel()) {
        qsyr2k_serial(uplo, trans, n, k, alpha_, a, lda, b, ldb,
                      beta_, c, ldc);
        return;
    }
#endif
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO  = blas_up(uplo);
    char TRANS = blas_up(trans);
    if (TRANS == 'C') TRANS = 'T';

    if (n <= 0) return;

    /* Degenerate update: pure triangular beta scale. */
    if (k == 0 || alpha == 0.0Q) {
        if (UPLO == 'U') qsyrk_beta_u(n, beta, c, ldc);
        else             qsyrk_beta_l(n, beta, c, ldc);
        return;
    }

    ptrdiff_t MC, KC, NC;
    qgemm_choose_blocks(k, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * (size_t)KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * (size_t)blas_round_up(NC, NR) * sizeof(TR);

#ifdef _OPENMP
    ptrdiff_t nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif

    /* SYR2K does ~ N^2 · K flops; tiny-cutoff sized to match qgemm. */
    long nnk = (long)n * (long)n * (long)k;
    if (nnk < 64L * 64L * 64L) nthreads = 1;

    /* Transpose: netlib-style unpacked inner-product, embarrassingly parallel
     * over the output columns (cyclic schedule balances the triangular load; no
     * shared packs, no barrier). Same code the serial entry runs at nthreads=1.
     * beta rides each column's store, so the old serial pre-region prescale
     * pass is gone (and its work is now spread across the team). */
    if (TRANS != 'N') {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static, 1) num_threads(nthreads)
#endif
        for (ptrdiff_t j = 0; j < n; ++j)
            qsyr2k_trans_col(j, UPLO, n, k, alpha, beta, a, lda, b, ldb, c, ldc);
        return;
    }

    /* NoTrans packed path: triangular beta pre-pass, then accumulate. */
    if (UPLO == 'U') qsyrk_beta_u(n, beta, c, ldc);
    else             qsyrk_beta_l(n, beta, c, ldc);

    /* Two shared B-packs (A and B in B-shape), two private A-packs per thread,
     * all allocated BEFORE the region: a thread that skipped the loop on a
     * failed in-region alloc would deadlock the others at the Bp barrier. */
    /* Persistent grow-only thread-local pack arena (Bp_A|Bp_B | per-thread
     * Ap_A/Ap_B slot pairs in one block, carved on the calling thread): a
     * per-call aligned_alloc+free of these mmap-threshold-sized buffers trips
     * glibc's trim heuristic and re-faults every touched page each call (see
     * etrsm_serial.c). Only the small pointer arrays stay per-call. */
    static __thread TR *g_pack = NULL;
    static __thread size_t g_pack_cap = 0;
    const size_t ap_al = (ap_bytes + 63) & ~(size_t)63;
    const size_t bp_al = (bp_bytes + 63) & ~(size_t)63;
    const size_t need  = 2 * bp_al + (size_t)nthreads * 2 * ap_al;
    if (need > g_pack_cap) {
        free(g_pack);
        size_t cap = need + (need >> 1);            /* 1.5x headroom to amortize regrow */
        cap = (cap + 63) & ~(size_t)63;
        g_pack = aligned_alloc(64, cap);
        g_pack_cap = g_pack ? cap : 0;
    }
    TR *Bp_A = g_pack;
    TR *Bp_B = (TR *)(void *)((char *)g_pack + bp_al);
    TR **Ap_A_arr = g_pack ? calloc((size_t)nthreads, sizeof(TR *)) : NULL;
    TR **Ap_B_arr = Ap_A_arr ? calloc((size_t)nthreads, sizeof(TR *)) : NULL;
    bool alloc_ok = (g_pack && Ap_A_arr && Ap_B_arr);
    for (ptrdiff_t t = 0; alloc_ok && t < nthreads; ++t) {
        Ap_A_arr[t] = (TR *)(void *)((char *)g_pack + 2 * bp_al + (size_t)t * 2 * ap_al);
        Ap_B_arr[t] = (TR *)(void *)((char *)g_pack + 2 * bp_al + (size_t)t * 2 * ap_al + ap_al);
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
            TR *Ap_A = Ap_A_arr[tid];
            TR *Ap_B = Ap_B_arr[tid];

            /* M-axis (= N output rows) partition into per-thread chunks, sized
             * by triangular AREA so each thread carries equal work (an equal-
             * row split caps the speedup at ~16/7 — the fat-end thread hogs
             * 7/16 of a triangular output). */
            ptrdiff_t m_lo, m_hi;
            qtri_row_bounds(UPLO, n, nth, tid, MR, &m_lo, &m_hi);

            for (ptrdiff_t js = 0; js < n; js += NC) {
                const ptrdiff_t jb = (n - js < NC) ? (n - js) : NC;

                /* UPLO clip of this thread's [m_lo, m_hi] for this js-band. */
                ptrdiff_t m_lo_eff = (UPLO == 'L' && m_lo < js) ? js : m_lo;
                ptrdiff_t m_hi_eff = (UPLO == 'U' && m_hi > js + jb) ? (js + jb) : m_hi;
                if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);
                if (m_lo_eff < m_lo) m_lo_eff = m_lo;

                for (ptrdiff_t ls = 0; ls < k; ls += KC) {
                    const ptrdiff_t pb = (k - ls < KC) ? (k - ls) : KC;

                    /* Pack the two shared B-side panels (A and B). */
#ifdef _OPENMP
                    #pragma omp barrier
                    #pragma omp single
#endif
                    {
                        qtri_tcopy(pb, jb, &a[(size_t)ls * lda + js], lda, Bp_A);
                        qtri_tcopy(pb, jb, &b[(size_t)ls * ldb + js], ldb, Bp_B);
                    }
                    /* implicit barrier at end of `single` → Bp safe to read */

                    for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                        const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                        qtri_tcopy(pb, min_i, &a[(size_t)ls * lda + is], lda, Ap_A);
                        qtri_tcopy(pb, min_i, &b[(size_t)ls * ldb + is], ldb, Ap_B);

                        TR *cij = &c[(size_t)js * ldc + is];
                        const ptrdiff_t off = (ptrdiff_t)(is - js);

                        /* Pass 1: alpha·A·B^T + symmetric diagonal merge. */
                        if (UPLO == 'U')
                            qsyr2k_kernel_u(min_i, jb, pb, alpha, Ap_A, Bp_B, cij, ldc, off, 1);
                        else
                            qsyr2k_kernel_l(min_i, jb, pb, alpha, Ap_A, Bp_B, cij, ldc, off, 1);

                        /* Pass 2: alpha·B·A^T into the off-diagonal strips. */
                        if (UPLO == 'U')
                            qsyr2k_kernel_u(min_i, jb, pb, alpha, Ap_B, Bp_A, cij, ldc, off, 0);
                        else
                            qsyr2k_kernel_l(min_i, jb, pb, alpha, Ap_B, Bp_A, cij, ldc, off, 0);
                    }
                }
            }
        }
    }

    free(Ap_A_arr);
    free(Ap_B_arr);
}

EPBLAS_FACADE_SYR2K(qsyr2k, TR, TR, TR)
