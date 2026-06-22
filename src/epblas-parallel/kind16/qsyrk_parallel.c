/*
 * qsyrk_ — kind16 (REAL(KIND=16) / __float128) symmetric rank-k update:
 * the public Fortran entry and threading-orchestration half of the qsyrk
 * overlay (all the math lives in qsyrk_serial.c / qsyrk_kernel.h). Faithful
 * __float128 port of kind10 esyrk / OpenBLAS interface/syrk.c →
 * driver/level3/level3_syrk.c threading.
 *
 *   C := alpha · A · A^T + beta · C    (trans='N', A is N×K)
 *   C := alpha · A^T · A + beta · C    (trans='T', A is K×N)
 *
 * Threading: one outer `omp parallel`. The right operand (Bp) is packed once
 * per (js, ls) band under `omp single` and shared; each thread owns a
 * CONTIGUOUS slice of the M axis (the N output rows), sized by triangular AREA
 * (qtri_row_bounds) so every thread carries equal work — an equal-row split
 * caps the speedup at ~16/7 because the thread owning the fat end of the
 * triangle hogs 7/16 of the work. It then runs the MC-blocked is loop within
 * its range. The per-band UPLO clip (m_lo_eff / m_hi_eff) further trims the
 * row range, but every thread still executes every js/ls iteration so the `omp
 * single`/`omp barrier` pair stays collective. Partitioning the M axis into
 * per-thread chunks (not by MC-block count) keeps threads busy on small/thin
 * shapes.
 *
 * Nesting guard: when qsyrk_ is called from inside another routine's parallel
 * region, delegate to qsyrk_serial_ and open no team of our own — calling only
 * the *serial* kernel primitives means no nested team either way.
 */

#include "qsyrk_kernel.h"
#include "../common/blas_char.h"
#include "qtri_kernel.h"
#include "qgemm_kernel.h"   /* qgemm_choose_blocks / qgemm_round_up */
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef qsyrk_T T;

#define MR QSYRK_MR
#define NR QSYRK_NR

static void qsyrk_core(
    char uplo_c, char trans_c,
    ptrdiff_t n, ptrdiff_t k,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    const T *beta_,
    T *c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Inside another team → run serial, open no region of our own. */
    if (omp_in_parallel()) {
        qsyrk_serial(uplo_c, trans_c, n, k, alpha_, a, lda, beta_, c, ldc);
        return;
    }
#endif
    const T alpha = *alpha_, beta = *beta_;
    const char uplo  = blas_up(uplo_c);
    const char trans = blas_up(trans_c);

    if (n <= 0) return;

    /* Triangular beta pre-pass on the UPLO triangle of C only. */
    if (uplo == 'U') qsyrk_beta_u(n, beta, c, ldc);
    else             qsyrk_beta_l(n, beta, c, ldc);

    if (k == 0 || alpha == 0.0Q) return;

    ptrdiff_t MC, KC, NC;
    qgemm_choose_blocks(k, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)qgemm_round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)qgemm_round_up(NC, NR) * sizeof(T);

#ifdef _OPENMP
    ptrdiff_t nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif

    /* SYRK does ~ N^2 · K / 2 flops; tiny-cutoff sized to match qgemm. */
    long nnk = (long)n * (long)n * (long)k;
    if (nnk < 64L * 64L * 64L) nthreads = 1;

    /* Transpose: netlib-style unpacked inner-product, embarrassingly parallel
     * over the output columns (cyclic schedule balances the triangular load; no
     * shared pack, no barrier). Same code the serial entry runs at nthreads=1. */
    if (trans != 'N') {
#ifdef _OPENMP
        #pragma omp parallel for schedule(static, 1) num_threads(nthreads)
#endif
        for (ptrdiff_t j = 0; j < n; ++j)
            qsyrk_trans_col(j, uplo, n, k, alpha, a, lda, c, ldc);
        return;
    }

    /* Shared Bp, one private Ap per thread, allocated BEFORE the region: a
     * thread that skipped the loop on a failed in-region alloc would deadlock
     * the others at the Bp barrier. */
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

            /* M-axis (= N output rows) partition into per-thread chunks, sized
             * by triangular AREA so each thread carries equal work (an equal-
             * row split caps the speedup at ~16/7 — the fat-end thread hogs
             * 7/16 of a triangular output). */
            ptrdiff_t m_lo, m_hi;
            qtri_row_bounds(uplo, n, nth, tid, MR, &m_lo, &m_hi);

            for (ptrdiff_t js = 0; js < n; js += NC) {
                const ptrdiff_t jb = (n - js < NC) ? (n - js) : NC;

                /* UPLO clip of this thread's [m_lo, m_hi] for this js-band. */
                ptrdiff_t m_lo_eff = (uplo == 'L' && m_lo < js) ? js : m_lo;
                ptrdiff_t m_hi_eff = (uplo == 'U' && m_hi > js + jb) ? (js + jb) : m_hi;
                if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);
                if (m_lo_eff < m_lo) m_lo_eff = m_lo;

                for (ptrdiff_t ls = 0; ls < k; ls += KC) {
                    const ptrdiff_t pb = (k - ls < KC) ? (k - ls) : KC;

                    /* Pack the shared Bp = the same A in OCOPY shape. */
#ifdef _OPENMP
                    #pragma omp barrier
                    #pragma omp single
#endif
                    {
                        qtri_tcopy(pb, jb, &a[(size_t)ls * lda + js], lda, Bp);
                    }
                    /* implicit barrier at end of `single` → Bp safe to read */

                    for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                        const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                        qtri_tcopy(pb, min_i, &a[(size_t)ls * lda + is], lda, Ap);

                        if (uplo == 'U')
                            qsyrk_kernel_u(min_i, jb, pb, alpha, Ap, Bp,
                                           &c[(size_t)js * ldc + is], ldc,
                                           (ptrdiff_t)(is - js));
                        else
                            qsyrk_kernel_l(min_i, jb, pb, alpha, Ap, Bp,
                                           &c[(size_t)js * ldc + is], ldc,
                                           (ptrdiff_t)(is - js));
                    }
                }
            }
        }
    }

    for (ptrdiff_t t = 0; t < nthreads && Ap_arr; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}

EPBLAS_FACADE_SYRK(qsyrk, T, T)
