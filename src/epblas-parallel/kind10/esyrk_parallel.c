/*
 * esyrk_ — kind10 (REAL(KIND=10) / long double) symmetric rank-k update:
 * the public Fortran entry and threading-orchestration half of the esyrk
 * overlay (all the math lives in esyrk_serial.c / esyrk_kernel.h). Faithful
 * port of OpenBLAS interface/syrk.c → driver/level3/level3_syrk.c threading.
 *
 *   C := alpha · A · A^T + beta · C    (trans='N', A is N×K)
 *   C := alpha · A^T · A + beta · C    (trans='T', A is K×N)
 *
 * Threading: one outer `omp parallel`. The right operand (Bp) is packed once
 * per (js, ls) band under `omp single` and shared; each thread owns a
 * CONTIGUOUS slice of the M axis (the N output rows, m_chunk = ceil(N/nth)
 * rounded to MR) and runs the MC-blocked is loop within it. The per-band UPLO
 * clip (m_lo_eff / m_hi_eff) further trims each thread's row range, but every
 * thread still executes every js/ls iteration so the `omp single`/`omp
 * barrier` pair stays collective. Partitioning the M axis into per-thread
 * chunks (not by MC-block count) keeps threads busy on small/thin shapes.
 *
 * Nesting guard: when esyrk_ is called from inside another routine's parallel
 * region, delegate to esyrk_serial and open no team of our own — calling only
 * the *serial* kernel primitives means no nested team either way, the libgomp
 * barrier-wedge cure.
 */

#include "esyrk_kernel.h"
#include "etri_kernel.h"
#include "egemm_kernel.h"   /* egemm_choose_blocks / egemm_round_up */
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef esyrk_T T;

#define MR ESYRK_MR
#define NR ESYRK_NR

void esyrk_(
    const char *uplo_p, const char *trans_p,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
#ifdef _OPENMP
    /* Inside another team → run serial, open no region of our own. */
    if (omp_in_parallel()) {
        const ptrdiff_t n_pt = *n_, k_pt = *k_, lda_pt = *lda_, ldc_pt = *ldc_;
        esyrk_serial(uplo_p, trans_p, &n_pt, &k_pt, alpha_, a, &lda_pt, beta_,
                     c, &ldc_pt, uplo_len, trans_len);
        return;
    }
#endif
    (void)uplo_len; (void)trans_len;
    const ptrdiff_t N = *n_, K = *k_;
    const T alpha = *alpha_, beta = *beta_;
    const ptrdiff_t lda = *lda_, ldc = *ldc_;
    const ptrdiff_t uplo  = (char)toupper((unsigned char)*uplo_p);
    const ptrdiff_t trans = (char)toupper((unsigned char)*trans_p);

    if (N <= 0) return;

    /* Triangular beta pre-pass on the UPLO triangle of C only. */
    if (uplo == 'U') esyrk_beta_u((ptrdiff_t)N, beta, c, (ptrdiff_t)ldc);
    else             esyrk_beta_l((ptrdiff_t)N, beta, c, (ptrdiff_t)ldc);

    if (K == 0 || alpha == 0.0L) return;

    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(K, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)egemm_round_up(NC, NR) * sizeof(T);

#ifdef _OPENMP
    ptrdiff_t nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif

    /* SYRK does ~ N^2 · K / 2 flops; tiny-cutoff sized to match egemm. */
    long nnk = (long)N * (long)N * (long)K;
    if (nnk < 64L * 64L * 64L) nthreads = 1;

    /* Shared Bp, one private Ap per thread, allocated BEFORE the region: a
     * thread that skipped the loop on a failed in-region alloc would deadlock
     * the others at the Bp barrier. */
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    T **Ap_arr = Bp ? calloc((size_t)nthreads, sizeof(T *)) : NULL;
    ptrdiff_t alloc_ok = (Bp && Ap_arr);
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

            /* M-axis (= N output rows) partition into per-thread chunks. */
            const ptrdiff_t m_chunk = egemm_round_up((N + nth - 1) / nth, MR);
            const ptrdiff_t m_lo = tid * m_chunk;
            ptrdiff_t m_hi = m_lo + m_chunk;
            if (m_hi > N) m_hi = N;

            for (ptrdiff_t js = 0; js < N; js += NC) {
                const ptrdiff_t jb = (N - js < NC) ? (N - js) : NC;

                /* UPLO clip of this thread's [m_lo, m_hi] for this js-band. */
                ptrdiff_t m_lo_eff = (uplo == 'L' && m_lo < js) ? js : m_lo;
                ptrdiff_t m_hi_eff = (uplo == 'U' && m_hi > js + jb) ? (js + jb) : m_hi;
                if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);
                if (m_lo_eff < m_lo) m_lo_eff = m_lo;

                for (ptrdiff_t ls = 0; ls < K; ls += KC) {
                    const ptrdiff_t pb = (K - ls < KC) ? (K - ls) : KC;

                    /* Pack the shared Bp = the same A in OCOPY shape. */
#ifdef _OPENMP
                    #pragma omp barrier
                    #pragma omp single
#endif
                    {
                        if (trans == 'N')
                            etri_tcopy(pb, jb, &a[(size_t)ls * lda + js], lda, Bp);
                        else
                            etri_ncopy(pb, jb, &a[(size_t)js * lda + ls], lda, Bp);
                    }
                    /* implicit barrier at end of `single` → Bp safe to read */

                    for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                        const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                        if (trans == 'N')
                            etri_tcopy(pb, min_i, &a[(size_t)ls * lda + is], lda, Ap);
                        else
                            etri_ncopy(pb, min_i, &a[(size_t)is * lda + ls], lda, Ap);

                        if (uplo == 'U')
                            esyrk_kernel_u(min_i, jb, pb, alpha, Ap, Bp,
                                           &c[(size_t)js * ldc + is], ldc,
                                           (ptrdiff_t)(is - js));
                        else
                            esyrk_kernel_l(min_i, jb, pb, alpha, Ap, Bp,
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
