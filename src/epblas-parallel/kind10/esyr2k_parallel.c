/*
 * esyr2k_ — kind10 (REAL(KIND=10) / long double) symmetric rank-2k update:
 * the public Fortran entry and threading-orchestration half of the esyr2k
 * overlay (all the math lives in esyr2k_serial.c / esyr2k_kernel.h). Faithful
 * port of OpenBLAS interface/syr2k.c → driver/level3/level3_syr2k.c threading.
 *
 *   C := alpha·(A·B^T + B·A^T) + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·(A^T·B + B^T·A) + beta·C   (trans='T', A,B are K×N)
 *
 * Threading: one outer `omp parallel`. The two right operands (Bp_A = A in
 * B-shape, Bp_B = B in B-shape) are packed once per (js, ls) band under
 * `omp single` and shared; each thread owns a CONTIGUOUS slice of the M axis
 * (the N output rows, m_chunk = ceil(N/nth) rounded to MR) and runs the
 * MC-blocked is loop within it, packing its own Ap_A/Ap_B and doing the two
 * kernel passes. The per-band UPLO clip (m_lo_eff / m_hi_eff) trims each
 * thread's row range, but every thread still executes every js/ls iteration so
 * the `omp single`/`omp barrier` pair stays collective. Partitioning the M
 * axis into per-thread chunks (not by MC-block count) keeps threads busy on
 * small/thin shapes.
 *
 * Nesting guard: when esyr2k_ is called from inside another routine's parallel
 * region, delegate to esyr2k_serial and open no team of our own — calling only
 * the *serial* kernel primitives means no nested team either way, the libgomp
 * barrier-wedge cure.
 */

#include "esyr2k_kernel.h"
#include "../common/blas_char.h"
#include "../common/epblas_facade.h"
#include "esyrk_kernel.h"   /* esyrk_beta_{u,l} — shared triangular β pre-pass */
#include "etri_kernel.h"
#include "egemm_kernel.h"   /* egemm_choose_blocks / egemm_round_up */
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef esyr2k_TR TR;

#define MR ESYR2K_MR
#define NR ESYR2K_NR

static void esyr2k_core(
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
        esyr2k_serial(uplo, trans, n, k, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO  = blas_up(uplo);
    const char TRANS = blas_up(trans);

    if (n <= 0) return;

    /* Triangular beta pre-pass on the UPLO triangle of C only. */
    if (UPLO == 'U') esyrk_beta_u((ptrdiff_t)n, beta, c, (ptrdiff_t)ldc);
    else             esyrk_beta_l((ptrdiff_t)n, beta, c, (ptrdiff_t)ldc);

    if (k == 0 || alpha == 0.0L) return;

    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(k, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * (size_t)KC * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * (size_t)egemm_round_up(NC, NR) * sizeof(TR);

#ifdef _OPENMP
    ptrdiff_t nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif

    /* SYR2K does ~ N^2 · K flops; tiny-cutoff sized to match egemm. */
    long nnk = (long)n * (long)n * (long)k;
    if (nnk < 64L * 64L * 64L) nthreads = 1;

    /* Two shared B-packs (A and B in B-shape), two private A-packs per thread,
     * all allocated BEFORE the region: a thread that skipped the loop on a
     * failed in-region alloc would deadlock the others at the Bp barrier. */
    TR *Bp_A = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    TR *Bp_B = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    TR **Ap_A_arr = (Bp_A && Bp_B) ? calloc((size_t)nthreads, sizeof(TR *)) : NULL;
    TR **Ap_B_arr = Ap_A_arr ? calloc((size_t)nthreads, sizeof(TR *)) : NULL;
    ptrdiff_t alloc_ok = (Bp_A && Bp_B && Ap_A_arr && Ap_B_arr);
    for (ptrdiff_t t = 0; alloc_ok && t < nthreads; ++t) {
        Ap_A_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        Ap_B_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (!Ap_A_arr[t] || !Ap_B_arr[t]) alloc_ok = 0;
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

            /* M-axis (= N output rows) partition into per-thread chunks. */
            const ptrdiff_t m_chunk = egemm_round_up((n + nth - 1) / nth, MR);
            const ptrdiff_t m_lo = tid * m_chunk;
            ptrdiff_t m_hi = m_lo + m_chunk;
            if (m_hi > n) m_hi = n;

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
                        if (TRANS == 'N') {
                            etri_tcopy(pb, jb, &a[(size_t)ls * lda + js], lda, Bp_A);
                            etri_tcopy(pb, jb, &b[(size_t)ls * ldb + js], ldb, Bp_B);
                        } else {
                            etri_ncopy(pb, jb, &a[(size_t)js * lda + ls], lda, Bp_A);
                            etri_ncopy(pb, jb, &b[(size_t)js * ldb + ls], ldb, Bp_B);
                        }
                    }
                    /* implicit barrier at end of `single` → Bp safe to read */

                    for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                        const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                        if (TRANS == 'N') {
                            etri_tcopy(pb, min_i, &a[(size_t)ls * lda + is], lda, Ap_A);
                            etri_tcopy(pb, min_i, &b[(size_t)ls * ldb + is], ldb, Ap_B);
                        } else {
                            etri_ncopy(pb, min_i, &a[(size_t)is * lda + ls], lda, Ap_A);
                            etri_ncopy(pb, min_i, &b[(size_t)is * ldb + ls], ldb, Ap_B);
                        }

                        TR *cij = &c[(size_t)js * ldc + is];
                        const ptrdiff_t off = (ptrdiff_t)(is - js);

                        /* Pass 1: alpha·A·B^T + symmetric diagonal merge. */
                        if (UPLO == 'U')
                            esyr2k_kernel_u(min_i, jb, pb, alpha, Ap_A, Bp_B, cij, ldc, off, 1);
                        else
                            esyr2k_kernel_l(min_i, jb, pb, alpha, Ap_A, Bp_B, cij, ldc, off, 1);

                        /* Pass 2: alpha·B·A^T into the off-diagonal strips. */
                        if (UPLO == 'U')
                            esyr2k_kernel_u(min_i, jb, pb, alpha, Ap_B, Bp_A, cij, ldc, off, 0);
                        else
                            esyr2k_kernel_l(min_i, jb, pb, alpha, Ap_B, Bp_A, cij, ldc, off, 0);
                    }
                }
            }
        }
    }

    for (ptrdiff_t t = 0; t < nthreads && Ap_A_arr; ++t) free(Ap_A_arr[t]);
    for (ptrdiff_t t = 0; t < nthreads && Ap_B_arr; ++t) free(Ap_B_arr[t]);
    free(Ap_A_arr);
    free(Ap_B_arr);
    free(Bp_A);
    free(Bp_B);
}

EPBLAS_FACADE_SYR2K(esyr2k, TR, TR, TR)
