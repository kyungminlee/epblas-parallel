/*
 * xsyr2k_ — kind16 complex (__complex128) symmetric rank-2k update: the public
 * Fortran entry and threading-orchestration half of the xsyr2k overlay (all
 * the math lives in xsyr2k_serial.c + the shared xl3_complex.c substrate).
 * Faithful port of OpenBLAS ZSYR2K interface/syr2k.c → level3_syr2k.c threading.
 *
 *   C := alpha·(A·Bᵀ + B·Aᵀ) + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·(Aᵀ·B + Bᵀ·A) + beta·C   (trans='T', A,B are K×N)
 *
 * Threading: one outer `omp parallel`. The two right operands (Bp_A = A in
 * B-shape, Bp_B = B in B-shape) are packed once per (js, ls) band under
 * `omp single` and shared; each thread owns a CONTIGUOUS slice of the M axis
 * (the N output rows, m_chunk = ceil(N/nth) rounded to MR) and runs the
 * MC-blocked is loop within it, packing its own Ap_A/Ap_B and doing the two
 * kernel passes. The per-band UPLO clip trims each thread's row range, but
 * every thread still executes every js/ls iteration so the `single`/`barrier`
 * pair stays collective.
 *
 * Nesting guard: when xsyr2k_ is called from inside another routine's parallel
 * region, delegate to xsyr2k_serial and open no team of our own.
 *
 * Arrays are interleaved (re,im) __float128; ld-args, k, n and offset are in COMPLEX
 * elements, so every pointer step is ×2.
 */

#include "xsyr2k_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "xl3_complex.h"
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef xsyr2k_TC TC;
typedef xsyr2k_TR TR;

#define MR QBLAS_XGEMM_MR
#define NR QBLAS_XGEMM_NR


static void xsyr2k_core(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TC *alpha_c,
    const TC *a_c, ptrdiff_t lda,
    const TC *b_c, ptrdiff_t ldb,
    const TC *beta_c,
    TC *c_c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Inside another team → run serial, open no region of our own. */
    if (omp_in_parallel()) {
        xsyr2k_serial(uplo, trans, n, k, alpha_c, a_c, lda, b_c, ldb,
                      beta_c, c_c, ldc);
        return;
    }
#endif
    /* Reinterpret the complex ABI as interleaved (re,im) __float128 storage. */
    const TR *alpha_ = (const TR *)alpha_c;
    const TR *a = (const TR *)a_c;
    const TR *b = (const TR *)b_c;
    const TR *beta_ = (const TR *)beta_c;
    TR *c = (TR *)c_c;
    const TR alphar = alpha_[0], alphai = alpha_[1];
    const TR beta_r = beta_[0],  beta_i = beta_[1];
    const char UPLO  = blas_up(uplo);
    const char TRANS = blas_up(trans);

    if (n <= 0) return;

    if (UPLO == 'U') qblas_xsyrk_beta_u(n, beta_r, beta_i, c, ldc);
    else             qblas_xsyrk_beta_l(n, beta_r, beta_i, c, ldc);

    if (k == 0 || (alphar == 0.0Q && alphai == 0.0Q)) return;

    ptrdiff_t MC0, KC0, NC0;
    qblas_xgemm_blocks(&MC0, &KC0, &NC0);
    ptrdiff_t MC = MC0, KC = KC0, NC = NC0;

    if (k <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;
        long target_mc = L2_TARGET_BYTES / ((long)k * 2L * (long)sizeof(TR));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = blas_round_up((ptrdiff_t)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    const size_t ap_bytes = (size_t)blas_round_up(MC, MR) * (size_t)KC * 2 * sizeof(TR);
    const size_t bp_bytes = (size_t)KC * (size_t)blas_round_up(NC, NR) * 2 * sizeof(TR);

#ifdef _OPENMP
    ptrdiff_t nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif

    long nnk = (long)n * (long)n * (long)k;
    if (nnk < 64L * 64L * 64L) nthreads = 1;

    /* Two shared B-packs, two private A-packs per thread, all allocated BEFORE
     * the region: a thread that skipped the loop on a failed in-region alloc
     * would deadlock the others at the Bp barrier. */
    TR *Bp_A = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    TR *Bp_B = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    TR **Ap_A_arr = (Bp_A && Bp_B) ? calloc((size_t)nthreads, sizeof(TR *)) : NULL;
    TR **Ap_B_arr = Ap_A_arr ? calloc((size_t)nthreads, sizeof(TR *)) : NULL;
    bool alloc_ok = (Bp_A && Bp_B && Ap_A_arr && Ap_B_arr);
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
            const ptrdiff_t m_chunk = blas_round_up((n + nth - 1) / nth, MR);
            const ptrdiff_t m_lo = tid * m_chunk;
            ptrdiff_t m_hi = m_lo + m_chunk;
            if (m_hi > n) m_hi = n;

            for (ptrdiff_t js = 0; js < n; js += NC) {
                const ptrdiff_t jb = (n - js < NC) ? (n - js) : NC;

                ptrdiff_t m_lo_eff = (UPLO == 'L' && m_lo < js) ? js : m_lo;
                ptrdiff_t m_hi_eff = (UPLO == 'U' && m_hi > js + jb) ? (js + jb) : m_hi;
                if (m_lo_eff & (MR - 1)) m_lo_eff &= ~(MR - 1);
                if (m_lo_eff < m_lo) m_lo_eff = m_lo;

                for (ptrdiff_t ls = 0; ls < k; ls += KC) {
                    const ptrdiff_t pb = (k - ls < KC) ? (k - ls) : KC;

#ifdef _OPENMP
                    #pragma omp barrier
                    #pragma omp single
#endif
                    {
                        if (TRANS == 'N') {
                            qblas_xgemm_tcopy(pb, jb, 0, &a[((size_t)ls * lda + js) * 2], lda, Bp_A);
                            qblas_xgemm_tcopy(pb, jb, 0, &b[((size_t)ls * ldb + js) * 2], ldb, Bp_B);
                        } else {
                            qblas_xgemm_ncopy(pb, jb, 0, &a[((size_t)js * lda + ls) * 2], lda, Bp_A);
                            qblas_xgemm_ncopy(pb, jb, 0, &b[((size_t)js * ldb + ls) * 2], ldb, Bp_B);
                        }
                    }
                    /* implicit barrier at end of `single` → Bp safe to read */

                    for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                        const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                        if (TRANS == 'N') {
                            qblas_xgemm_tcopy(pb, min_i, 0, &a[((size_t)ls * lda + is) * 2], lda, Ap_A);
                            qblas_xgemm_tcopy(pb, min_i, 0, &b[((size_t)ls * ldb + is) * 2], ldb, Ap_B);
                        } else {
                            qblas_xgemm_ncopy(pb, min_i, 0, &a[((size_t)is * lda + ls) * 2], lda, Ap_A);
                            qblas_xgemm_ncopy(pb, min_i, 0, &b[((size_t)is * ldb + ls) * 2], ldb, Ap_B);
                        }

                        TR *cij = &c[((size_t)js * ldc + is) * 2];
                        const ptrdiff_t off = is - js;

                        /* Pass 1: alpha·A·Bᵀ + symmetric diagonal merge. */
                        if (UPLO == 'U')
                            qblas_xsyr2k_kernel_u(min_i, jb, pb, alphar, alphai, Ap_A, Bp_B, cij, ldc, off, 1);
                        else
                            qblas_xsyr2k_kernel_l(min_i, jb, pb, alphar, alphai, Ap_A, Bp_B, cij, ldc, off, 1);

                        /* Pass 2: alpha·B·Aᵀ into the off-diagonal strips. */
                        if (UPLO == 'U')
                            qblas_xsyr2k_kernel_u(min_i, jb, pb, alphar, alphai, Ap_B, Bp_A, cij, ldc, off, 0);
                        else
                            qblas_xsyr2k_kernel_l(min_i, jb, pb, alphar, alphai, Ap_B, Bp_A, cij, ldc, off, 0);
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

/* Emit xsyr2k_ (LP64) + xsyr2k_64_ (ILP64) around the shared ptrdiff_t core.
 * alpha/beta/matrices are all complex (TC=__complex128) — SYR2K is symmetric,
 * so both scalars are complex; mirrors kind10 ysyr2k's
 * EPBLAS_FACADE_SYR2K(ysyr2k, TC, TC, TC). */
EPBLAS_FACADE_SYR2K(xsyr2k, TC, TC, TC)
