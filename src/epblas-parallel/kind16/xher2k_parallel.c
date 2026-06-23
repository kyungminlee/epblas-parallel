/*
 * xher2k_ — kind16 complex (__complex128) Hermitian rank-2k update: the public
 * Fortran entry and threading-orchestration half of the xher2k overlay (all
 * the math lives in xher2k_serial.c + the shared xl3_complex.c substrate).
 * Faithful port of OpenBLAS ZHER2K interface/syr2k.c → level3_syr2k.c threading.
 *
 *   C := alpha·A·Bᴴ + conj(alpha)·B·Aᴴ + beta·C   (trans='N', A,B are N×K)
 *   C := alpha·Aᴴ·B + conj(alpha)·Bᴴ·A + beta·C   (trans='C', A,B are K×N)
 *
 * alpha is COMPLEX, beta is REAL, C is HERMITIAN: only the UPLO triangle is
 * touched and the diagonal stays real on output.
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
 * Conjugation is absorbed at pack time: TRANS='N' conjugates the Bp side,
 * TRANS='C' conjugates the Ap side (upstream GEMM_KERNEL_R/_L). Pass 2 uses
 * conj(alpha) via −alphai.
 *
 * Nesting guard: when xher2k_ is called from inside another routine's parallel
 * region, delegate to xher2k_serial and open no team of our own.
 *
 * Public ABI is complex (__complex128 A/B/C/alpha) + real (__float128 beta);
 * the core reinterprets to interleaved (re,im) __float128 storage, ld-args, k, n
 * and offset in COMPLEX elements, so every internal pointer step is ×2.
 */

#include "xher2k_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "xl3_complex.h"
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef xher2k_TC TC;
typedef xher2k_TR TR;

#define MR QBLAS_XGEMM_MR
#define NR QBLAS_XGEMM_NR


static void xher2k_core(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TC *alpha_c,
    const TC *a_c, ptrdiff_t lda,
    const TC *b_c, ptrdiff_t ldb,
    const TR *beta_,
    TC *c_c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Inside another team → run serial, open no region of our own. */
    if (omp_in_parallel()) {
        xher2k_serial(uplo, trans, n, k, alpha_c, a_c, lda, b_c, ldb,
                      beta_, c_c, ldc);
        return;
    }
#endif
    /* Reinterpret the complex ABI as interleaved (re,im) __float128 storage. */
    const TR *alpha_ = (const TR *)alpha_c;
    const TR *a = (const TR *)a_c;
    const TR *b = (const TR *)b_c;
    TR *c = (TR *)c_c;
    const TR alphar = alpha_[0], alphai = alpha_[1];
    const TR beta_r = beta_[0];
    const char UPLO  = blas_up(uplo);
    const char TRANS = blas_up(trans);

    if (n <= 0) return;

    if (UPLO == 'U') qblas_xherk_beta_u(n, beta_r, c, ldc);
    else             qblas_xherk_beta_l(n, beta_r, c, ldc);

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

    /* Conjugation absorbed at pack time (upstream GEMM_KERNEL_R/_L choice):
     *   TRANS='N' → conjugate Bp;  TRANS='C' → conjugate Ap. */
    const bool conj_a_pack = (TRANS == 'C') ? 1 : 0;
    const bool conj_b_pack = (TRANS == 'N') ? 1 : 0;

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
                            qblas_xgemm_tcopy(pb, jb, conj_b_pack, &a[((size_t)ls * lda + js) * 2], lda, Bp_A);
                            qblas_xgemm_tcopy(pb, jb, conj_b_pack, &b[((size_t)ls * ldb + js) * 2], ldb, Bp_B);
                        } else {
                            qblas_xgemm_ncopy(pb, jb, conj_b_pack, &a[((size_t)js * lda + ls) * 2], lda, Bp_A);
                            qblas_xgemm_ncopy(pb, jb, conj_b_pack, &b[((size_t)js * ldb + ls) * 2], ldb, Bp_B);
                        }
                    }
                    /* implicit barrier at end of `single` → Bp safe to read */

                    for (ptrdiff_t is = m_lo_eff; is < m_hi_eff; is += MC) {
                        const ptrdiff_t min_i = (m_hi_eff - is < MC) ? (m_hi_eff - is) : MC;

                        if (TRANS == 'N') {
                            qblas_xgemm_tcopy(pb, min_i, conj_a_pack, &a[((size_t)ls * lda + is) * 2], lda, Ap_A);
                            qblas_xgemm_tcopy(pb, min_i, conj_a_pack, &b[((size_t)ls * ldb + is) * 2], ldb, Ap_B);
                        } else {
                            qblas_xgemm_ncopy(pb, min_i, conj_a_pack, &a[((size_t)is * lda + ls) * 2], lda, Ap_A);
                            qblas_xgemm_ncopy(pb, min_i, conj_a_pack, &b[((size_t)is * ldb + ls) * 2], ldb, Ap_B);
                        }

                        TR *cij = &c[((size_t)js * ldc + is) * 2];
                        const ptrdiff_t off = is - js;

                        /* Pass 1: alpha·A·Bᴴ + Hermitian diagonal merge. */
                        if (UPLO == 'U')
                            qblas_xher2k_kernel_u(min_i, jb, pb, alphar, alphai, Ap_A, Bp_B, cij, ldc, off, 1);
                        else
                            qblas_xher2k_kernel_l(min_i, jb, pb, alphar, alphai, Ap_A, Bp_B, cij, ldc, off, 1);

                        /* Pass 2: conj(alpha)·B·Aᴴ into the off-diagonal strips. */
                        if (UPLO == 'U')
                            qblas_xher2k_kernel_u(min_i, jb, pb, alphar, -alphai, Ap_B, Bp_A, cij, ldc, off, 0);
                        else
                            qblas_xher2k_kernel_l(min_i, jb, pb, alphar, -alphai, Ap_B, Bp_A, cij, ldc, off, 0);
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

/* Emit xher2k_ (LP64) + xher2k_64_ (ILP64) around the shared ptrdiff_t core.
 * alpha=complex (TC=__complex128), beta=real (TR=__float128), matrices=complex
 * (TC) — mirrors kind10 yher2k's EPBLAS_FACADE_SYR2K(yher2k, TC, TR, TC). */
EPBLAS_FACADE_SYR2K(xher2k, TC, TR, TC)
