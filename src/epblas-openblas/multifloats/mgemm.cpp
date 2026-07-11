/*
 * mgemm — kind10 (REAL(KIND=10) / 80-bit multifloats::float64x2) port of OpenBLAS DGEMM.
 *
 *   C := alpha * op(A) * op(B) + beta * C
 *
 * Port source: OpenBLAS.
 *   - interface/gemm.c          (argument-parse, transa/transb codes,
 *                                early-return, info / xerbla)
 *   - driver/level3/level3.c    (NC × KC × MC three-level blocking nest;
 *                                ICOPY(A) / OCOPY(B) / KERNEL sequence)
 *   - kernel/generic/
 *       gemmkernel_2x2.c       → mblas_egemm_kernel  (shared via common/)
 *       gemm_ncopy_2.c          → mblas_egemm_ncopy   (shared via common/)
 *       gemm_tcopy_2.c          → mblas_egemm_tcopy   (shared via common/)
 *       gemm_beta.c             → mblas_egemm_beta    (shared via common/)
 *
 * Differences from upstream DGEMM:
 *   - No SIMD. x86_64 has no AVX path for 80-bit multifloats::float64x2; the
 *     reference scalar `gemmkernel_2x2.c` is the kernel.
 *   - No `blas_queue` SMP runtime. Single-level OpenMP parallelism
 *     over the M-axis inside the (jc, pc) blocking — each thread
 *     keeps a private Ap buffer; Bp is shared and packed once per
 *     (jc, pc) under `omp single` (implicit barrier).
 *   - Adaptive MC when K is small (mirrors OpenBLAS level3.c lines
 *     313–321: gemm_p grown so MC*KC fits the L2 target).
 *
 * Fortran ABI:
 *   subroutine mgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb,
 *                    beta, c, ldc)
 *   - character args carry trailing hidden size_t lengths (gfortran)
 *   - all scalars by pointer; REAL(KIND=10) ↔ multifloats::float64x2
 */

#include "mblas_l3_real.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#define MR MBLAS_EGEMM_MR
#define NR MBLAS_EGEMM_NR


/* Map a Fortran trans character to a normalized code. For real input,
 * 'C' is equivalent to 'T' (no complex-conjugate operation possible). */
static int trans_code(const char *p) {
    char c = (char)toupper((unsigned char)*p);
    return (c == 'C') ? 'T' : c;
}

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


/* ── Single-thread level3 driver for one (m_lo..m_hi) × (0..N) slab ──
 *
 * Mirrors driver/level3/level3.c's `for(js..) for(ls..) for(is..)`
 * blocking nest. Each call is the work of one thread; the caller
 * passes (m_lo, m_hi) as the thread's M slice, and Ap is a private
 * per-thread scratch buffer. Bp is shared with the rest of the
 * threads (packed once per (js, ls)).
 *
 * Returns 0 on success, -1 on allocation failure (only happens for
 * the caller-supplied Ap if it was NULL).
 */
static void level3_kernel(int m_lo, int m_hi, int N, int K,
                          int MC, int KC, int NC,
                          int ta, int tb,
                          T alpha,
                          const T *A, int lda,
                          const T *B, int ldb,
                          T *C, int ldc,
                          T *Ap, const T *Bp_shared, int js, int ls, int pb, int jb)
{
    /* Process the (m_lo..m_hi) × (js..js+jb) C-band against
     * the K-panel B[ls..ls+pb, js..js+jb] that has already been
     * packed into Bp_shared by the caller. */
    for (int is = m_lo; is < m_hi; is += MC) {
        int min_i = (m_hi - is < MC) ? (m_hi - is) : MC;

        /* ICOPY(A) — pack into Ap. OpenBLAS naming convention:
         *   - ta == 'N' (normal A): use TCOPY. Source slab is A[is..,
         *     ls..] in col-major; tcopy(m=K, n=M) walks 2 source cols
         *     per K-strip, producing 2-M-row panels (one per j).
         *   - ta == 'T' (op(A)=A^T): use NCOPY. Source slab is A[ls..,
         *     is..] viewed col-major; ncopy(m=K, n=M) walks 2 M-cols
         *     per panel, with K elements interleaved within. */
        if (ta == 'N') {
            mblas_egemm_tcopy(pb, min_i,
                              &A[(size_t)ls * lda + is], lda, Ap);
        } else {
            mblas_egemm_ncopy(pb, min_i,
                              &A[(size_t)is * lda + ls], lda, Ap);
        }

        mblas_egemm_kernel(min_i, jb, pb, alpha,
                           Ap, Bp_shared,
                           &C[(size_t)js * ldc + is], ldc);
    }
}


/* ── Public entry ──────────────────────────────────────────────── */
extern "C" void mgemm_(
    const char *transa_p, const char *transb_p,
    const int *m_, const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_)
{
    const int M = *m_, N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const int ta = trans_code(transa_p);
    const int tb = trans_code(transb_p);

    /* OpenBLAS-style early-return: m,n=0 → no-op; k=0 still needs the
     * beta pass on C (handled below). */
    if (M <= 0 || N <= 0) return;

    /* Beta pre-pass — written exactly once before the K-walk starts. */
    mblas_egemm_beta((ptrdiff_t)M, (ptrdiff_t)N, beta, c, (ptrdiff_t)ldc);

    if (K == 0 || alpha == 0.0) return;

    /* Block sizes (env-overridable). */
    int MC0, KC, NC;
    mblas_egemm_blocks(&MC0, &KC, &NC);

    /* OpenBLAS adaptive MC: when K fits in one panel, grow MC so
     * MC*KC roughly fits the L2 target. */
    int MC = MC0;
    if (K <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;  /* P-core L2 nominal */
        long target_mc = L2_TARGET_BYTES / ((long)K * (long)sizeof(T));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((int)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    /* Pad allocations to MR / NR alignment so the kernel's edge
     * branches see a deterministic shape. */
    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * sizeof(T);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
#endif

    /* Don't fan out for tiny problems — overhead exceeds work. */
    long mnk = (long)M * (long)N * (long)K;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

    /* Allocate per-thread Ap and the shared Bp BEFORE the parallel
     * region. If any allocation fails we abort the whole call —
     * we can't have some threads skipping the loop body while others
     * proceed, since `omp single` / `omp barrier` require every
     * thread in the team to encounter them. */
    T *Bp = (T *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) return;

    T **Ap_arr = (T **)calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_arr) { free(Bp); return; }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = (T *)aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
        free(Ap_arr);
        free(Bp);
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
#else
        int tid = 0, nth = 1;
#endif
        T *Ap = Ap_arr[tid];

        /* Block-partition the M axis across threads (rounded to MR
         * so each thread's slab lines up with the kernel grid). */
        int m_chunk = round_up((M + nth - 1) / nth, MR);
        int m_lo = tid * m_chunk;
        int m_hi = m_lo + m_chunk;
        if (m_hi > M) m_hi = M;

        /* All threads enter this loop nest regardless of whether
         * their (m_lo, m_hi) range is empty — every thread MUST
         * encounter every `omp barrier` and `omp single`. Threads
         * with m_lo >= m_hi just do no work in level3_kernel (the
         * inner `for (is = m_lo; is < m_hi; ...)` is a no-op). */
        {
            /* Outer js / ls loops — mirror level3.c lines 305–311. */
                for (int js = 0; js < N; js += NC) {
                    int jb = (N - js < NC) ? (N - js) : NC;
                    for (int ls = 0; ls < K; ls += KC) {
                        int pb = (K - ls < KC) ? (K - ls) : KC;

                        /* OCOPY(B) — pack the full (pb × jb) panel
                         * once per (js, ls). One thread does it; the
                         * implicit `omp single` END barrier ensures
                         * the new Bp is visible before any thread
                         * starts level3_kernel. The explicit BEFORE
                         * barrier ensures no thread is still reading
                         * the previous Bp from its level3_kernel run.
                         * (For ls == 0 && js == 0 there is no previous
                         * Bp, but the barrier is still cheap and the
                         * code is cleaner without a special case.) */
#ifdef _OPENMP
                        #pragma omp barrier
                        #pragma omp single
#endif
                        {
                            /* OCOPY(B). Mirror of OpenBLAS's symmetry:
                             *   - tb == 'N' (normal B): NCOPY walks 2
                             *     source cols per panel — each panel
                             *     is 2 B cols × pb K-rows.
                             *   - tb == 'T' (op(B)=B^T): TCOPY walks
                             *     2 source cols (=2 K-cols of B^T) per
                             *     strip, producing 2-B^T-col panels. */
                            if (tb == 'N') {
                                mblas_egemm_ncopy(pb, jb,
                                    &b[(size_t)js * ldb + ls], ldb, Bp);
                            } else {
                                mblas_egemm_tcopy(pb, jb,
                                    &b[(size_t)ls * ldb + js], ldb, Bp);
                            }
                        }

                        /* Inner kernel — each thread handles its m
                         * slice against the shared Bp panel. */
                        level3_kernel(m_lo, m_hi, N, K, MC, KC, NC,
                                      ta, tb, alpha,
                                      a, lda, b, ldb,
                                      c, ldc,
                                      Ap, Bp, js, ls, pb, jb);

                        /* Next iteration's leading `#pragma omp barrier`
                         * (or the `omp parallel` end-barrier on the
                         * final iter) keeps Bp stable. */
                    }
                }
            }
    }

    for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}
