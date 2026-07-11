/*
 * xgemm — kind16 (COMPLEX(KIND=16)) port of OpenBLAS ZGEMM.
 *
 *   C := alpha * op(A) * op(B) + beta * C
 *
 * Port source: OpenBLAS.
 *   - interface/gemm.c          (argument-parse, transa/transb codes,
 *                                early-return, info / xerbla)
 *   - driver/level3/level3.c    (NC × KC × MC three-level blocking nest;
 *                                ICOPY(A) / OCOPY(B) / KERNEL sequence)
 *   - kernel/generic/
 *       zgemmkernel_2x2.c      → qblas_ygemm_kernel  (NN path only;
 *                                                     conjugation absorbed
 *                                                     into the packers)
 *       zgemm_ncopy_2.c         → qblas_ygemm_ncopy
 *       zgemm_tcopy_2.c         → qblas_ygemm_tcopy
 *       zgemm_beta.c            → qblas_ygemm_beta
 *
 * Differences from upstream ZGEMM:
 *   - No SIMD. x86_64 has no AVX path for 128-bit __float128 complex;
 *     the reference scalar `zgemmkernel_2x2.c` is the kernel template.
 *   - No `blas_queue` SMP runtime. Single-level OpenMP parallelism
 *     over the M-axis inside the (jc, pc) blocking — each thread
 *     keeps a private Ap buffer; Bp is shared and packed once per
 *     (jc, pc) under `omp single`.
 *   - Conjugation absorbed at pack-time (sign flip on the imag floats)
 *     so the kernel only implements the NN form. This is bit-exact
 *     (a sign flip is an exact IEEE-754 op) and keeps the K-loop
 *     branch-free across the 16 transa/transb code combinations.
 *   - Adaptive MC when K is small (mirrors OpenBLAS level3.c gemm_p
 *     grown so MC*KC of complex `__float128` fits the L2 target).
 *
 * Fortran ABI:
 *   subroutine xgemm(transa, transb, m, n, k, alpha, a, lda, b, ldb,
 *                    beta, c, ldc)
 *   - character args with trailing hidden size_t lengths (gfortran)
 *   - alpha, beta are COMPLEX(KIND=16): 2 __float128s each (re, im)
 *   - a, b, c are COMPLEX(KIND=16) arrays (interleaved re,im)
 *   - lda, ldb, ldc are in COMPLEX(KIND=16) elements
 */

#include "qblas_l3_complex.h"
#include <quadmath.h>
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __float128 T;

#define MR QBLAS_YGEMM_MR
#define NR QBLAS_YGEMM_NR


/* Map a Fortran trans character to a normalized code. Supports
 * 'N', 'T', 'C' (BLAS standard) and 'R' (extension for conj-no-trans). */
static int trans_code(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static int op_is_conj(int c)  { return (c == 'C' || c == 'R') ? 1 : 0; }
static int op_is_trans(int c) { return (c == 'T' || c == 'C') ? 1 : 0; }

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


/* Single-thread level3 driver for one (m_lo..m_hi) × (0..N) slab.
 *
 * Mirrors driver/level3/level3.c's `for(js..) for(ls..) for(is..)`
 * blocking nest. The caller has already packed Bp_shared via the
 * appropriate OCOPY for (js, ls, pb, jb); this function does the
 * ICOPY(A) for each M-block and runs the kernel against Bp_shared.
 */
static void level3_kernel(int m_lo, int m_hi,
                          int MC,
                          int ta,
                          T alphar, T alphai,
                          const T *A, int lda,
                          T *Ap, const T *Bp_shared,
                          int js, int ls, int pb, int jb,
                          T *C, int ldc)
{
    const int conj_a  = op_is_conj(ta);
    const int trans_a = op_is_trans(ta);

    for (int is = m_lo; is < m_hi; is += MC) {
        int min_i = (m_hi - is < MC) ? (m_hi - is) : MC;

        /* ICOPY(A). Source offsets follow OpenBLAS's ICOPY_OPERATION:
         *   normal-A: TCOPY at A + (is + ls*lda) * COMPSIZE
         *   trans-A:  NCOPY at A + (ls + is*lda) * COMPSIZE  */
        if (!trans_a) {
            qblas_ygemm_tcopy(pb, min_i, conj_a,
                              &A[((size_t)ls * lda + is) * 2], lda, Ap);
        } else {
            qblas_ygemm_ncopy(pb, min_i, conj_a,
                              &A[((size_t)is * lda + ls) * 2], lda, Ap);
        }

        qblas_ygemm_kernel(min_i, jb, pb, alphar, alphai,
                           Ap, Bp_shared,
                           &C[((size_t)js * ldc + is) * 2], ldc);
    }
}


/* ── Public entry ──────────────────────────────────────────────── */
void xgemm_(
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
    const T alphar = alpha_[0], alphai = alpha_[1];
    const T beta_r = beta_[0],  beta_i = beta_[1];
    const int ta = trans_code(transa_p);
    const int tb = trans_code(transb_p);

    if (M <= 0 || N <= 0) return;

    /* Beta pre-pass — written exactly once before the K-walk starts. */
    qblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, beta_r, beta_i,
                     c, (ptrdiff_t)ldc);

    if (K == 0 || (alphar == 0.0Q && alphai == 0.0Q)) return;

    /* Block sizes (env-overridable). */
    int MC0, KC, NC;
    qblas_ygemm_blocks(&MC0, &KC, &NC);

    /* Adaptive MC for small K, sized to keep Ap inside L2.
     * Complex `__float128` is 2 * sizeof(__float128) = 32 B/element. */
    int MC = MC0;
    if (K <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;
        long target_mc = L2_TARGET_BYTES / ((long)K * (long)(2 * sizeof(T)));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((int)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    /* Pack-buffer sizes: 2 __float128s per complex element. */
    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * 2 * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * 2 * sizeof(T);

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
     * region; abort cleanly on alloc failure so every thread in the
     * team hits every `omp barrier` / `omp single`. */
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (!Bp) return;

    T **Ap_arr = calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_arr) { free(Bp); return; }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
        free(Ap_arr);
        free(Bp);
        return;
    }

    const int conj_b  = op_is_conj(tb);
    const int trans_b = op_is_trans(tb);

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

        /* Block-partition the M axis across threads, rounded to MR. */
        int m_chunk = round_up((M + nth - 1) / nth, MR);
        int m_lo = tid * m_chunk;
        int m_hi = m_lo + m_chunk;
        if (m_hi > M) m_hi = M;

        for (int js = 0; js < N; js += NC) {
            int jb = (N - js < NC) ? (N - js) : NC;
            for (int ls = 0; ls < K; ls += KC) {
                int pb = (K - ls < KC) ? (K - ls) : KC;

                /* OCOPY(B) — pack the (pb × jb) panel once per (js, ls).
                 * The explicit BEFORE barrier ensures no thread is still
                 * reading the previous Bp from its kernel run; the
                 * implicit `omp single` END barrier ensures the new Bp
                 * is visible before any thread starts the kernel. */
#ifdef _OPENMP
                #pragma omp barrier
                #pragma omp single
#endif
                {
                    /* Source offsets follow OpenBLAS's OCOPY_OPERATION:
                     *   normal-B: NCOPY at b + (ls + js*ldb) * COMPSIZE
                     *   trans-B:  TCOPY at b + (js + ls*ldb) * COMPSIZE */
                    if (!trans_b) {
                        qblas_ygemm_ncopy(pb, jb, conj_b,
                            &b[((size_t)js * ldb + ls) * 2], ldb, Bp);
                    } else {
                        qblas_ygemm_tcopy(pb, jb, conj_b,
                            &b[((size_t)ls * ldb + js) * 2], ldb, Bp);
                    }
                }

                level3_kernel(m_lo, m_hi, MC,
                              ta, alphar, alphai,
                              a, lda, Ap, Bp,
                              js, ls, pb, jb,
                              c, ldc);
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}
