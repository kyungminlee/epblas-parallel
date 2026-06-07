/*
 * mtrmm — kind10 (REAL(KIND=10) / 80-bit multifloats::float64x2) port of OpenBLAS DTRMM.
 *
 *   B := alpha * op(A) * B    (SIDE='L')
 *   B := alpha * B * op(A)    (SIDE='R')
 *
 * Faithful L3-blocked port: full OpenBLAS structure with NC × KC × MC
 * three-level blocking, pack-and-conquer, diagonal-aware TRMM kernel.
 *
 * Port source: OpenBLAS.
 *   - interface/trsm.c                (TRMM macro path — args parsing,
 *                                      alpha pre-scale of B = beta-pass,
 *                                      side-of-thread dispatch)
 *   - driver/level3/trmm_L.c          (SIDE='L' driver, 4 (UPLO,TRANS)
 *                                      branches with packer/kernel selection)
 *   - driver/level3/trmm_R.c          (SIDE='R' driver, same)
 *   - kernel/generic/trmmkernel_2x2.c → mblas_etrmm_kernel (shared)
 *   - kernel/generic/trmm_{ut,un,lt,ln}copy_2.c
 *                                     → mblas_etrmm_i{ut,un,lt,ln}copy
 *
 * Algorithm reconstruction (from interface/trsm.c lines 64+, trmm_L.c
 * lines 92, 103-113): TRMM is implemented as
 *   1. alpha pre-scale B in-place (mblas_egemm_beta with beta=alpha)
 *   2. L3 nest runs with kernel-alpha = 1.0, overwriting B tile by tile.
 * After step 2, B has been transformed by op(A); since step 1 scaled B by
 * alpha, the final result is alpha * op(A) * B_old (linear in B), matching
 * the TRMM spec.
 *
 * OpenBLAS dispatches via gemm_thread_n (SIDE='L', partitions N-axis)
 * and gemm_thread_m (SIDE='R', partitions M-axis). We mirror this:
 *   - SIDE='L': each OMP thread takes a contiguous N-slice, runs the
 *     full trmm_L.c nest over that slice (its own js loop range, its
 *     own per-thread Ap, shared Bp via omp single).
 *   - SIDE='R': each thread takes an M-slice; runs full trmm_R.c nest.
 *
 * The TRMM kernel does C := alpha*A*B (overwrite), not C += ...; for
 * off-diagonal sub-tiles OpenBLAS calls GEMM_KERNEL with beta=0 (also
 * overwrite). Our shared mblas_egemm_kernel does +=, so we use
 * mblas_egemm_kernel_store (which zeros then calls the shared kernel).
 *
 * Fortran ABI:
 *   subroutine mtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
 *   - character args carry trailing hidden size_t lengths (gfortran)
 *   - all scalars by pointer; REAL(KIND=10) ↔ multifloats::float64x2
 *   - 'C' on transa is treated identically to 'T' (no conjugation for reals)
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

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


/* ── A-side packer dispatch for the TRMM driver ──────────────────────
 *
 * The same packer source files (TRMM_{UT,UN,LT,LN}COPY) serve both
 * SIDE=L (TRMM_I*COPY role) and SIDE=R (TRMM_O*COPY role) — same
 * function, different call-site context. The (uplo, trans) → packer
 * mapping however differs by SIDE because the "transpose direction"
 * of A in the kernel is reversed for SIDE=R vs SIDE=L:
 *
 *   SIDE=L:  (UPPER && !TRANS): UTCOPY    (trmm_L.c line 132)
 *            (!UPPER && TRANS):  LNCOPY    (line 134)
 *            (!UPPER && !TRANS): LTCOPY    (line 314)
 *            (UPPER && TRANS):   UNCOPY    (line 316)
 *
 *   SIDE=R:  (!UPPER && !TRANS): LNCOPY    (trmm_R.c line 158)
 *            (UPPER && TRANS):   UTCOPY    (line 160)
 *            (UPPER && !TRANS):  UNCOPY    (line 270)
 *            (!UPPER && TRANS):  LTCOPY    (line 272)
 *
 * Effectively: SIDE=R swaps the {UT↔UN} and {LT↔LN} packers relative
 * to SIDE=L for the same (uplo, trans). */
static inline void pack_trmm_a(int side_l, int uplo_upper, int trans, int unit,
                               ptrdiff_t m, ptrdiff_t n,
                               const T *a, ptrdiff_t lda,
                               ptrdiff_t posX, ptrdiff_t posY,
                               T *bp)
{
    if (side_l) {
        if (uplo_upper && !trans)       mblas_etrmm_iutcopy(m, n, a, lda, posX, posY, bp, unit);
        else if (uplo_upper &&  trans)  mblas_etrmm_iuncopy(m, n, a, lda, posX, posY, bp, unit);
        else if (!uplo_upper && !trans) mblas_etrmm_iltcopy(m, n, a, lda, posX, posY, bp, unit);
        else                            mblas_etrmm_ilncopy(m, n, a, lda, posX, posY, bp, unit);
    } else {
        if (uplo_upper && !trans)       mblas_etrmm_iuncopy(m, n, a, lda, posX, posY, bp, unit);
        else if (uplo_upper &&  trans)  mblas_etrmm_iutcopy(m, n, a, lda, posX, posY, bp, unit);
        else if (!uplo_upper && !trans) mblas_etrmm_ilncopy(m, n, a, lda, posX, posY, bp, unit);
        else                            mblas_etrmm_iltcopy(m, n, a, lda, posX, posY, bp, unit);
    }
}


/* ── SIDE='L' driver: port of trmm_L.c for one N-band (js0..js1) ─────
 *
 * Each thread calls this with its own N-slice and own per-thread Ap;
 * Bp is shared but each thread re-OCOPYs into private Bp slots since we
 * have one Bp per thread (no cross-thread sync needed). The OpenBLAS
 * source uses a single shared sa/sb per call site; for our per-thread-
 * over-N-slice partitioning, each thread has its own (Ap, Bp).
 */
static void trmm_L_band(int upper, int trans, int unit,
                        int M, int js0, int js1, int K_alias,
                        int MC, int KC, int NC,
                        const T *a, int lda,
                        T *b, int ldb,
                        T *Ap, T *Bp)
{
    (void)K_alias;  /* For SIDE='L', K = M */
    const T dp1 = 1.0;
    int m = M;

    /* Outer js-loop walks the thread's N-band in steps of NC = GEMM_R. */
    for (int js = js0; js < js1; js += NC) {
        int min_j = js1 - js;
        if (min_j > NC) min_j = NC;

        if ((upper && !trans) || (!upper && trans)) {
            /* trmm_L.c lines 119-299: TRMM_KERNEL_N = TRMM_KERNEL_LN
             * (kernel trans-flag = 0; the user trans is absorbed by the
             * packer choice).
             * Walk down-diagonal: pack A[ls..ls+min_l, ls..ls+min_l] as
             * the triangular block (TRMM_IUTCOPY for UPPER!TRANS,
             * TRMM_ILNCOPY for LOWER+TRANS), then off-diagonal GEMM tiles
             * above the diagonal block. */
            const int kt = 0;   /* TRMM_KERNEL_N */
            int min_l = m;
            if (min_l > KC) min_l = KC;
            int min_i = min_l;
            if (min_i > MC) min_i = MC;
            if (min_i > MR) min_i = (min_i / MR) * MR;

            /* TRMM_I*COPY(min_l, min_i, a, lda, posX=0, posY=0, sa) */
            pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda, 0, 0, Ap);

            /* Inner jjs loop — process min_j cols in NR-sized strips.
             * For our NR=2, the OpenBLAS jjs stride logic resolves to
             * min_jj = NR (or the trailing remnant). */
            for (int jjs = js; jjs < js + min_j; jjs += NR) {
                int min_jj = js + min_j - jjs;
                if (min_jj > NR) min_jj = NR;

                /* GEMM_ONCOPY(min_l, min_jj, b + jjs*ldb, ldb, sb + ...) */
                mblas_egemm_ncopy(min_l, min_jj,
                                  &b[(size_t)jjs * ldb], ldb,
                                  Bp + (size_t)min_l * (jjs - js));

                /* TRMM_KERNEL_N(min_i, min_jj, min_l, dp1, sa, sb_slice,
                 *               b + jjs*ldb, ldb, 0) */
                mblas_etrmm_kernel(/*left=*/1, /*trans=*/kt,
                                   min_i, min_jj, min_l, dp1,
                                   Ap, Bp + (size_t)min_l * (jjs - js),
                                   &b[(size_t)jjs * ldb], ldb,
                                   /*offset=*/0);
            }

            /* Lower-band: more MR-tiles of A below the diagonal block. */
            for (int is = min_i; is < min_l; is += min_i) {
                min_i = min_l - is;
                if (min_i > MC) min_i = MC;
                if (min_i > MR) min_i = (min_i / MR) * MR;

                pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda,
                            /*posX=*/0, /*posY=*/is, Ap);

                mblas_etrmm_kernel(/*left=*/1, /*trans=*/kt,
                                   min_i, min_j, min_l, dp1,
                                   Ap, Bp,
                                   &b[(size_t)is + (size_t)js * ldb], ldb,
                                   /*offset=*/is);
            }

            /* ls-loop continues for the remaining KC-bands of A. */
            for (int ls = min_l; ls < m; ls += KC) {
                min_l = m - ls;
                if (min_l > KC) min_l = KC;
                min_i = ls;
                if (min_i > MC) min_i = MC;
                if (min_i > MR) min_i = (min_i / MR) * MR;

                /* GEMM_I{T,N}COPY of A: for !TRANS use TCOPY (normal A),
                 * for TRANS use NCOPY. */
                if (!trans) {
                    mblas_egemm_tcopy(min_l, min_i,
                                      &a[(size_t)ls * lda], lda, Ap);
                } else {
                    mblas_egemm_ncopy(min_l, min_i,
                                      &a[(size_t)ls], lda, Ap);
                }

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;

                    mblas_egemm_ncopy(min_l, min_jj,
                                      &b[(size_t)ls + (size_t)jjs * ldb], ldb,
                                      Bp + (size_t)min_l * (jjs - js));

                    /* Pure GEMM into b[(jjs*ldb)..] — overwrite semantics */
                    mblas_egemm_kernel_store(min_i, min_jj, min_l, dp1,
                                             Ap, Bp + (size_t)min_l * (jjs - js),
                                             &b[(size_t)jjs * ldb], ldb);
                }

                for (int is = min_i; is < ls; is += min_i) {
                    min_i = ls - is;
                    if (min_i > MC) min_i = MC;
                    if (min_i > MR) min_i = (min_i / MR) * MR;

                    if (!trans) {
                        mblas_egemm_tcopy(min_l, min_i,
                                          &a[(size_t)is + (size_t)ls * lda], lda, Ap);
                    } else {
                        mblas_egemm_ncopy(min_l, min_i,
                                          &a[(size_t)ls + (size_t)is * lda], lda, Ap);
                    }

                    mblas_egemm_kernel_store(min_i, min_j, min_l, dp1,
                                             Ap, Bp,
                                             &b[(size_t)is + (size_t)js * ldb], ldb);
                }

                for (int is = ls; is < ls + min_l; is += min_i) {
                    min_i = ls + min_l - is;
                    if (min_i > MC) min_i = MC;
                    if (min_i > MR) min_i = (min_i / MR) * MR;

                    pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda,
                                /*posX=*/ls, /*posY=*/is, Ap);

                    mblas_etrmm_kernel(/*left=*/1, /*trans=*/kt,
                                       min_i, min_j, min_l, dp1,
                                       Ap, Bp,
                                       &b[(size_t)is + (size_t)js * ldb], ldb,
                                       /*offset=*/(is - ls));
                }
            }
        } else {
            /* The other branch: (UPPER && TRANS) || (LOWER && !TRANS).
             * trmm_L.c lines 301-488: uses TRMM_KERNEL_T = TRMM_KERNEL_LT
             * (kernel trans-flag = 1). Walk up-diagonal (ls from m - min_l
             * down). */
            const int kt = 1;   /* TRMM_KERNEL_T */
            int min_l = m;
            if (min_l > KC) min_l = KC;
            int min_i = min_l;
            if (min_i > MC) min_i = MC;
            if (min_i > MR) min_i = (min_i / MR) * MR;

            pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda,
                        /*posX=*/m - min_l, /*posY=*/m - min_l, Ap);

            for (int jjs = js; jjs < js + min_j; jjs += NR) {
                int min_jj = js + min_j - jjs;
                if (min_jj > NR) min_jj = NR;

                mblas_egemm_ncopy(min_l, min_jj,
                                  &b[(size_t)(m - min_l) + (size_t)jjs * ldb], ldb,
                                  Bp + (size_t)min_l * (jjs - js));

                mblas_etrmm_kernel(/*left=*/1, /*trans=*/kt,
                                   min_i, min_jj, min_l, dp1,
                                   Ap, Bp + (size_t)min_l * (jjs - js),
                                   &b[(size_t)(m - min_l) + (size_t)jjs * ldb], ldb,
                                   /*offset=*/0);
            }

            for (int is = m - min_l + min_i; is < m; is += min_i) {
                min_i = m - is;
                if (min_i > MC) min_i = MC;
                if (min_i > MR) min_i = (min_i / MR) * MR;

                pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda,
                            /*posX=*/m - min_l, /*posY=*/is, Ap);

                mblas_etrmm_kernel(/*left=*/1, /*trans=*/kt,
                                   min_i, min_j, min_l, dp1,
                                   Ap, Bp,
                                   &b[(size_t)is + (size_t)js * ldb], ldb,
                                   /*offset=*/(is - m + min_l));
            }

            for (int ls = m - min_l; ls > 0; ls -= KC) {
                min_l = ls;
                if (min_l > KC) min_l = KC;
                min_i = min_l;
                if (min_i > MC) min_i = MC;
                if (min_i > MR) min_i = (min_i / MR) * MR;

                pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda,
                            /*posX=*/ls - min_l, /*posY=*/ls - min_l, Ap);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;

                    mblas_egemm_ncopy(min_l, min_jj,
                                      &b[(size_t)(ls - min_l) + (size_t)jjs * ldb], ldb,
                                      Bp + (size_t)min_l * (jjs - js));

                    mblas_etrmm_kernel(/*left=*/1, /*trans=*/kt,
                                       min_i, min_jj, min_l, dp1,
                                       Ap, Bp + (size_t)min_l * (jjs - js),
                                       &b[(size_t)(ls - min_l) + (size_t)jjs * ldb], ldb,
                                       /*offset=*/0);
                }

                for (int is = ls - min_l + min_i; is < ls; is += min_i) {
                    min_i = ls - is;
                    if (min_i > MC) min_i = MC;
                    if (min_i > MR) min_i = (min_i / MR) * MR;

                    pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda,
                                /*posX=*/ls - min_l, /*posY=*/is, Ap);

                    mblas_etrmm_kernel(/*left=*/1, /*trans=*/kt,
                                       min_i, min_j, min_l, dp1,
                                       Ap, Bp,
                                       &b[(size_t)is + (size_t)js * ldb], ldb,
                                       /*offset=*/(is - ls + min_l));
                }

                for (int is = ls; is < m; is += min_i) {
                    min_i = m - is;
                    if (min_i > MC) min_i = MC;
                    if (min_i > MR) min_i = (min_i / MR) * MR;

                    /* GEMM_I{T,N}COPY of A: !TRANS → TCOPY of A at
                     * (is, ls-min_l); TRANS → NCOPY of A at (ls-min_l, is). */
                    if (!trans) {
                        mblas_egemm_tcopy(min_l, min_i,
                                          &a[(size_t)is + (size_t)(ls - min_l) * lda], lda, Ap);
                    } else {
                        mblas_egemm_ncopy(min_l, min_i,
                                          &a[(size_t)(ls - min_l) + (size_t)is * lda], lda, Ap);
                    }

                    mblas_egemm_kernel_store(min_i, min_j, min_l, dp1,
                                             Ap, Bp,
                                             &b[(size_t)is + (size_t)js * ldb], ldb);
                }
            }
        }
    }
}


/* ── SIDE='R' driver: port of trmm_R.c for one M-band ────────────────
 *
 * Each thread runs the FULL js/ls/is nest over its own M-slice (no
 * inter-thread sync needed since each thread reads/writes disjoint
 * M-row slices of B; A is read-only).
 *
 * The driver structure for SIDE='R' is more involved — `sa` is OCOPY
 * of B (B is the input being transformed), `sb` is the packed
 * triangular A. See trmm_R.c lines 109-241 for the
 * (!UPPER && !TRANS) || (UPPER && TRANS) branch, and 244-382 for the
 * opposite.
 */
static void trmm_R_band(int upper, int trans, int unit,
                        int N, int m_lo, int m_hi,
                        int MC, int KC, int NC,
                        const T *a, int lda,
                        T *b, int ldb,
                        T *Ap, T *Bp)
{
    (void)MC;
    const T dp1 = 1.0;
    int m_band = m_hi - m_lo;
    if (m_band <= 0) return;

    /* For SIDE='R', sa is OCOPY of B (one M-row strip), sb is the packed
     * A (triangular packer + GEMM packer combinations). The naming gets
     * confusing — Ap holds sa (a K-strip of B), Bp holds sb (the A
     * panels). The kernel sees `ba=sa, bb=sb` with bb being the
     * triangular A. */
    T *sa = Ap;   /* MC × KC slab for B's OCOPY */
    T *sb = Bp;   /* KC × NC slab for A's pack (incl. TRMM and GEMM) */

    if ((!upper && !trans) || (upper && trans)) {
        /* trmm_R.c lines 109-241. Uses TRMM_KERNEL_T = TRMM_KERNEL_RT
         * (kernel runs in (left=0, trans=1) mode). */
        const int kt = 1;
        for (int js = 0; js < N; js += NC) {
            int min_j = N - js;
            if (min_j > NC) min_j = NC;

            for (int ls = js; ls < js + min_j; ls += KC) {
                int min_l = js + min_j - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                /* GEMM_ITCOPY(min_l, min_i, b + ls*ldb, ldb, sa) — pack
                 * B[m_lo..m_lo+min_i, ls..ls+min_l] in TCOPY shape. */
                mblas_egemm_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                /* Off-diagonal GEMM pack of A's left part. */
                for (int jjs = 0; jjs < ls - js; jjs += NR) {
                    int min_jj = ls - js - jjs;
                    if (min_jj > NR) min_jj = NR;

                    /* GEMM_O{N,T}COPY(min_l, min_jj, A_slice, lda, sb + ...) */
                    if (!trans) {
                        mblas_egemm_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)(js + jjs) * lda], lda,
                                          sb + (size_t)min_l * jjs);
                    } else {
                        mblas_egemm_tcopy(min_l, min_jj,
                                          &a[(size_t)(js + jjs) + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * jjs);
                    }

                    mblas_egemm_kernel_store(min_i, min_jj, min_l, dp1,
                                             sa, sb + (size_t)min_l * jjs,
                                             &b[(size_t)m_lo + (size_t)(js + jjs) * ldb], ldb);
                }

                /* Diagonal block: TRMM_O*COPY then TRMM_KERNEL_T. */
                for (int jjs = 0; jjs < min_l; jjs += NR) {
                    int min_jj = min_l - jjs;
                    if (min_jj > NR) min_jj = NR;

                    /* TRMM_O{LN,UT}COPY(min_l, min_jj, a, lda, ls, ls+jjs, ...) */
                    pack_trmm_a(0, upper, trans, unit, min_l, min_jj, a, lda,
                                /*posX=*/ls, /*posY=*/ls + jjs,
                                sb + (size_t)min_l * (ls - js + jjs));

                    mblas_etrmm_kernel(/*left=*/0, /*trans=*/kt,
                                       min_i, min_jj, min_l, dp1,
                                       sa, sb + (size_t)min_l * (ls - js + jjs),
                                       &b[(size_t)m_lo + (size_t)(ls + jjs) * ldb], ldb,
                                       /*offset=*/-jjs);
                }

                /* Continue with more M-tiles (within the band). */
                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    mblas_egemm_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);

                    /* GEMM_KERNEL against the off-diagonal A pack (sb). */
                    mblas_egemm_kernel_store(min_i, ls - js, min_l, dp1,
                                             sa, sb,
                                             &b[(size_t)(m_lo + is) + (size_t)js * ldb], ldb);

                    /* TRMM_KERNEL against the diagonal A pack. */
                    mblas_etrmm_kernel(/*left=*/0, /*trans=*/kt,
                                       min_i, min_l, min_l, dp1,
                                       sa, sb + (size_t)(ls - js) * min_l,
                                       &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb,
                                       /*offset=*/0);
                }
            }

            /* Pure-GEMM tail for ls > js+min_j. */
            for (int ls = js + min_j; ls < N; ls += KC) {
                int min_l = N - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                mblas_egemm_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;

                    if (!trans) {
                        mblas_egemm_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)jjs * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    } else {
                        mblas_egemm_tcopy(min_l, min_jj,
                                          &a[(size_t)jjs + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    }

                    mblas_egemm_kernel_store(min_i, min_jj, min_l, dp1,
                                             sa, sb + (size_t)min_l * (jjs - js),
                                             &b[(size_t)m_lo + (size_t)jjs * ldb], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    mblas_egemm_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);

                    mblas_egemm_kernel_store(min_i, min_j, min_l, dp1,
                                             sa, sb,
                                             &b[(size_t)(m_lo + is) + (size_t)js * ldb], ldb);
                }
            }
        }
    } else {
        /* trmm_R.c lines 244-382: (!UPPER && TRANS) || (UPPER && !TRANS).
         * Uses TRMM_KERNEL_N = TRMM_KERNEL_RN (kernel runs in (left=0,
         * trans=0) mode). */
        const int kt = 0;
        for (int js = N; js > 0; js -= NC) {
            int min_j = js;
            if (min_j > NC) min_j = NC;

            int start_ls = js - min_j;
            while (start_ls + KC < js) start_ls += KC;

            for (int ls = start_ls; ls >= js - min_j; ls -= KC) {
                int min_l = js - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                mblas_egemm_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                /* Diagonal triangular A pack (TRMM_O{UN,LT}COPY). */
                for (int jjs = 0; jjs < min_l; jjs += NR) {
                    int min_jj = min_l - jjs;
                    if (min_jj > NR) min_jj = NR;

                    pack_trmm_a(0, upper, trans, unit, min_l, min_jj, a, lda,
                                /*posX=*/ls, /*posY=*/ls + jjs,
                                sb + (size_t)min_l * jjs);

                    mblas_etrmm_kernel(/*left=*/0, /*trans=*/kt,
                                       min_i, min_jj, min_l, dp1,
                                       sa, sb + (size_t)min_l * jjs,
                                       &b[(size_t)m_lo + (size_t)(ls + jjs) * ldb], ldb,
                                       /*offset=*/-jjs);
                }

                /* Off-diagonal GEMM pack of A's right part. */
                for (int jjs = 0; jjs < js - ls - min_l; jjs += NR) {
                    int min_jj = js - ls - min_l - jjs;
                    if (min_jj > NR) min_jj = NR;

                    if (!trans) {
                        mblas_egemm_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)(ls + min_l + jjs) * lda], lda,
                                          sb + (size_t)min_l * (min_l + jjs));
                    } else {
                        mblas_egemm_tcopy(min_l, min_jj,
                                          &a[(size_t)(ls + min_l + jjs) + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * (min_l + jjs));
                    }

                    mblas_egemm_kernel_store(min_i, min_jj, min_l, dp1,
                                             sa, sb + (size_t)min_l * (min_l + jjs),
                                             &b[(size_t)m_lo + (size_t)(ls + min_l + jjs) * ldb], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    mblas_egemm_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);

                    mblas_etrmm_kernel(/*left=*/0, /*trans=*/kt,
                                       min_i, min_l, min_l, dp1,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb,
                                       /*offset=*/0);

                    if (js - ls - min_l > 0) {
                        mblas_egemm_kernel_store(min_i, js - ls - min_l, min_l, dp1,
                                                 sa, sb + (size_t)min_l * min_l,
                                                 &b[(size_t)(m_lo + is) + (size_t)(ls + min_l) * ldb], ldb);
                    }
                }
            }

            /* Pure-GEMM tail for ls < js - min_j. */
            for (int ls = 0; ls < js - min_j; ls += KC) {
                int min_l = js - min_j - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                mblas_egemm_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = min_j + js - jjs;
                    if (min_jj > NR) min_jj = NR;

                    if (!trans) {
                        mblas_egemm_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)(jjs - min_j) * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    } else {
                        mblas_egemm_tcopy(min_l, min_jj,
                                          &a[(size_t)(jjs - min_j) + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    }

                    mblas_egemm_kernel_store(min_i, min_jj, min_l, dp1,
                                             sa, sb + (size_t)min_l * (jjs - js),
                                             &b[(size_t)m_lo + (size_t)(jjs - min_j) * ldb], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    mblas_egemm_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);

                    mblas_egemm_kernel_store(min_i, min_j, min_l, dp1,
                                             sa, sb,
                                             &b[(size_t)(m_lo + is) + (size_t)(js - min_j) * ldb], ldb);
                }
            }
        }
    }
}


/* ── Public entry ───────────────────────────────────────────────── */
extern "C" void mtrmm_(
    const char *side_p, const char *uplo_p,
    const char *transa_p, const char *diag_p,
    const int *m_, const int *n_, const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)
{
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;

    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;

    const int lside  = (toupper((unsigned char)*side_p)  == 'L');
    const int upper  = (toupper((unsigned char)*uplo_p)  == 'U');
    const char trc   = (char)toupper((unsigned char)*transa_p);
    const int trans  = (trc == 'T' || trc == 'C');
    const int nounit = (toupper((unsigned char)*diag_p)  == 'N');
    const int unit   = !nounit;

    if (M == 0 || N == 0) return;

    /* alpha pre-scale of B (mirrors trmm_L.c lines 103-113: if (beta!=ONE)
     * GEMM_BETA(...); if (beta==ZERO) return). */
    if (alpha != 1.0) {
        mblas_egemm_beta((ptrdiff_t)M, (ptrdiff_t)N, alpha, b, (ptrdiff_t)ldb);
    }
    if (alpha == 0.0) return;

    /* Block sizes (env-overridable). */
    int MC0, KC, NC;
    mblas_egemm_blocks(&MC0, &KC, &NC);

    /* Adaptive MC for small K (mirrors egemm). For SIDE='L' K = M; for
     * SIDE='R' K = N. */
    int K_eff = lside ? M : N;
    int MC = MC0;
    if (K_eff <= KC) {
        const long L2_TARGET_BYTES = 256L * 1024L;
        long target_mc = L2_TARGET_BYTES / ((long)K_eff * (long)sizeof(T));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((int)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    /* Per-thread Ap and Bp. The Ap buffer holds A-side packs (TRMM-pack
     * for diagonal, GEMM-pack for off-diagonal); Bp holds B-side OCOPY
     * panels. Sizes mirror egemm but multiplied by 2 to comfortably hold
     * either an MC*KC slab or a KC*NC slab (we use Ap and Bp for distinct
     * roles in the L/R drivers, but the larger of the two roles bounds
     * the size). */
    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * sizeof(T);
    /* For SIDE='R', Ap holds an MC*KC strip and Bp holds a KC*NC slab.
     * For SIDE='L', Ap holds an MC*KC strip (the triangular A pack) and
     * Bp holds a KC*NC slab (B OCOPY). Same sizes either way. */

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
#endif

    long mnk = (long)M * (long)N * (long)K_eff;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

    /* Cap threads to the partition-axis size (no benefit running more
     * threads than M-slices for SIDE='R' or N-slices for SIDE='L'). */
    int partition_axis = lside ? N : M;
    if (nthreads > partition_axis) nthreads = partition_axis;
    if (nthreads < 1) nthreads = 1;

    T **Ap_arr = (T **)calloc((size_t)nthreads, sizeof(T *));
    T **Bp_arr = (T **)calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_arr || !Bp_arr) { free(Ap_arr); free(Bp_arr); return; }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = (T *)aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        Bp_arr[t] = (T *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t] || !Bp_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) {
            if (Ap_arr) free(Ap_arr[t]);
            if (Bp_arr) free(Bp_arr[t]);
        }
        free(Ap_arr); free(Bp_arr);
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
        T *Bp = Bp_arr[tid];

        if (lside) {
            /* N-axis partition. Each thread's range [js0, js1). */
            int chunk = round_up((N + nth - 1) / nth, NR);
            int js0 = tid * chunk;
            int js1 = js0 + chunk;
            if (js0 > N) js0 = N;
            if (js1 > N) js1 = N;

            if (js0 < js1) {
                trmm_L_band(upper, trans, unit,
                            M, js0, js1, M,
                            MC, KC, NC,
                            a, lda, b, ldb,
                            Ap, Bp);
            }
        } else {
            /* M-axis partition. */
            int chunk = round_up((M + nth - 1) / nth, MR);
            int m_lo = tid * chunk;
            int m_hi = m_lo + chunk;
            if (m_lo > M) m_lo = M;
            if (m_hi > M) m_hi = M;

            if (m_lo < m_hi) {
                trmm_R_band(upper, trans, unit,
                            N, m_lo, m_hi,
                            MC, KC, NC,
                            a, lda, b, ldb,
                            Ap, Bp);
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) {
        free(Ap_arr[t]);
        free(Bp_arr[t]);
    }
    free(Ap_arr);
    free(Bp_arr);
}
