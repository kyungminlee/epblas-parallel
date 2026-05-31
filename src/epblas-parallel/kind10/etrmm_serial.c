/*
 * etrmm_serial — kind10 (REAL(KIND=10) / long double) triangular
 * matrix-multiply, single-thread half of the etrmm overlay. Owns the
 * SIDE='L'/'R' L3 band drivers (faithful ports of OpenBLAS
 * driver/level3/trmm_{L,R}.c) and the pure-serial Fortran-ABI entry
 * `etrmm_serial`. The math primitives live in etrmm_kernel.c / etrmm_pack.c
 * and the shared substrate etri_kernel.c.
 *
 * Each band driver runs the full trmm_{L,R}.c nest over one slice of the
 * free axis (B columns for SIDE='L', rows for SIDE='R') with caller-owned
 * per-thread Ap/Bp scratch, so etrmm_parallel.c can give each thread a
 * disjoint slice with no cross-thread synchronization. The serial entry
 * runs one band spanning the whole free axis.
 *
 * The TRMM kernel overwrites B (C := alpha·A·B); off-diagonal sub-tiles go
 * through etri_kernel_store (zero + accumulate, matching OpenBLAS's
 * GEMM_KERNEL beta=0 inside the TRMM driver). alpha pre-scaling of B is done
 * by the entry (egemm_beta_prepass), so the band drivers run kernel-alpha=1.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>

#include "etrmm_kernel.h"
#include "etri_kernel.h"    /* etri_ncopy / etri_tcopy / etri_kernel_store */
#include "egemm_kernel.h"   /* egemm_choose_blocks / egemm_beta_prepass / egemm_round_up */

typedef etrmm_T T;

/* Register-tile dims — must match the packed layout (etrmm_kernel.c). */
#define MR 2
#define NR 2

static inline void pack_trmm_a(int side_l, int uplo_upper, int trans, int unit,
                               ptrdiff_t m, ptrdiff_t n,
                               const T *a, ptrdiff_t lda,
                               ptrdiff_t posX, ptrdiff_t posY,
                               T *bp)
{
    if (side_l) {
        if (uplo_upper && !trans)       etrmm_iutcopy(m, n, a, lda, posX, posY, bp, unit);
        else if (uplo_upper &&  trans)  etrmm_iuncopy(m, n, a, lda, posX, posY, bp, unit);
        else if (!uplo_upper && !trans) etrmm_iltcopy(m, n, a, lda, posX, posY, bp, unit);
        else                            etrmm_ilncopy(m, n, a, lda, posX, posY, bp, unit);
    } else {
        if (uplo_upper && !trans)       etrmm_iuncopy(m, n, a, lda, posX, posY, bp, unit);
        else if (uplo_upper &&  trans)  etrmm_iutcopy(m, n, a, lda, posX, posY, bp, unit);
        else if (!uplo_upper && !trans) etrmm_ilncopy(m, n, a, lda, posX, posY, bp, unit);
        else                            etrmm_iltcopy(m, n, a, lda, posX, posY, bp, unit);
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
void etrmm_L_band(int upper, int trans, int unit,
                        int M, int js0, int js1,
                        int MC, int KC, int NC,
                        const T *a, int lda,
                        T *b, int ldb,
                        T *Ap, T *Bp)
{
    const T dp1 = 1.0L;
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
                etri_ncopy(min_l, min_jj,
                                  &b[(size_t)jjs * ldb], ldb,
                                  Bp + (size_t)min_l * (jjs - js));

                /* TRMM_KERNEL_N(min_i, min_jj, min_l, dp1, sa, sb_slice,
                 *               b + jjs*ldb, ldb, 0) */
                etrmm_kernel(/*left=*/1, /*trans=*/kt,
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

                etrmm_kernel(/*left=*/1, /*trans=*/kt,
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
                    etri_tcopy(min_l, min_i,
                                      &a[(size_t)ls * lda], lda, Ap);
                } else {
                    etri_ncopy(min_l, min_i,
                                      &a[(size_t)ls], lda, Ap);
                }

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;

                    etri_ncopy(min_l, min_jj,
                                      &b[(size_t)ls + (size_t)jjs * ldb], ldb,
                                      Bp + (size_t)min_l * (jjs - js));

                    /* Pure GEMM into b[(jjs*ldb)..] — overwrite semantics */
                    etri_kernel_store(min_i, min_jj, min_l, dp1,
                                             Ap, Bp + (size_t)min_l * (jjs - js),
                                             &b[(size_t)jjs * ldb], ldb);
                }

                for (int is = min_i; is < ls; is += min_i) {
                    min_i = ls - is;
                    if (min_i > MC) min_i = MC;
                    if (min_i > MR) min_i = (min_i / MR) * MR;

                    if (!trans) {
                        etri_tcopy(min_l, min_i,
                                          &a[(size_t)is + (size_t)ls * lda], lda, Ap);
                    } else {
                        etri_ncopy(min_l, min_i,
                                          &a[(size_t)ls + (size_t)is * lda], lda, Ap);
                    }

                    etri_kernel_store(min_i, min_j, min_l, dp1,
                                             Ap, Bp,
                                             &b[(size_t)is + (size_t)js * ldb], ldb);
                }

                for (int is = ls; is < ls + min_l; is += min_i) {
                    min_i = ls + min_l - is;
                    if (min_i > MC) min_i = MC;
                    if (min_i > MR) min_i = (min_i / MR) * MR;

                    pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda,
                                /*posX=*/ls, /*posY=*/is, Ap);

                    etrmm_kernel(/*left=*/1, /*trans=*/kt,
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

                etri_ncopy(min_l, min_jj,
                                  &b[(size_t)(m - min_l) + (size_t)jjs * ldb], ldb,
                                  Bp + (size_t)min_l * (jjs - js));

                etrmm_kernel(/*left=*/1, /*trans=*/kt,
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

                etrmm_kernel(/*left=*/1, /*trans=*/kt,
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

                    etri_ncopy(min_l, min_jj,
                                      &b[(size_t)(ls - min_l) + (size_t)jjs * ldb], ldb,
                                      Bp + (size_t)min_l * (jjs - js));

                    etrmm_kernel(/*left=*/1, /*trans=*/kt,
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

                    etrmm_kernel(/*left=*/1, /*trans=*/kt,
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
                        etri_tcopy(min_l, min_i,
                                          &a[(size_t)is + (size_t)(ls - min_l) * lda], lda, Ap);
                    } else {
                        etri_ncopy(min_l, min_i,
                                          &a[(size_t)(ls - min_l) + (size_t)is * lda], lda, Ap);
                    }

                    etri_kernel_store(min_i, min_j, min_l, dp1,
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
void etrmm_R_band(int upper, int trans, int unit,
                        int N, int m_lo, int m_hi,
                        int MC, int KC, int NC,
                        const T *a, int lda,
                        T *b, int ldb,
                        T *Ap, T *Bp)
{
    (void)MC;
    const T dp1 = 1.0L;
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
                etri_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                /* Off-diagonal GEMM pack of A's left part. */
                for (int jjs = 0; jjs < ls - js; jjs += NR) {
                    int min_jj = ls - js - jjs;
                    if (min_jj > NR) min_jj = NR;

                    /* GEMM_O{N,T}COPY(min_l, min_jj, A_slice, lda, sb + ...) */
                    if (!trans) {
                        etri_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)(js + jjs) * lda], lda,
                                          sb + (size_t)min_l * jjs);
                    } else {
                        etri_tcopy(min_l, min_jj,
                                          &a[(size_t)(js + jjs) + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * jjs);
                    }

                    etri_kernel_store(min_i, min_jj, min_l, dp1,
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

                    etrmm_kernel(/*left=*/0, /*trans=*/kt,
                                       min_i, min_jj, min_l, dp1,
                                       sa, sb + (size_t)min_l * (ls - js + jjs),
                                       &b[(size_t)m_lo + (size_t)(ls + jjs) * ldb], ldb,
                                       /*offset=*/-jjs);
                }

                /* Continue with more M-tiles (within the band). */
                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    etri_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);

                    /* GEMM_KERNEL against the off-diagonal A pack (sb). */
                    etri_kernel_store(min_i, ls - js, min_l, dp1,
                                             sa, sb,
                                             &b[(size_t)(m_lo + is) + (size_t)js * ldb], ldb);

                    /* TRMM_KERNEL against the diagonal A pack. */
                    etrmm_kernel(/*left=*/0, /*trans=*/kt,
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

                etri_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;

                    if (!trans) {
                        etri_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)jjs * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    } else {
                        etri_tcopy(min_l, min_jj,
                                          &a[(size_t)jjs + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    }

                    etri_kernel_store(min_i, min_jj, min_l, dp1,
                                             sa, sb + (size_t)min_l * (jjs - js),
                                             &b[(size_t)m_lo + (size_t)jjs * ldb], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    etri_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);

                    etri_kernel_store(min_i, min_j, min_l, dp1,
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

                etri_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                /* Diagonal triangular A pack (TRMM_O{UN,LT}COPY). */
                for (int jjs = 0; jjs < min_l; jjs += NR) {
                    int min_jj = min_l - jjs;
                    if (min_jj > NR) min_jj = NR;

                    pack_trmm_a(0, upper, trans, unit, min_l, min_jj, a, lda,
                                /*posX=*/ls, /*posY=*/ls + jjs,
                                sb + (size_t)min_l * jjs);

                    etrmm_kernel(/*left=*/0, /*trans=*/kt,
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
                        etri_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)(ls + min_l + jjs) * lda], lda,
                                          sb + (size_t)min_l * (min_l + jjs));
                    } else {
                        etri_tcopy(min_l, min_jj,
                                          &a[(size_t)(ls + min_l + jjs) + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * (min_l + jjs));
                    }

                    etri_kernel_store(min_i, min_jj, min_l, dp1,
                                             sa, sb + (size_t)min_l * (min_l + jjs),
                                             &b[(size_t)m_lo + (size_t)(ls + min_l + jjs) * ldb], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    etri_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);

                    etrmm_kernel(/*left=*/0, /*trans=*/kt,
                                       min_i, min_l, min_l, dp1,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb,
                                       /*offset=*/0);

                    if (js - ls - min_l > 0) {
                        etri_kernel_store(min_i, js - ls - min_l, min_l, dp1,
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

                etri_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = min_j + js - jjs;
                    if (min_jj > NR) min_jj = NR;

                    if (!trans) {
                        etri_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)(jjs - min_j) * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    } else {
                        etri_tcopy(min_l, min_jj,
                                          &a[(size_t)(jjs - min_j) + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    }

                    etri_kernel_store(min_i, min_jj, min_l, dp1,
                                             sa, sb + (size_t)min_l * (jjs - js),
                                             &b[(size_t)m_lo + (size_t)(jjs - min_j) * ldb], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    etri_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);

                    etri_kernel_store(min_i, min_j, min_l, dp1,
                                             sa, sb,
                                             &b[(size_t)(m_lo + is) + (size_t)(js - min_j) * ldb], ldb);
                }
            }
        }
    }
}

/* ── Pure-serial Fortran-ABI entry ───────────────────────────────────
 *
 * Runs the full SIDE/UPLO/TRANSA dispatch over one band spanning the whole
 * free axis (no threading). etrmm_parallel.c's etrmm_ delegates here when
 * called from inside another routine's parallel region.
 */
void etrmm_serial(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)
{
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;

    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;

    const int lside = (toupper((unsigned char)*side)   == 'L');
    const int upper = (toupper((unsigned char)*uplo)   == 'U');
    const char trc  = (char)toupper((unsigned char)*transa);
    const int trans = (trc == 'T' || trc == 'C');   /* real: 'C' ≡ 'T' */
    const int unit  = (toupper((unsigned char)*diag) == 'U');

    if (M == 0 || N == 0) return;

    /* α pre-scale of B in place, then the nest runs kernel-alpha = 1
     * (mirrors trmm_{L,R}.c GEMM_BETA pass; alpha == 0 → B := 0). */
    if (alpha != 1.0L) egemm_beta_prepass(M, N, alpha, b, ldb);
    if (alpha == 0.0L) return;

    const int K_eff = lside ? M : N;
    int MC, KC, NC;
    egemm_choose_blocks(K_eff, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)egemm_round_up(NC, NR) * sizeof(T);
    T *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (Ap && Bp) {
        if (lside) etrmm_L_band(upper, trans, unit, M, 0, N,
                                MC, KC, NC, a, lda, b, ldb, Ap, Bp);
        else       etrmm_R_band(upper, trans, unit, N, 0, M,
                                MC, KC, NC, a, lda, b, ldb, Ap, Bp);
    }
    free(Ap);
    free(Bp);
}
