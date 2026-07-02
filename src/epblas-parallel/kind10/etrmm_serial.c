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
#include <stdbool.h>
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include <stdlib.h>
#include <ctype.h>

#include "etrmm_kernel.h"
#include "etri_kernel.h"    /* etri_ncopy / etri_tcopy / etri_kernel_store */
#include "egemm_kernel.h"   /* egemm_choose_blocks / egemm_beta_prepass / blas_round_up */

typedef etrmm_TR TR;

/* Register-tile dims — must match the packed layout (etrmm_kernel.c). */
#define MR 2
#define NR 2

static inline void pack_trmm_a(bool side_l, bool uplo_upper, bool trans, bool unit,
                               ptrdiff_t m, ptrdiff_t n,
                               const TR *a, ptrdiff_t lda,
                               ptrdiff_t posX, ptrdiff_t posY,
                               TR *bp)
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
void etrmm_L_band(bool upper, bool trans, bool unit,
                        ptrdiff_t m, ptrdiff_t js0, ptrdiff_t js1,
                        ptrdiff_t MC, ptrdiff_t KC, ptrdiff_t NC,
                        const TR *a, ptrdiff_t lda,
                        TR *b, ptrdiff_t ldb,
                        TR *Ap, TR *Bp)
{
    const TR dp1 = 1.0L;

    /* Outer js-loop walks the thread's N-band in steps of NC = GEMM_R. */
    for (ptrdiff_t js = js0; js < js1; js += NC) {
        ptrdiff_t min_j = js1 - js;
        if (min_j > NC) min_j = NC;

        if ((upper && !trans) || (!upper && trans)) {
            /* trmm_L.c lines 119-299: TRMM_KERNEL_N = TRMM_KERNEL_LN
             * (kernel trans-flag = 0; the user trans is absorbed by the
             * packer choice).
             * Walk down-diagonal: pack A[ls..ls+min_l, ls..ls+min_l] as
             * the triangular block (TRMM_IUTCOPY for UPPER!TRANS,
             * TRMM_ILNCOPY for LOWER+TRANS), then off-diagonal GEMM tiles
             * above the diagonal block. */
            const ptrdiff_t kt = 0;   /* TRMM_KERNEL_N */
            ptrdiff_t min_l = m;
            if (min_l > KC) min_l = KC;
            ptrdiff_t min_i = min_l;
            if (min_i > MC) min_i = MC;
            if (min_i > MR) min_i = (min_i / MR) * MR;

            /* TRMM_I*COPY(min_l, min_i, a, lda, posX=0, posY=0, sa) */
            pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda, 0, 0, Ap);

            /* Inner jjs loop — process min_j cols in NR-sized strips.
             * For our NR=2, the OpenBLAS jjs stride logic resolves to
             * min_jj = NR (or the trailing remnant). */
            for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                ptrdiff_t min_jj = js + min_j - jjs;
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
            for (ptrdiff_t is = min_i; is < min_l; is += min_i) {
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
            for (ptrdiff_t ls = min_l; ls < m; ls += KC) {
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

                for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                    ptrdiff_t min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;

                    etri_ncopy(min_l, min_jj,
                                      &b[(size_t)ls + (size_t)jjs * ldb], ldb,
                                      Bp + (size_t)min_l * (jjs - js));

                    /* Pure GEMM into b[(jjs*ldb)..] — overwrite semantics */
                    etri_kernel_store(min_i, min_jj, min_l, dp1,
                                             Ap, Bp + (size_t)min_l * (jjs - js),
                                             &b[(size_t)jjs * ldb], ldb);
                }

                for (ptrdiff_t is = min_i; is < ls; is += min_i) {
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

                for (ptrdiff_t is = ls; is < ls + min_l; is += min_i) {
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
            const ptrdiff_t kt = 1;   /* TRMM_KERNEL_T */
            ptrdiff_t min_l = m;
            if (min_l > KC) min_l = KC;
            ptrdiff_t min_i = min_l;
            if (min_i > MC) min_i = MC;
            if (min_i > MR) min_i = (min_i / MR) * MR;

            pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda,
                        /*posX=*/m - min_l, /*posY=*/m - min_l, Ap);

            for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                ptrdiff_t min_jj = js + min_j - jjs;
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

            for (ptrdiff_t is = m - min_l + min_i; is < m; is += min_i) {
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

            for (ptrdiff_t ls = m - min_l; ls > 0; ls -= KC) {
                min_l = ls;
                if (min_l > KC) min_l = KC;
                min_i = min_l;
                if (min_i > MC) min_i = MC;
                if (min_i > MR) min_i = (min_i / MR) * MR;

                pack_trmm_a(1, upper, trans, unit, min_l, min_i, a, lda,
                            /*posX=*/ls - min_l, /*posY=*/ls - min_l, Ap);

                for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                    ptrdiff_t min_jj = js + min_j - jjs;
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

                for (ptrdiff_t is = ls - min_l + min_i; is < ls; is += min_i) {
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

                for (ptrdiff_t is = ls; is < m; is += min_i) {
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
void etrmm_R_band(bool upper, bool trans, bool unit,
                        ptrdiff_t n, ptrdiff_t m_lo, ptrdiff_t m_hi,
                        ptrdiff_t MC, ptrdiff_t KC, ptrdiff_t NC,
                        const TR *a, ptrdiff_t lda,
                        TR *b, ptrdiff_t ldb,
                        TR *Ap, TR *Bp)
{
    (void)MC;
    const TR dp1 = 1.0L;
    ptrdiff_t m_band = m_hi - m_lo;
    if (m_band <= 0) return;

    /* For SIDE='R', sa is OCOPY of B (one M-row strip), sb is the packed
     * A (triangular packer + GEMM packer combinations). The naming gets
     * confusing — Ap holds sa (a K-strip of B), Bp holds sb (the A
     * panels). The kernel sees `ba=sa, bb=sb` with bb being the
     * triangular A. */
    TR *sa = Ap;   /* MC × KC slab for B's OCOPY */
    TR *sb = Bp;   /* KC × NC slab for A's pack (incl. TRMM and GEMM) */

    if ((!upper && !trans) || (upper && trans)) {
        /* trmm_R.c lines 109-241. Uses TRMM_KERNEL_T = TRMM_KERNEL_RT
         * (kernel runs in (left=0, trans=1) mode). */
        const ptrdiff_t kt = 1;
        for (ptrdiff_t js = 0; js < n; js += NC) {
            ptrdiff_t min_j = n - js;
            if (min_j > NC) min_j = NC;

            for (ptrdiff_t ls = js; ls < js + min_j; ls += KC) {
                ptrdiff_t min_l = js + min_j - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = m_band;
                if (min_i > MC) min_i = MC;

                /* GEMM_ITCOPY(min_l, min_i, b + ls*ldb, ldb, sa) — pack
                 * B[m_lo..m_lo+min_i, ls..ls+min_l] in TCOPY shape. */
                etri_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                /* Off-diagonal GEMM pack of A's left part. */
                for (ptrdiff_t jjs = 0; jjs < ls - js; jjs += NR) {
                    ptrdiff_t min_jj = ls - js - jjs;
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
                for (ptrdiff_t jjs = 0; jjs < min_l; jjs += NR) {
                    ptrdiff_t min_jj = min_l - jjs;
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
                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
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
            for (ptrdiff_t ls = js + min_j; ls < n; ls += KC) {
                ptrdiff_t min_l = n - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = m_band;
                if (min_i > MC) min_i = MC;

                etri_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                    ptrdiff_t min_jj = js + min_j - jjs;
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

                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
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
        const ptrdiff_t kt = 0;
        for (ptrdiff_t js = n; js > 0; js -= NC) {
            ptrdiff_t min_j = js;
            if (min_j > NC) min_j = NC;

            ptrdiff_t start_ls = js - min_j;
            while (start_ls + KC < js) start_ls += KC;

            for (ptrdiff_t ls = start_ls; ls >= js - min_j; ls -= KC) {
                ptrdiff_t min_l = js - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = m_band;
                if (min_i > MC) min_i = MC;

                etri_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                /* Diagonal triangular A pack (TRMM_O{UN,LT}COPY). */
                for (ptrdiff_t jjs = 0; jjs < min_l; jjs += NR) {
                    ptrdiff_t min_jj = min_l - jjs;
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
                for (ptrdiff_t jjs = 0; jjs < js - ls - min_l; jjs += NR) {
                    ptrdiff_t min_jj = js - ls - min_l - jjs;
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

                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
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
            for (ptrdiff_t ls = 0; ls < js - min_j; ls += KC) {
                ptrdiff_t min_l = js - min_j - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = m_band;
                if (min_i > MC) min_i = MC;

                etri_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                    ptrdiff_t min_jj = min_j + js - jjs;
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

                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
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
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const TR *alpha_,
    const TR *a, ptrdiff_t lda,
    TR *b, ptrdiff_t ldb)
{
    const TR alpha = *alpha_;

    const bool lside = (blas_up(side)   == 'L');
    const bool upper = (blas_up(uplo)   == 'U');
    const char TRANS  = blas_up(transa);
    const bool trans = (TRANS == 'T' || TRANS == 'C');   /* real: 'C' ≡ 'T' */
    const bool unit  = (blas_up(diag) == 'U');

    if (m == 0 || n == 0) return;

    /* α pre-scale of B in place, then the nest runs kernel-alpha = 1
     * (mirrors trmm_{L,R}.c GEMM_BETA pass; alpha == 0 → B := 0). */
    if (alpha != 1.0L) egemm_beta_prepass(m, n, alpha, b, ldb);
    if (alpha == 0.0L) return;

    const ptrdiff_t K_eff = lside ? m : n;
    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(K_eff, &MC, &KC, &NC);

    /* Bound the pack buffers to the actual problem, not the full cache-block
     * params — every block is capped by the remaining extent, so a small
     * matrix never packs more than min(block, dim). The per-axis dims are the
     * SAME for both sides: Ap's MR-panel row axis is m (SIDE='L' packs op(A)
     * tiles of ≤min(MC,m) rows; SIDE='R' packs B-row strips of ≤min(MC,m)
     * rows — bounded by m, NOT by K_eff: with m≫n the adaptive MC grown for
     * small K exceeds n and a K_eff bound overflows Ap into Bp), the KC axis
     * is the triangular dim K_eff, and the NC sweep walks B's columns n on
     * both sides. */
    const ptrdiff_t mc_eff = (MC < m)     ? MC : m;
    const ptrdiff_t kc_eff = (KC < K_eff) ? KC : K_eff;
    const ptrdiff_t nc_eff = (NC < n)     ? NC : n;
    const size_t ap_bytes = (size_t)blas_round_up(mc_eff, MR) * (size_t)kc_eff * sizeof(TR);
    const size_t bp_bytes = (size_t)kc_eff * (size_t)blas_round_up(nc_eff, NR) * sizeof(TR);

    /* Persistent, grow-only, thread-local pack scratch (Ap|Bp in one block).
     * A per-call aligned_alloc+free of these 256KB–2MB buffers is NOT free:
     * both exceed glibc's mmap threshold, and freeing both each call trips
     * malloc_trim → the arena is handed back to the OS and re-faulted next
     * call, pure page-fault kernel time (see etrsm_serial.c for the measured
     * fault counts). One grow-only buffer per thread (never freed; released
     * at thread/exit) sidesteps the heuristic entirely. */
    static __thread TR *g_pack = NULL;
    static __thread size_t g_pack_cap = 0;
    const size_t ap_al = ((ap_bytes + 63) & ~(size_t)63) + ETRI_PACK_GUARD;
    const size_t bp_al = ((bp_bytes + 63) & ~(size_t)63) + ETRI_PACK_GUARD;
    const size_t need  = ap_al + bp_al;
    if (need > g_pack_cap) {
        free(g_pack);
        size_t cap = need + (need >> 1);            /* 1.5× headroom to amortize regrow */
        cap = (cap + 63) & ~(size_t)63;
        g_pack = aligned_alloc(64, cap);
        g_pack_cap = g_pack ? cap : 0;
    }
    if (g_pack) {
        TR *Ap = g_pack;
        TR *Bp = (TR *)(void *)((char *)g_pack + ap_al);
        etri_pack_guard_poison(Ap, ap_bytes, ap_al);
        etri_pack_guard_poison(Bp, bp_bytes, bp_al);
        if (lside) etrmm_L_band(upper, trans, unit, m, 0, n,
                                MC, KC, NC, a, lda, b, ldb, Ap, Bp);
        else       etrmm_R_band(upper, trans, unit, n, 0, m,
                                MC, KC, NC, a, lda, b, ldb, Ap, Bp);
        etri_pack_guard_check(Ap, ap_bytes, ap_al, "etrmm_serial Ap");
        etri_pack_guard_check(Bp, bp_bytes, bp_al, "etrmm_serial Bp");
    }
}
