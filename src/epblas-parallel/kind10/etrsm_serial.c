/*
 * etrsm_serial — kind10 (REAL(KIND=10) / long double) triangular solve:
 * the SIDE='L'/'R' L3 band drivers and the pure-serial Fortran-ABI entry
 * `etrsm_serial`. Faithful port of OpenBLAS driver/level3/trsm_{L,R}.c
 * (sibling of the openblas overlay's etrsm.c).
 *
 * A "band" is one contiguous slice of the partition (free) axis: a column
 * band [js0, js1) of B for SIDE='L', a row band [m_lo, m_hi) for SIDE='R'.
 * The serial entry runs one band spanning the whole axis; the threaded
 * entry (etrsm_parallel.c) splits the axis across a team and calls these
 * same drivers, one band per thread, with per-thread Ap/Bp scratch.
 *
 * The numeric substrate (GEMM micro-kernel, ncopy/tcopy, diagonal-aware
 * TRSM kernel) lives in etrsm_kernel.c; the diagonal-inverting A-packers
 * in etrsm_pack.c. Block-size policy (incl. the L2-detected adaptive MC)
 * and the α pre-scale are the layout-agnostic egemm primitives, reused
 * directly — see etrsm_kernel.h for why the rest is NOT shared with egemm.
 *
 * Contains no `#pragma omp`. The libgomp barrier-wedge nesting guard
 * lives in etrsm_parallel.c.
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>

#include "etrsm_kernel.h"
#include "etri_kernel.h"    /* etri_gemm_kernel / etri_ncopy / etri_tcopy */
#include "egemm_kernel.h"   /* egemm_choose_blocks / egemm_beta_prepass / egemm_round_up */

typedef etrsm_T T;

/* Register-tile dims — must match the packed layout (etrsm_kernel.c). */
#define MR 2
#define NR 2

static inline void pack_trsm_a_lside_forward(bool upper, bool trans, bool unit,
                                             ptrdiff_t m, ptrdiff_t n,
                                             const T *a, ptrdiff_t lda,
                                             ptrdiff_t offset, T *bp)
{
    /* forward direction (UPPER+!TRANS or !UPPER+TRANS branch is
     * backwards; this is for !UPPER+!TRANS / UPPER+TRANS). */
    if (!trans) {
        etrsm_iltcopy(m, n, a, lda, offset, bp, unit);
    } else {
        etrsm_iuncopy(m, n, a, lda, offset, bp, unit);
    }
}

static inline void pack_trsm_a_lside_backward(bool upper, bool trans, bool unit,
                                              ptrdiff_t m, ptrdiff_t n,
                                              const T *a, ptrdiff_t lda,
                                              ptrdiff_t offset, T *bp)
{
    /* backward direction: UPPER+!TRANS uses IUTCOPY, !UPPER+TRANS uses ILNCOPY */
    if (!trans) {
        etrsm_iutcopy(m, n, a, lda, offset, bp, unit);
    } else {
        etrsm_ilncopy(m, n, a, lda, offset, bp, unit);
    }
}

static inline void pack_trsm_a_rside_forward(bool upper, bool trans, bool unit,
                                             ptrdiff_t m, ptrdiff_t n,
                                             const T *a, ptrdiff_t lda,
                                             ptrdiff_t offset, T *bp)
{
    /* trsm_R.c lines 170/172: UPPER+!TRANS → OUNCOPY = iuncopy;
     *                          !UPPER+TRANS → OLTCOPY = iltcopy. */
    if (!trans) {
        etrsm_iuncopy(m, n, a, lda, offset, bp, unit);
    } else {
        etrsm_iltcopy(m, n, a, lda, offset, bp, unit);
    }
}

static inline void pack_trsm_a_rside_backward(bool upper, bool trans, bool unit,
                                              ptrdiff_t m, ptrdiff_t n,
                                              const T *a, ptrdiff_t lda,
                                              ptrdiff_t offset, T *bp)
{
    /* trsm_R.c lines 290/293: !UPPER+!TRANS → OLNCOPY = ilncopy;
     *                          UPPER+TRANS → OUTCOPY = iutcopy. */
    if (!trans) {
        etrsm_ilncopy(m, n, a, lda, offset, bp, unit);
    } else {
        etrsm_iutcopy(m, n, a, lda, offset, bp, unit);
    }
}

/* ── SIDE='L' driver: port of trsm_L.c for one N-band [js0..js1) ───── */
void etrsm_L_band(bool upper, bool trans, bool unit,
                        ptrdiff_t M, ptrdiff_t js0, ptrdiff_t js1,
                        ptrdiff_t MC, ptrdiff_t KC, ptrdiff_t NC,
                        const T *a, ptrdiff_t lda,
                        T *b, ptrdiff_t ldb,
                        T *Ap, T *Bp)
{
    const T dm1 = -1.0L;
    ptrdiff_t m = M;
    /* Pick which (uplo, trans) branch (forward vs backward ls). Forward
     * = !UPPER+!TRANS || UPPER+TRANS (ls walks 0..m). */
    const ptrdiff_t forward = (!upper && !trans) || (upper && trans);
    /* TRSM_KERNEL trans flag: forward branch → LT (kt=1); backward → LN (kt=0).
     * See trsm_L.c #define logic at top of file. */
    const ptrdiff_t kt = forward ? 1 : 0;

    for (ptrdiff_t js = js0; js < js1; js += NC) {
        ptrdiff_t min_j = js1 - js;
        if (min_j > NC) min_j = NC;

        if (forward) {
            /* trsm_L.c lines 119-182 (forward ls walk). */
            for (ptrdiff_t ls = 0; ls < m; ls += KC) {
                ptrdiff_t min_l = m - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = min_l;
                if (min_i > MC) min_i = MC;

                /* TRSM_I*COPY of A diagonal block [ls..ls+min_l, ls..ls+min_l].
                 * In trsm_L.c the packed shape is (min_l, min_i) with the
                 * is-iteration packing only `min_i` rows at a time of the
                 * `min_l × min_l` diagonal block. */
                pack_trsm_a_lside_forward(upper, trans, unit,
                                          min_l, min_i,
                                          &a[(size_t)ls + (size_t)ls * lda], lda,
                                          /*offset=*/0, Ap);

                /* jjs loop: pack B-strip [ls..ls+min_l, jjs..jjs+min_jj] then TRSM_KERNEL. */
                for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                    ptrdiff_t min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    etri_ncopy(min_l, min_jj,
                                      &b[(size_t)ls + (size_t)jjs * ldb], ldb,
                                      Bp + (size_t)min_l * (jjs - js));
                    etrsm_solve_kernel(/*left=*/1, kt,
                                       min_i, min_jj, min_l,
                                       Ap, Bp + (size_t)min_l * (jjs - js),
                                       &b[(size_t)ls + (size_t)jjs * ldb], ldb,
                                       /*offset=*/0);
                }

                /* is loop: more A rows from the same KC-band (still
                 * containing the diagonal entries below it). */
                for (ptrdiff_t is = ls + min_i; is < ls + min_l; is += MC) {
                    min_i = ls + min_l - is;
                    if (min_i > MC) min_i = MC;
                    pack_trsm_a_lside_forward(upper, trans, unit,
                                              min_l, min_i,
                                              !trans ? &a[(size_t)is + (size_t)ls * lda]
                                                     : &a[(size_t)ls + (size_t)is * lda],
                                              lda,
                                              /*offset=*/is - ls, Ap);
                    etrsm_solve_kernel(/*left=*/1, kt,
                                       min_i, min_j, min_l,
                                       Ap, Bp,
                                       &b[(size_t)is + (size_t)js * ldb], ldb,
                                       /*offset=*/is - ls);
                }

                /* is loop: rows entirely below the diagonal — pure GEMM. */
                for (ptrdiff_t is = ls + min_l; is < m; is += MC) {
                    min_i = m - is;
                    if (min_i > MC) min_i = MC;
                    if (!trans) {
                        etri_tcopy(min_l, min_i,
                                          &a[(size_t)is + (size_t)ls * lda], lda, Ap);
                    } else {
                        etri_ncopy(min_l, min_i,
                                          &a[(size_t)ls + (size_t)is * lda], lda, Ap);
                    }
                    etri_gemm_kernel(min_i, min_j, min_l, dm1,
                                       Ap, Bp,
                                       &b[(size_t)is + (size_t)js * ldb], ldb);
                }
            }
        } else {
            /* trsm_L.c lines 184-248 (backward ls walk: ls from m down). */
            for (ptrdiff_t ls = m; ls > 0; ls -= KC) {
                ptrdiff_t min_l = ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t start_is = ls - min_l;
                while (start_is + MC < ls) start_is += MC;
                ptrdiff_t min_i = ls - start_is;
                if (min_i > MC) min_i = MC;

                /* Pack diagonal block — note OpenBLAS passes
                 *   a + (start_is + (ls - min_l)*lda)   [for !TRANS / IUTCOPY]
                 *   a + ((ls - min_l) + start_is*lda)   [for TRANS / ILNCOPY]
                 * with offset = start_is - (ls - min_l). */
                pack_trsm_a_lside_backward(upper, trans, unit,
                                           min_l, min_i,
                                           !trans
                                             ? &a[(size_t)start_is + (size_t)(ls - min_l) * lda]
                                             : &a[(size_t)(ls - min_l) + (size_t)start_is * lda],
                                           lda,
                                           /*offset=*/start_is - (ls - min_l), Ap);

                for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                    ptrdiff_t min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    etri_ncopy(min_l, min_jj,
                                      &b[(size_t)(ls - min_l) + (size_t)jjs * ldb], ldb,
                                      Bp + (size_t)min_l * (jjs - js));
                    etrsm_solve_kernel(/*left=*/1, kt,
                                       min_i, min_jj, min_l,
                                       Ap, Bp + (size_t)min_l * (jjs - js),
                                       &b[(size_t)start_is + (size_t)jjs * ldb], ldb,
                                       /*offset=*/start_is - ls + min_l);
                }

                /* is loop: rows above start_is, within [ls-min_l, ls). */
                for (ptrdiff_t is = start_is - MC; is >= ls - min_l; is -= MC) {
                    min_i = ls - is;
                    if (min_i > MC) min_i = MC;
                    pack_trsm_a_lside_backward(upper, trans, unit,
                                               min_l, min_i,
                                               !trans
                                                 ? &a[(size_t)is + (size_t)(ls - min_l) * lda]
                                                 : &a[(size_t)(ls - min_l) + (size_t)is * lda],
                                               lda,
                                               /*offset=*/is - (ls - min_l), Ap);
                    etrsm_solve_kernel(/*left=*/1, kt,
                                       min_i, min_j, min_l,
                                       Ap, Bp,
                                       &b[(size_t)is + (size_t)js * ldb], ldb,
                                       /*offset=*/is - (ls - min_l));
                }

                /* is loop: rows entirely above the diagonal band. */
                for (ptrdiff_t is = 0; is < ls - min_l; is += MC) {
                    min_i = ls - min_l - is;
                    if (min_i > MC) min_i = MC;
                    if (!trans) {
                        etri_tcopy(min_l, min_i,
                                          &a[(size_t)is + (size_t)(ls - min_l) * lda], lda, Ap);
                    } else {
                        etri_ncopy(min_l, min_i,
                                          &a[(size_t)(ls - min_l) + (size_t)is * lda], lda, Ap);
                    }
                    etri_gemm_kernel(min_i, min_j, min_l, dm1,
                                       Ap, Bp,
                                       &b[(size_t)is + (size_t)js * ldb], ldb);
                }
            }
        }
    }
}

/* ── SIDE='R' driver: port of trsm_R.c for one M-band [m_lo..m_hi) ──── */
void etrsm_R_band(bool upper, bool trans, bool unit,
                        ptrdiff_t N, ptrdiff_t m_lo, ptrdiff_t m_hi,
                        ptrdiff_t MC, ptrdiff_t KC, ptrdiff_t NC,
                        const T *a, ptrdiff_t lda,
                        T *b, ptrdiff_t ldb,
                        T *Ap, T *Bp)
{
    const T dm1 = -1.0L;
    const ptrdiff_t m_band = m_hi - m_lo;
    if (m_band <= 0) return;
    /* SIDE=R forward direction = UPPER+!TRANS || !UPPER+TRANS (js walks
     * up; for each js, ls walks 0..js then js..js+min_j). */
    const ptrdiff_t forward = (upper && !trans) || (!upper && trans);
    /* TRSM_KERNEL trans flag (left=0): forward branch → RN (kt=0);
     * backward → RT (kt=1). See trsm_R.c #define logic. */
    const ptrdiff_t kt = forward ? 0 : 1;

    /* sa = B-tile pack (Ap); sb = A pack (Bp) — matches trsm_R.c naming. */
    T *sa = Ap;
    T *sb = Bp;

    if (forward) {
        /* trsm_R.c lines 115-229 (forward js walk). */
        for (ptrdiff_t js = 0; js < N; js += NC) {
            ptrdiff_t min_j = N - js;
            if (min_j > NC) min_j = NC;

            /* ls loop part 1: A-cols [ls, ls+min_l) entirely above the
             * diagonal of B's js-band — pure GEMM. */
            for (ptrdiff_t ls = 0; ls < js; ls += KC) {
                ptrdiff_t min_l = js - ls;
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
                    etri_gemm_kernel(min_i, min_jj, min_l, dm1,
                                       sa, sb + (size_t)min_l * (jjs - js),
                                       &b[(size_t)m_lo + (size_t)jjs * ldb], ldb);
                }

                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    etri_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);
                    etri_gemm_kernel(min_i, min_j, min_l, dm1,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) + (size_t)js * ldb], ldb);
                }
            }

            /* ls loop part 2: A-cols [ls, ls+min_l) intersecting the
             * diagonal — TRSM packers + TRSM_KERNEL. */
            for (ptrdiff_t ls = js; ls < js + min_j; ls += KC) {
                ptrdiff_t min_l = js + min_j - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = m_band;
                if (min_i > MC) min_i = MC;

                etri_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                /* TRSM_O*COPY of A diagonal block. */
                pack_trsm_a_rside_forward(upper, trans, unit,
                                          min_l, min_l,
                                          &a[(size_t)ls + (size_t)ls * lda], lda,
                                          /*offset=*/0, sb);

                etrsm_solve_kernel(/*left=*/0, kt,
                                   min_i, min_l, min_l,
                                   sa, sb,
                                   &b[(size_t)m_lo + (size_t)ls * ldb], ldb,
                                   /*offset=*/0);

                /* Off-diagonal A pack to the right of the diagonal block. */
                for (ptrdiff_t jjs = 0; jjs < min_j - min_l - ls + js; jjs += NR) {
                    ptrdiff_t min_jj = min_j - min_l - ls + js - jjs;
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
                    etri_gemm_kernel(min_i, min_jj, min_l, dm1,
                                       sa, sb + (size_t)min_l * (min_l + jjs),
                                       &b[(size_t)m_lo + (size_t)(min_l + ls + jjs) * ldb], ldb);
                }

                /* is loop: more M-rows of B with same A-packs. */
                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    etri_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);
                    etrsm_solve_kernel(/*left=*/0, kt,
                                       min_i, min_l, min_l,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb,
                                       /*offset=*/0);
                    if (min_j - min_l + js - ls > 0) {
                        etri_gemm_kernel(min_i, min_j - min_l + js - ls, min_l, dm1,
                                           sa, sb + (size_t)min_l * min_l,
                                           &b[(size_t)(m_lo + is) + (size_t)(min_l + ls) * ldb], ldb);
                    }
                }
            }
        }
    } else {
        /* trsm_R.c lines 232-352 (backward js walk: js from N down). */
        for (ptrdiff_t js = N; js > 0; js -= NC) {
            ptrdiff_t min_j = js;
            if (min_j > NC) min_j = NC;

            /* ls loop part 1: A-cols [ls, ls+min_l) entirely below the
             * diagonal of B's js-band — pure GEMM. */
            for (ptrdiff_t ls = js; ls < N; ls += KC) {
                ptrdiff_t min_l = N - ls;
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
                    etri_gemm_kernel(min_i, min_jj, min_l, dm1,
                                       sa, sb + (size_t)min_l * (jjs - js),
                                       &b[(size_t)m_lo + (size_t)(jjs - min_j) * ldb], ldb);
                }

                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    etri_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);
                    etri_gemm_kernel(min_i, min_j, min_l, dm1,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) + (size_t)(js - min_j) * ldb], ldb);
                }
            }

            /* ls loop part 2: walk down the diagonal band, from
             * start_ls to (js-min_j) in steps of -KC. */
            ptrdiff_t start_ls = js - min_j;
            while (start_ls + KC < js) start_ls += KC;

            for (ptrdiff_t ls = start_ls; ls >= js - min_j; ls -= KC) {
                ptrdiff_t min_l = js - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = m_band;
                if (min_i > MC) min_i = MC;

                etri_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                /* TRSM_O*COPY of A diagonal block.
                 * sb offset = min_l * (min_j - js + ls) — packs diag
                 * block at the tail of sb so the off-diagonal pack to
                 * its LEFT can be sb + 0. */
                pack_trsm_a_rside_backward(upper, trans, unit,
                                           min_l, min_l,
                                           &a[(size_t)ls + (size_t)ls * lda], lda,
                                           /*offset=*/0,
                                           sb + (size_t)min_l * (min_j - js + ls));

                etrsm_solve_kernel(/*left=*/0, kt,
                                   min_i, min_l, min_l,
                                   sa, sb + (size_t)min_l * (min_j - js + ls),
                                   &b[(size_t)m_lo + (size_t)ls * ldb], ldb,
                                   /*offset=*/0);

                /* Off-diagonal A pack to the left of the diagonal
                 * block: A-cols [js-min_j, ls). */
                for (ptrdiff_t jjs = 0; jjs < min_j - js + ls; jjs += NR) {
                    ptrdiff_t min_jj = min_j - js + ls - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        etri_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)(js - min_j + jjs) * lda], lda,
                                          sb + (size_t)min_l * jjs);
                    } else {
                        etri_tcopy(min_l, min_jj,
                                          &a[(size_t)(js - min_j + jjs) + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * jjs);
                    }
                    etri_gemm_kernel(min_i, min_jj, min_l, dm1,
                                       sa, sb + (size_t)min_l * jjs,
                                       &b[(size_t)m_lo + (size_t)(js - min_j + jjs) * ldb], ldb);
                }

                /* is loop: more M-rows of B with same A-packs. */
                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    etri_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);
                    etrsm_solve_kernel(/*left=*/0, kt,
                                       min_i, min_l, min_l,
                                       sa, sb + (size_t)min_l * (min_j - js + ls),
                                       &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb,
                                       /*offset=*/0);
                    if (min_j - js + ls > 0) {
                        etri_gemm_kernel(min_i, min_j - js + ls, min_l, dm1,
                                           sa, sb,
                                           &b[(size_t)(m_lo + is) + (size_t)(js - min_j) * ldb], ldb);
                    }
                }
            }
        }
    }
}


/* ── Pure-serial Fortran-ABI entry ───────────────────────────────────
 *
 * Runs the full SIDE/UPLO/TRANSA dispatch over one band spanning the
 * whole free axis (no threading). etrsm_parallel.c's etrsm_ delegates
 * here when called from inside another routine's parallel region.
 */
void etrsm_serial(
    char side, char uplo, char transa, char diag,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
    const T alpha = *alpha_;

    const bool lside = (blas_up(side)   == 'L');
    const bool upper = (blas_up(uplo)   == 'U');
    const char TR  = blas_up(transa);
    const bool trans = (TR == 'T' || TR == 'C');   /* real: 'C' ≡ 'T' */
    const bool unit  = (blas_up(diag) == 'U');

    if (M == 0 || N == 0) return;

    /* α pre-scale of B (mirrors trsm_{L,R}.c GEMM_BETA pass). */
    if (alpha != 1.0L) egemm_beta_prepass(M, N, alpha, b, ldb);
    if (alpha == 0.0L) return;

    const ptrdiff_t K_eff = lside ? M : N;
    ptrdiff_t MC, KC, NC;
    egemm_choose_blocks(K_eff, &MC, &KC, &NC);

    const size_t ap_bytes = (size_t)egemm_round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)egemm_round_up(NC, NR) * sizeof(T);
    T *Ap = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
    T *Bp = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
    if (Ap && Bp) {
        if (lside) etrsm_L_band(upper, trans, unit, M, 0, N,
                                MC, KC, NC, a, lda, b, ldb, Ap, Bp);
        else       etrsm_R_band(upper, trans, unit, N, 0, M,
                                MC, KC, NC, a, lda, b, ldb, Ap, Bp);
    }
    free(Ap);
    free(Bp);
}
