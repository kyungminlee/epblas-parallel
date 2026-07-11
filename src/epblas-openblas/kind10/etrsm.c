/*
 * etrsm — kind10 (REAL(KIND=10) / 80-bit long double) port of OpenBLAS DTRSM.
 *
 *   op(A) * X = alpha * B    (SIDE='L')   →  B := alpha * inv(op(A)) * B_old
 *   X * op(A) = alpha * B    (SIDE='R')   →  B := alpha * B_old * inv(op(A))
 *
 * Faithful L3-blocked port (sibling of etrmm.c). Full OpenBLAS NC×KC×MC
 * three-level blocking, pack-and-conquer with diagonal-aware TRSM kernel
 * + diagonal-inverting TRSM packers + plain GEMM-kernel + GEMM-packer
 * for the "below-the-current-KC-band" rows of A.
 *
 * Port source: OpenBLAS.
 *   - interface/trsm.c                 (TRSM dispatch — args parsing,
 *                                       alpha pre-scale of B = beta-pass,
 *                                       side dispatch via gemm_thread_n/m)
 *   - driver/level3/trsm_L.c           (SIDE='L' driver, 2 ls-walk-direction
 *                                       branches × runtime (uplo,trans)
 *                                       packer choice)
 *   - driver/level3/trsm_R.c           (SIDE='R' driver, similar)
 *   - kernel/generic/trsm_kernel_{LN,LT,RN,RT}.c
 *                                      → eblas_etrsm_kernel (collapsed)
 *   - kernel/generic/trsm_{ut,un,lt,ln}copy_2.c
 *                                      → eblas_etrsm_i{ut,un,lt,ln}copy
 *
 * Algorithm: TRSM is implemented as
 *   1. alpha pre-scale of B in-place (eblas_egemm_beta with beta=alpha).
 *   2. L3 nest runs with kernel-alpha = dm1 = -1 (per OpenBLAS — internal
 *      TRSM_KERNEL solve is `X = inv(A)*X` where X already carries the
 *      alpha factor from step 1; trailing GEMM updates subtract using
 *      dm1 = -1).
 *
 * OpenBLAS dispatch:
 *   - SIDE='L': gemm_thread_n → range_n partitioning (N-axis).
 *   - SIDE='R': gemm_thread_m → range_m partitioning (M-axis).
 * We mirror this: each OMP thread takes a contiguous slice on the
 * partition axis and runs the full L3 nest on its slice (per-thread Ap
 * and Bp scratch — no cross-thread sync).
 *
 * Fortran ABI matches the migrated reference:
 *   subroutine etrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
 *   - character args carry trailing hidden size_t lengths (gfortran)
 *   - REAL(KIND=10) scalars are passed by pointer
 *   - 'C' on transa is treated identically to 'T' for reals
 */

#include "eblas_l3_real.h"
#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef long double T;

#define MR EBLAS_EGEMM_MR
#define NR EBLAS_EGEMM_NR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


/* ── A-side TRSM packer dispatch (real) ────────────────────────────────
 *
 * SIDE=L mappings (trsm_L.c):
 *   (!UPPER && !TRANS):  TRSM_ILTCOPY    (line 127)
 *   (UPPER && TRANS):    TRSM_IUNCOPY    (line 129)
 *   (UPPER && !TRANS):   TRSM_IUTCOPY    (line 193)
 *   (!UPPER && TRANS):   TRSM_ILNCOPY    (line 195)
 *
 * SIDE=R mappings (trsm_R.c):
 *   (UPPER && !TRANS):   TRSM_OUNCOPY    (line 170)  ← reuses iuncopy
 *   (!UPPER && TRANS):   TRSM_OLTCOPY    (line 172)  ← reuses iltcopy
 *   (!UPPER && !TRANS):  TRSM_OLNCOPY    (line 290)  ← reuses ilncopy
 *   (UPPER && TRANS):    TRSM_OUTCOPY    (line 293)  ← reuses iutcopy
 *
 * Same packer source serves both ICOPY (sa-role) and OCOPY (sb-role) —
 * upstream uses the same .c file for both because the data layout is
 * identical; only the call site / argument shape differs. */
static inline void pack_trsm_a_lside_forward(int upper, int trans, int unit,
                                             ptrdiff_t m, ptrdiff_t n,
                                             const T *a, ptrdiff_t lda,
                                             ptrdiff_t offset, T *bp)
{
    /* forward direction (UPPER+!TRANS or !UPPER+TRANS branch is
     * backwards; this is for !UPPER+!TRANS / UPPER+TRANS). */
    if (!trans) {
        eblas_etrsm_iltcopy(m, n, a, lda, offset, bp, unit);
    } else {
        eblas_etrsm_iuncopy(m, n, a, lda, offset, bp, unit);
    }
}

static inline void pack_trsm_a_lside_backward(int upper, int trans, int unit,
                                              ptrdiff_t m, ptrdiff_t n,
                                              const T *a, ptrdiff_t lda,
                                              ptrdiff_t offset, T *bp)
{
    /* backward direction: UPPER+!TRANS uses IUTCOPY, !UPPER+TRANS uses ILNCOPY */
    if (!trans) {
        eblas_etrsm_iutcopy(m, n, a, lda, offset, bp, unit);
    } else {
        eblas_etrsm_ilncopy(m, n, a, lda, offset, bp, unit);
    }
}

static inline void pack_trsm_a_rside_forward(int upper, int trans, int unit,
                                             ptrdiff_t m, ptrdiff_t n,
                                             const T *a, ptrdiff_t lda,
                                             ptrdiff_t offset, T *bp)
{
    /* trsm_R.c lines 170/172: UPPER+!TRANS → OUNCOPY = iuncopy;
     *                          !UPPER+TRANS → OLTCOPY = iltcopy. */
    if (!trans) {
        eblas_etrsm_iuncopy(m, n, a, lda, offset, bp, unit);
    } else {
        eblas_etrsm_iltcopy(m, n, a, lda, offset, bp, unit);
    }
}

static inline void pack_trsm_a_rside_backward(int upper, int trans, int unit,
                                              ptrdiff_t m, ptrdiff_t n,
                                              const T *a, ptrdiff_t lda,
                                              ptrdiff_t offset, T *bp)
{
    /* trsm_R.c lines 290/293: !UPPER+!TRANS → OLNCOPY = ilncopy;
     *                          UPPER+TRANS → OUTCOPY = iutcopy. */
    if (!trans) {
        eblas_etrsm_ilncopy(m, n, a, lda, offset, bp, unit);
    } else {
        eblas_etrsm_iutcopy(m, n, a, lda, offset, bp, unit);
    }
}


/* ── SIDE='L' driver: port of trsm_L.c for one N-band [js0..js1) ───── */
static void trsm_L_band(int upper, int trans, int unit,
                        int M, int js0, int js1,
                        int MC, int KC, int NC,
                        const T *a, int lda,
                        T *b, int ldb,
                        T *Ap, T *Bp)
{
    const T dm1 = -1.0L;
    int m = M;
    /* Pick which (uplo, trans) branch (forward vs backward ls). Forward
     * = !UPPER+!TRANS || UPPER+TRANS (ls walks 0..m). */
    const int forward = (!upper && !trans) || (upper && trans);
    /* TRSM_KERNEL trans flag: forward branch → LT (kt=1); backward → LN (kt=0).
     * See trsm_L.c #define logic at top of file. */
    const int kt = forward ? 1 : 0;

    for (int js = js0; js < js1; js += NC) {
        int min_j = js1 - js;
        if (min_j > NC) min_j = NC;

        if (forward) {
            /* trsm_L.c lines 119-182 (forward ls walk). */
            for (int ls = 0; ls < m; ls += KC) {
                int min_l = m - ls;
                if (min_l > KC) min_l = KC;
                int min_i = min_l;
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
                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    eblas_egemm_ncopy(min_l, min_jj,
                                      &b[(size_t)ls + (size_t)jjs * ldb], ldb,
                                      Bp + (size_t)min_l * (jjs - js));
                    eblas_etrsm_kernel(/*left=*/1, kt,
                                       min_i, min_jj, min_l,
                                       Ap, Bp + (size_t)min_l * (jjs - js),
                                       &b[(size_t)ls + (size_t)jjs * ldb], ldb,
                                       /*offset=*/0);
                }

                /* is loop: more A rows from the same KC-band (still
                 * containing the diagonal entries below it). */
                for (int is = ls + min_i; is < ls + min_l; is += MC) {
                    min_i = ls + min_l - is;
                    if (min_i > MC) min_i = MC;
                    pack_trsm_a_lside_forward(upper, trans, unit,
                                              min_l, min_i,
                                              !trans ? &a[(size_t)is + (size_t)ls * lda]
                                                     : &a[(size_t)ls + (size_t)is * lda],
                                              lda,
                                              /*offset=*/is - ls, Ap);
                    eblas_etrsm_kernel(/*left=*/1, kt,
                                       min_i, min_j, min_l,
                                       Ap, Bp,
                                       &b[(size_t)is + (size_t)js * ldb], ldb,
                                       /*offset=*/is - ls);
                }

                /* is loop: rows entirely below the diagonal — pure GEMM. */
                for (int is = ls + min_l; is < m; is += MC) {
                    min_i = m - is;
                    if (min_i > MC) min_i = MC;
                    if (!trans) {
                        eblas_egemm_tcopy(min_l, min_i,
                                          &a[(size_t)is + (size_t)ls * lda], lda, Ap);
                    } else {
                        eblas_egemm_ncopy(min_l, min_i,
                                          &a[(size_t)ls + (size_t)is * lda], lda, Ap);
                    }
                    eblas_egemm_kernel(min_i, min_j, min_l, dm1,
                                       Ap, Bp,
                                       &b[(size_t)is + (size_t)js * ldb], ldb);
                }
            }
        } else {
            /* trsm_L.c lines 184-248 (backward ls walk: ls from m down). */
            for (int ls = m; ls > 0; ls -= KC) {
                int min_l = ls;
                if (min_l > KC) min_l = KC;
                int start_is = ls - min_l;
                while (start_is + MC < ls) start_is += MC;
                int min_i = ls - start_is;
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

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    eblas_egemm_ncopy(min_l, min_jj,
                                      &b[(size_t)(ls - min_l) + (size_t)jjs * ldb], ldb,
                                      Bp + (size_t)min_l * (jjs - js));
                    eblas_etrsm_kernel(/*left=*/1, kt,
                                       min_i, min_jj, min_l,
                                       Ap, Bp + (size_t)min_l * (jjs - js),
                                       &b[(size_t)start_is + (size_t)jjs * ldb], ldb,
                                       /*offset=*/start_is - ls + min_l);
                }

                /* is loop: rows above start_is, within [ls-min_l, ls). */
                for (int is = start_is - MC; is >= ls - min_l; is -= MC) {
                    min_i = ls - is;
                    if (min_i > MC) min_i = MC;
                    pack_trsm_a_lside_backward(upper, trans, unit,
                                               min_l, min_i,
                                               !trans
                                                 ? &a[(size_t)is + (size_t)(ls - min_l) * lda]
                                                 : &a[(size_t)(ls - min_l) + (size_t)is * lda],
                                               lda,
                                               /*offset=*/is - (ls - min_l), Ap);
                    eblas_etrsm_kernel(/*left=*/1, kt,
                                       min_i, min_j, min_l,
                                       Ap, Bp,
                                       &b[(size_t)is + (size_t)js * ldb], ldb,
                                       /*offset=*/is - (ls - min_l));
                }

                /* is loop: rows entirely above the diagonal band. */
                for (int is = 0; is < ls - min_l; is += MC) {
                    min_i = ls - min_l - is;
                    if (min_i > MC) min_i = MC;
                    if (!trans) {
                        eblas_egemm_tcopy(min_l, min_i,
                                          &a[(size_t)is + (size_t)(ls - min_l) * lda], lda, Ap);
                    } else {
                        eblas_egemm_ncopy(min_l, min_i,
                                          &a[(size_t)(ls - min_l) + (size_t)is * lda], lda, Ap);
                    }
                    eblas_egemm_kernel(min_i, min_j, min_l, dm1,
                                       Ap, Bp,
                                       &b[(size_t)is + (size_t)js * ldb], ldb);
                }
            }
        }
    }
}


/* ── SIDE='R' driver: port of trsm_R.c for one M-band [m_lo..m_hi) ──── */
static void trsm_R_band(int upper, int trans, int unit,
                        int N, int m_lo, int m_hi,
                        int MC, int KC, int NC,
                        const T *a, int lda,
                        T *b, int ldb,
                        T *Ap, T *Bp)
{
    const T dm1 = -1.0L;
    const int m_band = m_hi - m_lo;
    if (m_band <= 0) return;
    /* SIDE=R forward direction = UPPER+!TRANS || !UPPER+TRANS (js walks
     * up; for each js, ls walks 0..js then js..js+min_j). */
    const int forward = (upper && !trans) || (!upper && trans);
    /* TRSM_KERNEL trans flag (left=0): forward branch → RN (kt=0);
     * backward → RT (kt=1). See trsm_R.c #define logic. */
    const int kt = forward ? 0 : 1;

    /* sa = B-tile pack (Ap); sb = A pack (Bp) — matches trsm_R.c naming. */
    T *sa = Ap;
    T *sb = Bp;

    if (forward) {
        /* trsm_R.c lines 115-229 (forward js walk). */
        for (int js = 0; js < N; js += NC) {
            int min_j = N - js;
            if (min_j > NC) min_j = NC;

            /* ls loop part 1: A-cols [ls, ls+min_l) entirely above the
             * diagonal of B's js-band — pure GEMM. */
            for (int ls = 0; ls < js; ls += KC) {
                int min_l = js - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                eblas_egemm_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        eblas_egemm_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)jjs * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    } else {
                        eblas_egemm_tcopy(min_l, min_jj,
                                          &a[(size_t)jjs + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    }
                    eblas_egemm_kernel(min_i, min_jj, min_l, dm1,
                                       sa, sb + (size_t)min_l * (jjs - js),
                                       &b[(size_t)m_lo + (size_t)jjs * ldb], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    eblas_egemm_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);
                    eblas_egemm_kernel(min_i, min_j, min_l, dm1,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) + (size_t)js * ldb], ldb);
                }
            }

            /* ls loop part 2: A-cols [ls, ls+min_l) intersecting the
             * diagonal — TRSM packers + TRSM_KERNEL. */
            for (int ls = js; ls < js + min_j; ls += KC) {
                int min_l = js + min_j - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                eblas_egemm_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                /* TRSM_O*COPY of A diagonal block. */
                pack_trsm_a_rside_forward(upper, trans, unit,
                                          min_l, min_l,
                                          &a[(size_t)ls + (size_t)ls * lda], lda,
                                          /*offset=*/0, sb);

                eblas_etrsm_kernel(/*left=*/0, kt,
                                   min_i, min_l, min_l,
                                   sa, sb,
                                   &b[(size_t)m_lo + (size_t)ls * ldb], ldb,
                                   /*offset=*/0);

                /* Off-diagonal A pack to the right of the diagonal block. */
                for (int jjs = 0; jjs < min_j - min_l - ls + js; jjs += NR) {
                    int min_jj = min_j - min_l - ls + js - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        eblas_egemm_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)(ls + min_l + jjs) * lda], lda,
                                          sb + (size_t)min_l * (min_l + jjs));
                    } else {
                        eblas_egemm_tcopy(min_l, min_jj,
                                          &a[(size_t)(ls + min_l + jjs) + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * (min_l + jjs));
                    }
                    eblas_egemm_kernel(min_i, min_jj, min_l, dm1,
                                       sa, sb + (size_t)min_l * (min_l + jjs),
                                       &b[(size_t)m_lo + (size_t)(min_l + ls + jjs) * ldb], ldb);
                }

                /* is loop: more M-rows of B with same A-packs. */
                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    eblas_egemm_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);
                    eblas_etrsm_kernel(/*left=*/0, kt,
                                       min_i, min_l, min_l,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb,
                                       /*offset=*/0);
                    if (min_j - min_l + js - ls > 0) {
                        eblas_egemm_kernel(min_i, min_j - min_l + js - ls, min_l, dm1,
                                           sa, sb + (size_t)min_l * min_l,
                                           &b[(size_t)(m_lo + is) + (size_t)(min_l + ls) * ldb], ldb);
                    }
                }
            }
        }
    } else {
        /* trsm_R.c lines 232-352 (backward js walk: js from N down). */
        for (int js = N; js > 0; js -= NC) {
            int min_j = js;
            if (min_j > NC) min_j = NC;

            /* ls loop part 1: A-cols [ls, ls+min_l) entirely below the
             * diagonal of B's js-band — pure GEMM. */
            for (int ls = js; ls < N; ls += KC) {
                int min_l = N - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                eblas_egemm_tcopy(min_l, min_i,
                                  &b[(size_t)m_lo + (size_t)ls * ldb], ldb, sa);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = min_j + js - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        eblas_egemm_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)(jjs - min_j) * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    } else {
                        eblas_egemm_tcopy(min_l, min_jj,
                                          &a[(size_t)(jjs - min_j) + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * (jjs - js));
                    }
                    eblas_egemm_kernel(min_i, min_jj, min_l, dm1,
                                       sa, sb + (size_t)min_l * (jjs - js),
                                       &b[(size_t)m_lo + (size_t)(jjs - min_j) * ldb], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    eblas_egemm_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);
                    eblas_egemm_kernel(min_i, min_j, min_l, dm1,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) + (size_t)(js - min_j) * ldb], ldb);
                }
            }

            /* ls loop part 2: walk down the diagonal band, from
             * start_ls to (js-min_j) in steps of -KC. */
            int start_ls = js - min_j;
            while (start_ls + KC < js) start_ls += KC;

            for (int ls = start_ls; ls >= js - min_j; ls -= KC) {
                int min_l = js - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                eblas_egemm_tcopy(min_l, min_i,
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

                eblas_etrsm_kernel(/*left=*/0, kt,
                                   min_i, min_l, min_l,
                                   sa, sb + (size_t)min_l * (min_j - js + ls),
                                   &b[(size_t)m_lo + (size_t)ls * ldb], ldb,
                                   /*offset=*/0);

                /* Off-diagonal A pack to the left of the diagonal
                 * block: A-cols [js-min_j, ls). */
                for (int jjs = 0; jjs < min_j - js + ls; jjs += NR) {
                    int min_jj = min_j - js + ls - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        eblas_egemm_ncopy(min_l, min_jj,
                                          &a[(size_t)ls + (size_t)(js - min_j + jjs) * lda], lda,
                                          sb + (size_t)min_l * jjs);
                    } else {
                        eblas_egemm_tcopy(min_l, min_jj,
                                          &a[(size_t)(js - min_j + jjs) + (size_t)ls * lda], lda,
                                          sb + (size_t)min_l * jjs);
                    }
                    eblas_egemm_kernel(min_i, min_jj, min_l, dm1,
                                       sa, sb + (size_t)min_l * jjs,
                                       &b[(size_t)m_lo + (size_t)(js - min_j + jjs) * ldb], ldb);
                }

                /* is loop: more M-rows of B with same A-packs. */
                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    eblas_egemm_tcopy(min_l, min_i,
                                      &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb, sa);
                    eblas_etrsm_kernel(/*left=*/0, kt,
                                       min_i, min_l, min_l,
                                       sa, sb + (size_t)min_l * (min_j - js + ls),
                                       &b[(size_t)(m_lo + is) + (size_t)ls * ldb], ldb,
                                       /*offset=*/0);
                    if (min_j - js + ls > 0) {
                        eblas_egemm_kernel(min_i, min_j - js + ls, min_l, dm1,
                                           sa, sb,
                                           &b[(size_t)(m_lo + is) + (size_t)(js - min_j) * ldb], ldb);
                    }
                }
            }
        }
    }
}


/* ── Public entry ──────────────────────────────────────────────────── */
void etrsm_(
    const char *side_p, const char *uplo_p,
    const char *transa_p, const char *diag_p,
    const int *m_, const int *n_, const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_)
{

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

    /* alpha pre-scale of B (mirrors trsm_L.c/trsm_R.c GEMM_BETA pass —
     * args.beta is set to alpha by interface/trsm.c). */
    if (alpha != 1.0L) {
        eblas_egemm_beta((ptrdiff_t)M, (ptrdiff_t)N, alpha, b, (ptrdiff_t)ldb);
    }
    if (alpha == 0.0L) return;

    int MC0, KC, NC;
    eblas_egemm_blocks(&MC0, &KC, &NC);

    /* Adaptive MC for small K (mirrors egemm/etrmm). For SIDE=L K = M;
     * for SIDE=R K = N. */
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

    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * sizeof(T);

#ifdef _OPENMP
    int nthreads = omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
#endif

    long mnk = (long)M * (long)N * (long)K_eff;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

    int partition_axis = lside ? N : M;
    if (nthreads > partition_axis) nthreads = partition_axis;
    if (nthreads < 1) nthreads = 1;

    T **Ap_arr = calloc((size_t)nthreads, sizeof(T *));
    T **Bp_arr = calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_arr || !Bp_arr) { free(Ap_arr); free(Bp_arr); return; }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        Bp_arr[t] = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
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
            int chunk = round_up((N + nth - 1) / nth, NR);
            int js0 = tid * chunk;
            int js1 = js0 + chunk;
            if (js0 > N) js0 = N;
            if (js1 > N) js1 = N;
            if (js0 < js1) {
                trsm_L_band(upper, trans, unit,
                            M, js0, js1,
                            MC, KC, NC,
                            a, lda, b, ldb,
                            Ap, Bp);
            }
        } else {
            int chunk = round_up((M + nth - 1) / nth, MR);
            int m_lo = tid * chunk;
            int m_hi = m_lo + chunk;
            if (m_lo > M) m_lo = M;
            if (m_hi > M) m_hi = M;
            if (m_lo < m_hi) {
                trsm_R_band(upper, trans, unit,
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
