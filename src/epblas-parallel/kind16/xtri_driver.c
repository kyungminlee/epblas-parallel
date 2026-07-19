/*
 * xtri_driver — kind16 (COMPLEX(KIND=16)) blocked/packed TRSM driver.
 *
 * Faithful parallel-overlay port of the OpenBLAS quad-complex ZTRSM driver
 * (src/epblas-openblas/kind16/xtrsm.c), the complex-quad twin of the real
 * qtri path. Reuses par's packed complex substrate (qblas_xgemm_* +
 * qblas_xtrsm_* in xl3_complex.{c,h}). Interleaved-complex storage: every
 * element = 2 __float128 (re, im); lda/ldb are in complex elements.
 *
 *   op(A) * X = alpha * B    (SIDE='L')   →  B := alpha * inv(op(A)) * B_old
 *   X * op(A) = alpha * B    (SIDE='R')   →  B := alpha * B_old * inv(op(A))
 *
 * Only the L-side Trans/ConjTrans cells route here from xtrsm_core (where
 * OpenBLAS's blocked packed GEMM otherwise pulls ~2-4% ahead of par's naive
 * dot cores); every other cell keeps the naive cores, where par already wins.
 * Conjugation ('C' on transa) is absorbed at pack time so the kernel runs
 * only the NN form of the complex multiply.
 */

#include "xl3_complex.h"
#include <quadmath.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __float128 T;

#define MR QBLAS_XGEMM_MR
#define NR QBLAS_XGEMM_NR

static ptrdiff_t round_up(ptrdiff_t v, ptrdiff_t m) { return ((v + m - 1) / m) * m; }


/* ── Complex TRSM packer dispatch (mirrors qtrsm.c's real twin). */
static inline void pack_trsm_a_lside_forward(bool trans, bool unit, bool conj,
                                             ptrdiff_t m, ptrdiff_t n,
                                             const T *a, ptrdiff_t lda,
                                             ptrdiff_t offset, T *bp)
{
    if (!trans) {
        qblas_xtrsm_iltcopy(m, n, a, lda, offset, bp, unit, conj);
    } else {
        qblas_xtrsm_iuncopy(m, n, a, lda, offset, bp, unit, conj);
    }
}

static inline void pack_trsm_a_lside_backward(bool trans, bool unit, bool conj,
                                              ptrdiff_t m, ptrdiff_t n,
                                              const T *a, ptrdiff_t lda,
                                              ptrdiff_t offset, T *bp)
{
    if (!trans) {
        qblas_xtrsm_iutcopy(m, n, a, lda, offset, bp, unit, conj);
    } else {
        qblas_xtrsm_ilncopy(m, n, a, lda, offset, bp, unit, conj);
    }
}

static inline void pack_trsm_a_rside_forward(bool trans, bool unit, bool conj,
                                             ptrdiff_t m, ptrdiff_t n,
                                             const T *a, ptrdiff_t lda,
                                             ptrdiff_t offset, T *bp)
{
    if (!trans) {
        qblas_xtrsm_iuncopy(m, n, a, lda, offset, bp, unit, conj);
    } else {
        qblas_xtrsm_iltcopy(m, n, a, lda, offset, bp, unit, conj);
    }
}

static inline void pack_trsm_a_rside_backward(bool trans, bool unit, bool conj,
                                              ptrdiff_t m, ptrdiff_t n,
                                              const T *a, ptrdiff_t lda,
                                              ptrdiff_t offset, T *bp)
{
    if (!trans) {
        qblas_xtrsm_ilncopy(m, n, a, lda, offset, bp, unit, conj);
    } else {
        qblas_xtrsm_iutcopy(m, n, a, lda, offset, bp, unit, conj);
    }
}


/* ── SIDE='L' driver — complex twin of trsm_L_band ─────────────────── */
static void trsm_L_band(bool upper, bool trans, bool unit, bool conj,
                        ptrdiff_t M, ptrdiff_t js0, ptrdiff_t js1,
                        ptrdiff_t MC, ptrdiff_t KC, ptrdiff_t NC,
                        const T *a, ptrdiff_t lda,
                        T *b, ptrdiff_t ldb,
                        T *Ap, T *Bp)
{
    const T dm1r = -1.0Q, dm1i = 0.0Q;
    ptrdiff_t m = M;
    const bool forward = (!upper && !trans) || (upper && trans);
    const bool kt = forward;

    for (ptrdiff_t js = js0; js < js1; js += NC) {
        ptrdiff_t min_j = js1 - js;
        if (min_j > NC) min_j = NC;

        if (forward) {
            for (ptrdiff_t ls = 0; ls < m; ls += KC) {
                ptrdiff_t min_l = m - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = min_l;
                if (min_i > MC) min_i = MC;

                pack_trsm_a_lside_forward(trans, unit, conj,
                                          min_l, min_i,
                                          &a[(size_t)ls * 2 + (size_t)ls * lda * 2], lda,
                                          0, Ap);

                for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                    ptrdiff_t min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    qblas_xgemm_ncopy(min_l, min_jj, /*conj=*/0,
                                      &b[(size_t)ls * 2 + (size_t)jjs * ldb * 2], ldb,
                                      Bp + (size_t)min_l * (jjs - js) * 2);
                    qblas_xtrsm_kernel(/*left=*/1, kt,
                                       min_i, min_jj, min_l,
                                       Ap, Bp + (size_t)min_l * (jjs - js) * 2,
                                       &b[(size_t)ls * 2 + (size_t)jjs * ldb * 2], ldb,
                                       /*offset=*/0);
                }

                for (ptrdiff_t is = ls + min_i; is < ls + min_l; is += MC) {
                    min_i = ls + min_l - is;
                    if (min_i > MC) min_i = MC;
                    pack_trsm_a_lside_forward(trans, unit, conj,
                                              min_l, min_i,
                                              !trans
                                                ? &a[(size_t)is * 2 + (size_t)ls * lda * 2]
                                                : &a[(size_t)ls * 2 + (size_t)is * lda * 2],
                                              lda,
                                              is - ls, Ap);
                    qblas_xtrsm_kernel(/*left=*/1, kt,
                                       min_i, min_j, min_l,
                                       Ap, Bp,
                                       &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb,
                                       is - ls);
                }

                for (ptrdiff_t is = ls + min_l; is < m; is += MC) {
                    min_i = m - is;
                    if (min_i > MC) min_i = MC;
                    if (!trans) {
                        qblas_xgemm_tcopy(min_l, min_i, conj,
                                          &a[(size_t)is * 2 + (size_t)ls * lda * 2], lda, Ap);
                    } else {
                        qblas_xgemm_ncopy(min_l, min_i, conj,
                                          &a[(size_t)ls * 2 + (size_t)is * lda * 2], lda, Ap);
                    }
                    qblas_xgemm_kernel(min_i, min_j, min_l, dm1r, dm1i,
                                       Ap, Bp,
                                       &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb);
                }
            }
        } else {
            for (ptrdiff_t ls = m; ls > 0; ls -= KC) {
                ptrdiff_t min_l = ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t start_is = ls - min_l;
                while (start_is + MC < ls) start_is += MC;
                ptrdiff_t min_i = ls - start_is;
                if (min_i > MC) min_i = MC;

                pack_trsm_a_lside_backward(trans, unit, conj,
                                           min_l, min_i,
                                           !trans
                                             ? &a[(size_t)start_is * 2 + (size_t)(ls - min_l) * lda * 2]
                                             : &a[(size_t)(ls - min_l) * 2 + (size_t)start_is * lda * 2],
                                           lda,
                                           start_is - (ls - min_l), Ap);

                for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                    ptrdiff_t min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    qblas_xgemm_ncopy(min_l, min_jj, /*conj=*/0,
                                      &b[(size_t)(ls - min_l) * 2 + (size_t)jjs * ldb * 2], ldb,
                                      Bp + (size_t)min_l * (jjs - js) * 2);
                    qblas_xtrsm_kernel(/*left=*/1, kt,
                                       min_i, min_jj, min_l,
                                       Ap, Bp + (size_t)min_l * (jjs - js) * 2,
                                       &b[(size_t)start_is * 2 + (size_t)jjs * ldb * 2], ldb,
                                       start_is - ls + min_l);
                }

                for (ptrdiff_t is = start_is - MC; is >= ls - min_l; is -= MC) {
                    min_i = ls - is;
                    if (min_i > MC) min_i = MC;
                    pack_trsm_a_lside_backward(trans, unit, conj,
                                               min_l, min_i,
                                               !trans
                                                 ? &a[(size_t)is * 2 + (size_t)(ls - min_l) * lda * 2]
                                                 : &a[(size_t)(ls - min_l) * 2 + (size_t)is * lda * 2],
                                               lda,
                                               is - (ls - min_l), Ap);
                    qblas_xtrsm_kernel(/*left=*/1, kt,
                                       min_i, min_j, min_l,
                                       Ap, Bp,
                                       &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb,
                                       is - (ls - min_l));
                }

                for (ptrdiff_t is = 0; is < ls - min_l; is += MC) {
                    min_i = ls - min_l - is;
                    if (min_i > MC) min_i = MC;
                    if (!trans) {
                        qblas_xgemm_tcopy(min_l, min_i, conj,
                                          &a[(size_t)is * 2 + (size_t)(ls - min_l) * lda * 2], lda, Ap);
                    } else {
                        qblas_xgemm_ncopy(min_l, min_i, conj,
                                          &a[(size_t)(ls - min_l) * 2 + (size_t)is * lda * 2], lda, Ap);
                    }
                    qblas_xgemm_kernel(min_i, min_j, min_l, dm1r, dm1i,
                                       Ap, Bp,
                                       &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb);
                }
            }
        }
    }
}


/* ── SIDE='R' driver — complex twin of trsm_R_band ─────────────────── */
static void trsm_R_band(bool upper, bool trans, bool unit, bool conj,
                        ptrdiff_t N, ptrdiff_t m_lo, ptrdiff_t m_hi,
                        ptrdiff_t MC, ptrdiff_t KC, ptrdiff_t NC,
                        const T *a, ptrdiff_t lda,
                        T *b, ptrdiff_t ldb,
                        T *Ap, T *Bp)
{
    const T dm1r = -1.0Q, dm1i = 0.0Q;
    const ptrdiff_t m_band = m_hi - m_lo;
    if (m_band <= 0) return;
    const bool forward = (upper && !trans) || (!upper && trans);
    const bool kt = forward ? 0 : 1;
    T *sa = Ap;
    T *sb = Bp;

    if (forward) {
        for (ptrdiff_t js = 0; js < N; js += NC) {
            ptrdiff_t min_j = N - js;
            if (min_j > NC) min_j = NC;

            for (ptrdiff_t ls = 0; ls < js; ls += KC) {
                ptrdiff_t min_l = js - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = m_band;
                if (min_i > MC) min_i = MC;

                qblas_xgemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                    ptrdiff_t min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        qblas_xgemm_ncopy(min_l, min_jj, conj,
                                          &a[(size_t)ls * 2 + (size_t)jjs * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    } else {
                        qblas_xgemm_tcopy(min_l, min_jj, conj,
                                          &a[(size_t)jjs * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    }
                    qblas_xgemm_kernel(min_i, min_jj, min_l, dm1r, dm1i,
                                       sa, sb + (size_t)min_l * (jjs - js) * 2,
                                       &b[(size_t)m_lo * 2 + (size_t)jjs * ldb * 2], ldb);
                }

                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    qblas_xgemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);
                    qblas_xgemm_kernel(min_i, min_j, min_l, dm1r, dm1i,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) * 2 + (size_t)js * ldb * 2], ldb);
                }
            }

            for (ptrdiff_t ls = js; ls < js + min_j; ls += KC) {
                ptrdiff_t min_l = js + min_j - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = m_band;
                if (min_i > MC) min_i = MC;

                qblas_xgemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                pack_trsm_a_rside_forward(trans, unit, conj,
                                          min_l, min_l,
                                          &a[(size_t)ls * 2 + (size_t)ls * lda * 2], lda,
                                          0, sb);

                qblas_xtrsm_kernel(/*left=*/0, kt,
                                   min_i, min_l, min_l,
                                   sa, sb,
                                   &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb,
                                   /*offset=*/0);

                for (ptrdiff_t jjs = 0; jjs < min_j - min_l - ls + js; jjs += NR) {
                    ptrdiff_t min_jj = min_j - min_l - ls + js - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        qblas_xgemm_ncopy(min_l, min_jj, conj,
                                          &a[(size_t)ls * 2 + (size_t)(ls + min_l + jjs) * lda * 2], lda,
                                          sb + (size_t)min_l * (min_l + jjs) * 2);
                    } else {
                        qblas_xgemm_tcopy(min_l, min_jj, conj,
                                          &a[(size_t)(ls + min_l + jjs) * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * (min_l + jjs) * 2);
                    }
                    qblas_xgemm_kernel(min_i, min_jj, min_l, dm1r, dm1i,
                                       sa, sb + (size_t)min_l * (min_l + jjs) * 2,
                                       &b[(size_t)m_lo * 2 + (size_t)(min_l + ls + jjs) * ldb * 2], ldb);
                }

                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    qblas_xgemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);
                    qblas_xtrsm_kernel(/*left=*/0, kt,
                                       min_i, min_l, min_l,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb,
                                       0);
                    if (min_j - min_l + js - ls > 0) {
                        qblas_xgemm_kernel(min_i, min_j - min_l + js - ls, min_l, dm1r, dm1i,
                                           sa, sb + (size_t)min_l * min_l * 2,
                                           &b[(size_t)(m_lo + is) * 2 + (size_t)(min_l + ls) * ldb * 2], ldb);
                    }
                }
            }
        }
    } else {
        for (ptrdiff_t js = N; js > 0; js -= NC) {
            ptrdiff_t min_j = js;
            if (min_j > NC) min_j = NC;

            for (ptrdiff_t ls = js; ls < N; ls += KC) {
                ptrdiff_t min_l = N - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = m_band;
                if (min_i > MC) min_i = MC;

                qblas_xgemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                for (ptrdiff_t jjs = js; jjs < js + min_j; jjs += NR) {
                    ptrdiff_t min_jj = min_j + js - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        qblas_xgemm_ncopy(min_l, min_jj, conj,
                                          &a[(size_t)ls * 2 + (size_t)(jjs - min_j) * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    } else {
                        qblas_xgemm_tcopy(min_l, min_jj, conj,
                                          &a[(size_t)(jjs - min_j) * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    }
                    qblas_xgemm_kernel(min_i, min_jj, min_l, dm1r, dm1i,
                                       sa, sb + (size_t)min_l * (jjs - js) * 2,
                                       &b[(size_t)m_lo * 2 + (size_t)(jjs - min_j) * ldb * 2], ldb);
                }

                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    qblas_xgemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);
                    qblas_xgemm_kernel(min_i, min_j, min_l, dm1r, dm1i,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) * 2 + (size_t)(js - min_j) * ldb * 2], ldb);
                }
            }

            ptrdiff_t start_ls = js - min_j;
            while (start_ls + KC < js) start_ls += KC;

            for (ptrdiff_t ls = start_ls; ls >= js - min_j; ls -= KC) {
                ptrdiff_t min_l = js - ls;
                if (min_l > KC) min_l = KC;
                ptrdiff_t min_i = m_band;
                if (min_i > MC) min_i = MC;

                qblas_xgemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                pack_trsm_a_rside_backward(trans, unit, conj,
                                           min_l, min_l,
                                           &a[(size_t)ls * 2 + (size_t)ls * lda * 2], lda,
                                           0,
                                           sb + (size_t)min_l * (min_j - js + ls) * 2);

                qblas_xtrsm_kernel(/*left=*/0, kt,
                                   min_i, min_l, min_l,
                                   sa, sb + (size_t)min_l * (min_j - js + ls) * 2,
                                   &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb,
                                   0);

                for (ptrdiff_t jjs = 0; jjs < min_j - js + ls; jjs += NR) {
                    ptrdiff_t min_jj = min_j - js + ls - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        qblas_xgemm_ncopy(min_l, min_jj, conj,
                                          &a[(size_t)ls * 2 + (size_t)(js - min_j + jjs) * lda * 2], lda,
                                          sb + (size_t)min_l * jjs * 2);
                    } else {
                        qblas_xgemm_tcopy(min_l, min_jj, conj,
                                          &a[(size_t)(js - min_j + jjs) * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * jjs * 2);
                    }
                    qblas_xgemm_kernel(min_i, min_jj, min_l, dm1r, dm1i,
                                       sa, sb + (size_t)min_l * jjs * 2,
                                       &b[(size_t)m_lo * 2 + (size_t)(js - min_j + jjs) * ldb * 2], ldb);
                }

                for (ptrdiff_t is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    qblas_xgemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);
                    qblas_xtrsm_kernel(/*left=*/0, kt,
                                       min_i, min_l, min_l,
                                       sa, sb + (size_t)min_l * (min_j - js + ls) * 2,
                                       &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb,
                                       0);
                    if (min_j - js + ls > 0) {
                        qblas_xgemm_kernel(min_i, min_j - js + ls, min_l, dm1r, dm1i,
                                           sa, sb,
                                           &b[(size_t)(m_lo + is) * 2 + (size_t)(js - min_j) * ldb * 2], ldb);
                    }
                }
            }
        }
    }
}


/* ── Internal entry — decoded flags, split alpha, interleaved-complex a/b.
 *
 * `lside/upper/trans/conj/unit` are booleans; a/b are reinterpreted from
 * the caller's `_Complex __float128` buffers (identical layout: 2 __float128
 * per element). lda/ldb are in complex elements. Threads only when not
 * already inside a parallel region (par convention). */
void xtrsm_packed(bool lside, bool upper, bool trans, bool conj, bool unit,
                  ptrdiff_t M, ptrdiff_t N,
                  __float128 alpha_r, __float128 alpha_i,
                  const T *a, ptrdiff_t lda,
                  T *b, ptrdiff_t ldb)
{
    if (M == 0 || N == 0) return;

    /* Pre-scale B by alpha (complex). */
    if (alpha_r != 1.0Q || alpha_i != 0.0Q) {
        qblas_xgemm_beta(M, N, alpha_r, alpha_i, b, ldb);
    }
    if (alpha_r == 0.0Q && alpha_i == 0.0Q) return;

    ptrdiff_t MC0_p, KC_p, NC_p;
    qblas_xgemm_blocks(&MC0_p, &KC_p, &NC_p);
    ptrdiff_t MC0 = MC0_p, KC = KC_p, NC = NC_p;

    ptrdiff_t K_eff = lside ? M : N;
    ptrdiff_t MC = MC0;
    if (K_eff <= KC) {
        const ptrdiff_t L2_TARGET_BYTES = 256L * 1024L;
        ptrdiff_t target_mc = L2_TARGET_BYTES / (K_eff * (ptrdiff_t)sizeof(T) * 2);
        if (target_mc > MC) {
            if (target_mc > 4 * MC0) target_mc = 4 * MC0;
            MC = round_up(target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    /* 2x the size for complex (2 __float128s per element). */
    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * sizeof(T) * 2;
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * sizeof(T) * 2;

#ifdef _OPENMP
    ptrdiff_t nthreads = omp_in_parallel() ? 1 : omp_get_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif

    ptrdiff_t mnk = M * N * K_eff;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

    ptrdiff_t partition_axis = lside ? N : M;
    if (nthreads > partition_axis) nthreads = partition_axis;
    if (nthreads < 1) nthreads = 1;

    /* Per-thread Ap/Bp scratch carved from a persistent grow-only
     * thread-local arena on the calling thread: a per-call
     * aligned_alloc+free of these mmap-threshold-sized buffers trips
     * glibc's trim heuristic and re-faults every touched page each call
     * (see kind10 etrsm_serial.c). Only the small pointer arrays stay
     * per-call. */
    T **Ap_arr = calloc((size_t)nthreads, sizeof(T *));
    T **Bp_arr = calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_arr || !Bp_arr) { free(Ap_arr); free(Bp_arr); return; }
    const size_t ap_al = (ap_bytes + 63) & ~(size_t)63;
    const size_t bp_al = (bp_bytes + 63) & ~(size_t)63;
    static __thread T *g_pack = NULL;
    static __thread size_t g_pack_cap = 0;
    const size_t need = (size_t)nthreads * (ap_al + bp_al);
    if (need > g_pack_cap) {
        free(g_pack);
        size_t cap = need + (need >> 1);            /* 1.5× headroom to amortize regrow */
        cap = (cap + 63) & ~(size_t)63;
        g_pack = aligned_alloc(64, cap);
        g_pack_cap = g_pack ? cap : 0;
    }
    if (!g_pack) { free(Ap_arr); free(Bp_arr); return; }
    for (ptrdiff_t t = 0; t < nthreads; ++t) {
        Ap_arr[t] = (T *)(void *)((char *)g_pack + (size_t)t * (ap_al + bp_al));
        Bp_arr[t] = (T *)(void *)((char *)g_pack + (size_t)t * (ap_al + bp_al) + ap_al);
    }

#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
#else
        ptrdiff_t tid = 0, nth = 1;
#endif
        T *Ap = Ap_arr[tid];
        T *Bp = Bp_arr[tid];

        if (lside) {
            ptrdiff_t chunk = round_up((N + nth - 1) / nth, NR);
            ptrdiff_t js0 = tid * chunk;
            ptrdiff_t js1 = js0 + chunk;
            if (js0 > N) js0 = N;
            if (js1 > N) js1 = N;
            if (js0 < js1) {
                trsm_L_band(upper, trans, unit, conj,
                            M, js0, js1,
                            MC, KC, NC,
                            a, lda, b, ldb,
                            Ap, Bp);
            }
        } else {
            ptrdiff_t chunk = round_up((M + nth - 1) / nth, MR);
            ptrdiff_t m_lo = tid * chunk;
            ptrdiff_t m_hi = m_lo + chunk;
            if (m_lo > M) m_lo = M;
            if (m_hi > M) m_hi = M;
            if (m_lo < m_hi) {
                trsm_R_band(upper, trans, unit, conj,
                            N, m_lo, m_hi,
                            MC, KC, NC,
                            a, lda, b, ldb,
                            Ap, Bp);
            }
        }
    }

    free(Ap_arr);
    free(Bp_arr);
}
