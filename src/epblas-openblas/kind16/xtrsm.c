/*
 * xtrsm — kind16 (COMPLEX(KIND=16)) port of OpenBLAS ZTRSM.
 *
 *   op(A) * X = alpha * B    (SIDE='L')   →  B := alpha * inv(op(A)) * B_old
 *   X * op(A) = alpha * B    (SIDE='R')   →  B := alpha * B_old * inv(op(A))
 *
 * Faithful L3-blocked port (complex twin of qtrsm.c).
 *
 * Port source: OpenBLAS.
 *   - interface/trsm.c                (Z-variant dispatch)
 *   - driver/level3/trsm_L.c          (SIDE='L' driver)
 *   - driver/level3/trsm_R.c          (SIDE='R' driver)
 *   - kernel/generic/trsm_kernel_{LN,LT,RN,RT}.c (Z-variant of solve)
 *                                     → qblas_ytrsm_kernel (NN-only path;
 *                                        conjugation absorbed by packers)
 *   - kernel/generic/ztrsm_{ut,un,lt,ln}copy_2.c
 *                                     → qblas_ytrsm_i{ut,un,lt,ln}copy
 *
 * See qtrsm.c for the per-routine commentary — same algorithm modulo
 * per-element width (= 2 __float128s for complex). Conjugation: 'C'
 * on transa absorbs the imag-sign flip at pack time so the kernel runs
 * only the NN form of the complex multiply (matches xgemm/xtrmm).
 *
 * Fortran ABI:
 *   subroutine xtrsm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
 *   - alpha is COMPLEX(KIND=16) (2 __float128s re,im)
 *   - lda, ldb in COMPLEX(KIND=16) elements
 */

#include "qblas_l3_complex.h"
#include "qblas_tuning.h"
#include <quadmath.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef __float128 T;

#define MR QBLAS_YGEMM_MR
#define NR QBLAS_YGEMM_NR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


/* ── Complex TRSM packer dispatch (mirrors qtrsm.c's real twin). */
static inline void pack_trsm_a_lside_forward(int upper, int trans, int unit, int conj,
                                             ptrdiff_t m, ptrdiff_t n,
                                             const T *a, ptrdiff_t lda,
                                             ptrdiff_t offset, T *bp)
{
    if (!trans) {
        qblas_ytrsm_iltcopy(m, n, a, lda, offset, bp, unit, conj);
    } else {
        qblas_ytrsm_iuncopy(m, n, a, lda, offset, bp, unit, conj);
    }
}

static inline void pack_trsm_a_lside_backward(int upper, int trans, int unit, int conj,
                                              ptrdiff_t m, ptrdiff_t n,
                                              const T *a, ptrdiff_t lda,
                                              ptrdiff_t offset, T *bp)
{
    if (!trans) {
        qblas_ytrsm_iutcopy(m, n, a, lda, offset, bp, unit, conj);
    } else {
        qblas_ytrsm_ilncopy(m, n, a, lda, offset, bp, unit, conj);
    }
}

static inline void pack_trsm_a_rside_forward(int upper, int trans, int unit, int conj,
                                             ptrdiff_t m, ptrdiff_t n,
                                             const T *a, ptrdiff_t lda,
                                             ptrdiff_t offset, T *bp)
{
    if (!trans) {
        qblas_ytrsm_iuncopy(m, n, a, lda, offset, bp, unit, conj);
    } else {
        qblas_ytrsm_iltcopy(m, n, a, lda, offset, bp, unit, conj);
    }
}

static inline void pack_trsm_a_rside_backward(int upper, int trans, int unit, int conj,
                                              ptrdiff_t m, ptrdiff_t n,
                                              const T *a, ptrdiff_t lda,
                                              ptrdiff_t offset, T *bp)
{
    if (!trans) {
        qblas_ytrsm_ilncopy(m, n, a, lda, offset, bp, unit, conj);
    } else {
        qblas_ytrsm_iutcopy(m, n, a, lda, offset, bp, unit, conj);
    }
}


/* ── SIDE='L' driver — complex twin of trsm_L_band ─────────────────── */
static void trsm_L_band(int upper, int trans, int unit, int conj,
                        int M, int js0, int js1,
                        int MC, int KC, int NC,
                        const T *a, int lda,
                        T *b, int ldb,
                        T *Ap, T *Bp)
{
    const T dm1r = -1.0Q, dm1i = 0.0Q;
    int m = M;
    const int forward = (!upper && !trans) || (upper && trans);
    const int kt = forward ? 1 : 0;

    for (int js = js0; js < js1; js += NC) {
        int min_j = js1 - js;
        if (min_j > NC) min_j = NC;

        if (forward) {
            for (int ls = 0; ls < m; ls += KC) {
                int min_l = m - ls;
                if (min_l > KC) min_l = KC;
                int min_i = min_l;
                if (min_i > MC) min_i = MC;

                pack_trsm_a_lside_forward(upper, trans, unit, conj,
                                          min_l, min_i,
                                          &a[(size_t)ls * 2 + (size_t)ls * lda * 2], lda,
                                          0, Ap);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    qblas_ygemm_ncopy(min_l, min_jj, /*conj=*/0,
                                      &b[(size_t)ls * 2 + (size_t)jjs * ldb * 2], ldb,
                                      Bp + (size_t)min_l * (jjs - js) * 2);
                    qblas_ytrsm_kernel(/*left=*/1, kt,
                                       min_i, min_jj, min_l,
                                       Ap, Bp + (size_t)min_l * (jjs - js) * 2,
                                       &b[(size_t)ls * 2 + (size_t)jjs * ldb * 2], ldb,
                                       /*offset=*/0);
                }

                for (int is = ls + min_i; is < ls + min_l; is += MC) {
                    min_i = ls + min_l - is;
                    if (min_i > MC) min_i = MC;
                    pack_trsm_a_lside_forward(upper, trans, unit, conj,
                                              min_l, min_i,
                                              !trans
                                                ? &a[(size_t)is * 2 + (size_t)ls * lda * 2]
                                                : &a[(size_t)ls * 2 + (size_t)is * lda * 2],
                                              lda,
                                              is - ls, Ap);
                    qblas_ytrsm_kernel(/*left=*/1, kt,
                                       min_i, min_j, min_l,
                                       Ap, Bp,
                                       &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb,
                                       is - ls);
                }

                for (int is = ls + min_l; is < m; is += MC) {
                    min_i = m - is;
                    if (min_i > MC) min_i = MC;
                    if (!trans) {
                        qblas_ygemm_tcopy(min_l, min_i, conj,
                                          &a[(size_t)is * 2 + (size_t)ls * lda * 2], lda, Ap);
                    } else {
                        qblas_ygemm_ncopy(min_l, min_i, conj,
                                          &a[(size_t)ls * 2 + (size_t)is * lda * 2], lda, Ap);
                    }
                    qblas_ygemm_kernel(min_i, min_j, min_l, dm1r, dm1i,
                                       Ap, Bp,
                                       &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb);
                }
            }
        } else {
            for (int ls = m; ls > 0; ls -= KC) {
                int min_l = ls;
                if (min_l > KC) min_l = KC;
                int start_is = ls - min_l;
                while (start_is + MC < ls) start_is += MC;
                int min_i = ls - start_is;
                if (min_i > MC) min_i = MC;

                pack_trsm_a_lside_backward(upper, trans, unit, conj,
                                           min_l, min_i,
                                           !trans
                                             ? &a[(size_t)start_is * 2 + (size_t)(ls - min_l) * lda * 2]
                                             : &a[(size_t)(ls - min_l) * 2 + (size_t)start_is * lda * 2],
                                           lda,
                                           start_is - (ls - min_l), Ap);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    qblas_ygemm_ncopy(min_l, min_jj, /*conj=*/0,
                                      &b[(size_t)(ls - min_l) * 2 + (size_t)jjs * ldb * 2], ldb,
                                      Bp + (size_t)min_l * (jjs - js) * 2);
                    qblas_ytrsm_kernel(/*left=*/1, kt,
                                       min_i, min_jj, min_l,
                                       Ap, Bp + (size_t)min_l * (jjs - js) * 2,
                                       &b[(size_t)start_is * 2 + (size_t)jjs * ldb * 2], ldb,
                                       start_is - ls + min_l);
                }

                for (int is = start_is - MC; is >= ls - min_l; is -= MC) {
                    min_i = ls - is;
                    if (min_i > MC) min_i = MC;
                    pack_trsm_a_lside_backward(upper, trans, unit, conj,
                                               min_l, min_i,
                                               !trans
                                                 ? &a[(size_t)is * 2 + (size_t)(ls - min_l) * lda * 2]
                                                 : &a[(size_t)(ls - min_l) * 2 + (size_t)is * lda * 2],
                                               lda,
                                               is - (ls - min_l), Ap);
                    qblas_ytrsm_kernel(/*left=*/1, kt,
                                       min_i, min_j, min_l,
                                       Ap, Bp,
                                       &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb,
                                       is - (ls - min_l));
                }

                for (int is = 0; is < ls - min_l; is += MC) {
                    min_i = ls - min_l - is;
                    if (min_i > MC) min_i = MC;
                    if (!trans) {
                        qblas_ygemm_tcopy(min_l, min_i, conj,
                                          &a[(size_t)is * 2 + (size_t)(ls - min_l) * lda * 2], lda, Ap);
                    } else {
                        qblas_ygemm_ncopy(min_l, min_i, conj,
                                          &a[(size_t)(ls - min_l) * 2 + (size_t)is * lda * 2], lda, Ap);
                    }
                    qblas_ygemm_kernel(min_i, min_j, min_l, dm1r, dm1i,
                                       Ap, Bp,
                                       &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb);
                }
            }
        }
    }
}


/* ── SIDE='R' driver — complex twin of trsm_R_band ─────────────────── */
static void trsm_R_band(int upper, int trans, int unit, int conj,
                        int N, int m_lo, int m_hi,
                        int MC, int KC, int NC,
                        const T *a, int lda,
                        T *b, int ldb,
                        T *Ap, T *Bp)
{
    const T dm1r = -1.0Q, dm1i = 0.0Q;
    const int m_band = m_hi - m_lo;
    if (m_band <= 0) return;
    const int forward = (upper && !trans) || (!upper && trans);
    const int kt = forward ? 0 : 1;
    T *sa = Ap;
    T *sb = Bp;

    if (forward) {
        for (int js = 0; js < N; js += NC) {
            int min_j = N - js;
            if (min_j > NC) min_j = NC;

            for (int ls = 0; ls < js; ls += KC) {
                int min_l = js - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                qblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        qblas_ygemm_ncopy(min_l, min_jj, conj,
                                          &a[(size_t)ls * 2 + (size_t)jjs * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    } else {
                        qblas_ygemm_tcopy(min_l, min_jj, conj,
                                          &a[(size_t)jjs * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    }
                    qblas_ygemm_kernel(min_i, min_jj, min_l, dm1r, dm1i,
                                       sa, sb + (size_t)min_l * (jjs - js) * 2,
                                       &b[(size_t)m_lo * 2 + (size_t)jjs * ldb * 2], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    qblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);
                    qblas_ygemm_kernel(min_i, min_j, min_l, dm1r, dm1i,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) * 2 + (size_t)js * ldb * 2], ldb);
                }
            }

            for (int ls = js; ls < js + min_j; ls += KC) {
                int min_l = js + min_j - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                qblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                pack_trsm_a_rside_forward(upper, trans, unit, conj,
                                          min_l, min_l,
                                          &a[(size_t)ls * 2 + (size_t)ls * lda * 2], lda,
                                          0, sb);

                qblas_ytrsm_kernel(/*left=*/0, kt,
                                   min_i, min_l, min_l,
                                   sa, sb,
                                   &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb,
                                   /*offset=*/0);

                for (int jjs = 0; jjs < min_j - min_l - ls + js; jjs += NR) {
                    int min_jj = min_j - min_l - ls + js - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        qblas_ygemm_ncopy(min_l, min_jj, conj,
                                          &a[(size_t)ls * 2 + (size_t)(ls + min_l + jjs) * lda * 2], lda,
                                          sb + (size_t)min_l * (min_l + jjs) * 2);
                    } else {
                        qblas_ygemm_tcopy(min_l, min_jj, conj,
                                          &a[(size_t)(ls + min_l + jjs) * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * (min_l + jjs) * 2);
                    }
                    qblas_ygemm_kernel(min_i, min_jj, min_l, dm1r, dm1i,
                                       sa, sb + (size_t)min_l * (min_l + jjs) * 2,
                                       &b[(size_t)m_lo * 2 + (size_t)(min_l + ls + jjs) * ldb * 2], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    qblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);
                    qblas_ytrsm_kernel(/*left=*/0, kt,
                                       min_i, min_l, min_l,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb,
                                       0);
                    if (min_j - min_l + js - ls > 0) {
                        qblas_ygemm_kernel(min_i, min_j - min_l + js - ls, min_l, dm1r, dm1i,
                                           sa, sb + (size_t)min_l * min_l * 2,
                                           &b[(size_t)(m_lo + is) * 2 + (size_t)(min_l + ls) * ldb * 2], ldb);
                    }
                }
            }
        }
    } else {
        for (int js = N; js > 0; js -= NC) {
            int min_j = js;
            if (min_j > NC) min_j = NC;

            for (int ls = js; ls < N; ls += KC) {
                int min_l = N - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                qblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = min_j + js - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        qblas_ygemm_ncopy(min_l, min_jj, conj,
                                          &a[(size_t)ls * 2 + (size_t)(jjs - min_j) * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    } else {
                        qblas_ygemm_tcopy(min_l, min_jj, conj,
                                          &a[(size_t)(jjs - min_j) * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    }
                    qblas_ygemm_kernel(min_i, min_jj, min_l, dm1r, dm1i,
                                       sa, sb + (size_t)min_l * (jjs - js) * 2,
                                       &b[(size_t)m_lo * 2 + (size_t)(jjs - min_j) * ldb * 2], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    qblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);
                    qblas_ygemm_kernel(min_i, min_j, min_l, dm1r, dm1i,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) * 2 + (size_t)(js - min_j) * ldb * 2], ldb);
                }
            }

            int start_ls = js - min_j;
            while (start_ls + KC < js) start_ls += KC;

            for (int ls = start_ls; ls >= js - min_j; ls -= KC) {
                int min_l = js - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                qblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                pack_trsm_a_rside_backward(upper, trans, unit, conj,
                                           min_l, min_l,
                                           &a[(size_t)ls * 2 + (size_t)ls * lda * 2], lda,
                                           0,
                                           sb + (size_t)min_l * (min_j - js + ls) * 2);

                qblas_ytrsm_kernel(/*left=*/0, kt,
                                   min_i, min_l, min_l,
                                   sa, sb + (size_t)min_l * (min_j - js + ls) * 2,
                                   &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb,
                                   0);

                for (int jjs = 0; jjs < min_j - js + ls; jjs += NR) {
                    int min_jj = min_j - js + ls - jjs;
                    if (min_jj > NR) min_jj = NR;
                    if (!trans) {
                        qblas_ygemm_ncopy(min_l, min_jj, conj,
                                          &a[(size_t)ls * 2 + (size_t)(js - min_j + jjs) * lda * 2], lda,
                                          sb + (size_t)min_l * jjs * 2);
                    } else {
                        qblas_ygemm_tcopy(min_l, min_jj, conj,
                                          &a[(size_t)(js - min_j + jjs) * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * jjs * 2);
                    }
                    qblas_ygemm_kernel(min_i, min_jj, min_l, dm1r, dm1i,
                                       sa, sb + (size_t)min_l * jjs * 2,
                                       &b[(size_t)m_lo * 2 + (size_t)(js - min_j + jjs) * ldb * 2], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;
                    qblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);
                    qblas_ytrsm_kernel(/*left=*/0, kt,
                                       min_i, min_l, min_l,
                                       sa, sb + (size_t)min_l * (min_j - js + ls) * 2,
                                       &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb,
                                       0);
                    if (min_j - js + ls > 0) {
                        qblas_ygemm_kernel(min_i, min_j - js + ls, min_l, dm1r, dm1i,
                                           sa, sb,
                                           &b[(size_t)(m_lo + is) * 2 + (size_t)(js - min_j) * ldb * 2], ldb);
                    }
                }
            }
        }
    }
}


/* ── Public entry ──────────────────────────────────────────────────── */
void xtrsm_(
    const char *side_p, const char *uplo_p,
    const char *transa_p, const char *diag_p,
    const int *m_, const int *n_, const T *alpha_,   /* alpha is 2 __float128s */
    const T *a, const int *lda_,
    T *b, const int *ldb_)
{

    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha_r = alpha_[0];
    const T alpha_i = alpha_[1];

    const int lside  = (toupper((unsigned char)*side_p)  == 'L');
    const int upper  = (toupper((unsigned char)*uplo_p)  == 'U');
    const char trc   = (char)toupper((unsigned char)*transa_p);
    const int trans  = (trc == 'T' || trc == 'C');
    const int conj   = (trc == 'C');
    const int nounit = (toupper((unsigned char)*diag_p)  == 'N');
    const int unit   = !nounit;

    if (M == 0 || N == 0) return;

    /* alpha == 0 zeroes B outright; no pack buffers are needed. */
    if (alpha_r == 0.0Q && alpha_i == 0.0Q) {
        qblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, alpha_r, alpha_i, b, (ptrdiff_t)ldb);
        return;
    }

    int MC0, KC, NC;
    qblas_ygemm_blocks(&MC0, &KC, &NC);

    int K_eff = lside ? M : N;
    int MC = MC0;
    if (K_eff <= KC) {
        const long L2_TARGET_BYTES = QBLAS_L2_TARGET_BYTES;
        long target_mc = L2_TARGET_BYTES / ((long)K_eff * (long)sizeof(T) * 2);
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((int)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    /* 2x the size for complex (2 __float128s per element). */
    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * sizeof(T) * 2;
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * sizeof(T) * 2;

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
    if (!Ap_arr || !Bp_arr) {
        free(Ap_arr); free(Bp_arr);
        fprintf(stderr, "xtrsm: pack buffer allocation failed\n");
        abort();
    }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        Bp_arr[t] = aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t] || !Bp_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) {
            free(Ap_arr[t]);
            free(Bp_arr[t]);
        }
        free(Ap_arr); free(Bp_arr);
        fprintf(stderr, "xtrsm: pack buffer allocation failed\n");
        abort();
    }

    /* Pre-scale B by alpha (complex). Deferred until after pack-buffer
     * allocation so an allocation failure can never leave B scaled but
     * untransformed. */
    if (alpha_r != 1.0Q || alpha_i != 0.0Q) {
        qblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, alpha_r, alpha_i, b, (ptrdiff_t)ldb);
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
                trsm_L_band(upper, trans, unit, conj,
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
                trsm_R_band(upper, trans, unit, conj,
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
