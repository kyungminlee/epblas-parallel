/*
 * wtrmm — kind10 (COMPLEX(KIND=10)) port of OpenBLAS ZTRMM.
 *
 *   B := alpha * op(A) * B    (SIDE='L')
 *   B := alpha * B * op(A)    (SIDE='R')
 *
 * Faithful L3-blocked port (complex twin of etrmm.c).
 *
 * Port source: OpenBLAS.
 *   - interface/trsm.c                (TRMM macro path, Z-variant)
 *   - driver/level3/trmm_L.c          (SIDE='L' driver)
 *   - driver/level3/trmm_R.c          (SIDE='R' driver)
 *   - kernel/generic/ztrmmkernel_2x2.c → mblas_ytrmm_kernel (NN-only;
 *                                        conjugation absorbed into the
 *                                        packers via the `conj` flag)
 *   - kernel/generic/ztrmm_{ut,un,lt,ln}copy_2.c
 *                                     → mblas_ytrmm_i{ut,un,lt,ln}copy
 *
 * Same algorithmic structure as etrmm.c — see that file's banner for
 * the (uplo, trans, side) → packer/kernel routing.
 *
 * Complex notes:
 *   - Per-element width = 2 multifloats::float64x2s (interleaved re,im).
 *   - lda, ldb in COMPLEX(KIND=10) elements; the kernels and packers
 *     handle the float-stride doubling internally.
 *   - 'C' on transa = A^H (conjugate transpose). The packers take a
 *     `conj` flag that sign-flips the imag word at write time, so
 *     the kernel always runs the NN form (matches ygemm pattern).
 *
 * Fortran ABI:
 *   subroutine wtrmm(side, uplo, transa, diag, m, n, alpha, a, lda, b, ldb)
 *   - character args are plain char* — NO trailing hidden length args
 *     (declaring them caused the v0.9.1 frame corruption; never re-add)
 *   - alpha is COMPLEX(KIND=10): 2 multifloats::float64x2s (re, im)
 *   - a, b are COMPLEX(KIND=10) arrays (interleaved re,im)
 *   - lda, ldb are in COMPLEX(KIND=10) elements
 */

#include "mblas_l3_complex.h"
#include "mblas_tuning.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#endif

typedef multifloats::float64x2 T;

#define MR MBLAS_YGEMM_MR
#define NR MBLAS_YGEMM_NR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }


/* Same dispatch logic as etrmm.c's pack_trmm_a — see comment there for
 * the SIDE-dependent (uplo, trans) → packer mapping. `conj` is the
 * conjugation flag (set when user asked for 'C' transa). */
static inline void pack_trmm_a(int side_l, int uplo_upper, int trans, int unit, int conj,
                               ptrdiff_t m, ptrdiff_t n,
                               const T *a, ptrdiff_t lda,
                               ptrdiff_t posX, ptrdiff_t posY,
                               T *bp)
{
    if (side_l) {
        if (uplo_upper && !trans)       mblas_ytrmm_iutcopy(m, n, a, lda, posX, posY, bp, unit, conj);
        else if (uplo_upper &&  trans)  mblas_ytrmm_iuncopy(m, n, a, lda, posX, posY, bp, unit, conj);
        else if (!uplo_upper && !trans) mblas_ytrmm_iltcopy(m, n, a, lda, posX, posY, bp, unit, conj);
        else                            mblas_ytrmm_ilncopy(m, n, a, lda, posX, posY, bp, unit, conj);
    } else {
        if (uplo_upper && !trans)       mblas_ytrmm_iuncopy(m, n, a, lda, posX, posY, bp, unit, conj);
        else if (uplo_upper &&  trans)  mblas_ytrmm_iutcopy(m, n, a, lda, posX, posY, bp, unit, conj);
        else if (!uplo_upper && !trans) mblas_ytrmm_ilncopy(m, n, a, lda, posX, posY, bp, unit, conj);
        else                            mblas_ytrmm_iltcopy(m, n, a, lda, posX, posY, bp, unit, conj);
    }
}


/* ── SIDE='L' driver — complex twin of trmm_L_band ──────────────────── */
static void trmm_L_band(int upper, int trans, int unit, int conj,
                        int M, int js0, int js1,
                        int MC, int KC, int NC,
                        const T *a, int lda,
                        T *b, int ldb,
                        T *Ap, T *Bp)
{
    const T dp1r = 1.0, dp1i = 0.0;
    int m = M;

    for (int js = js0; js < js1; js += NC) {
        int min_j = js1 - js;
        if (min_j > NC) min_j = NC;

        if ((upper && !trans) || (!upper && trans)) {
            const int kt = 0;   /* TRMM_KERNEL_N */
            int min_l = m;
            if (min_l > KC) min_l = KC;
            int min_i = min_l;
            if (min_i > MC) min_i = MC;
            if (min_i > MR) min_i = (min_i / MR) * MR;

            pack_trmm_a(1, upper, trans, unit, conj, min_l, min_i, a, lda, 0, 0, Ap);

            for (int jjs = js; jjs < js + min_j; jjs += NR) {
                int min_jj = js + min_j - jjs;
                if (min_jj > NR) min_jj = NR;

                mblas_ygemm_ncopy(min_l, min_jj, /*conj=*/0,
                                  &b[(size_t)jjs * ldb * 2], ldb,
                                  Bp + (size_t)min_l * (jjs - js) * 2);

                mblas_ytrmm_kernel(/*left=*/1, kt,
                                   min_i, min_jj, min_l, dp1r, dp1i,
                                   Ap, Bp + (size_t)min_l * (jjs - js) * 2,
                                   &b[(size_t)jjs * ldb * 2], ldb, 0);
            }

            for (int is = min_i; is < min_l; is += min_i) {
                min_i = min_l - is;
                if (min_i > MC) min_i = MC;
                if (min_i > MR) min_i = (min_i / MR) * MR;

                pack_trmm_a(1, upper, trans, unit, conj, min_l, min_i, a, lda, 0, is, Ap);

                mblas_ytrmm_kernel(/*left=*/1, kt,
                                   min_i, min_j, min_l, dp1r, dp1i,
                                   Ap, Bp,
                                   &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb, is);
            }

            for (int ls = min_l; ls < m; ls += KC) {
                min_l = m - ls;
                if (min_l > KC) min_l = KC;
                min_i = ls;
                if (min_i > MC) min_i = MC;
                if (min_i > MR) min_i = (min_i / MR) * MR;

                /* GEMM_I{T,N}COPY with the user's conj. For !TRANS use
                 * TCOPY (normal A); for TRANS use NCOPY. The conjugation
                 * flag is the user's `conj` only when trans is set
                 * (since 'C' = TRANS+CONJ; 'T' = TRANS+!CONJ; 'N' is
                 * never paired with conj). */
                if (!trans) {
                    mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &a[(size_t)ls * lda * 2], lda, Ap);
                } else {
                    mblas_ygemm_ncopy(min_l, min_i, /*conj=*/conj,
                                      &a[(size_t)ls * 2], lda, Ap);
                }

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;

                    mblas_ygemm_ncopy(min_l, min_jj, /*conj=*/0,
                                      &b[(size_t)ls * 2 + (size_t)jjs * ldb * 2], ldb,
                                      Bp + (size_t)min_l * (jjs - js) * 2);

                    mblas_ygemm_kernel_store(min_i, min_jj, min_l, dp1r, dp1i,
                                             Ap, Bp + (size_t)min_l * (jjs - js) * 2,
                                             &b[(size_t)jjs * ldb * 2], ldb);
                }

                for (int is = min_i; is < ls; is += min_i) {
                    min_i = ls - is;
                    if (min_i > MC) min_i = MC;
                    if (min_i > MR) min_i = (min_i / MR) * MR;

                    if (!trans) {
                        mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                          &a[(size_t)is * 2 + (size_t)ls * lda * 2], lda, Ap);
                    } else {
                        mblas_ygemm_ncopy(min_l, min_i, /*conj=*/conj,
                                          &a[(size_t)ls * 2 + (size_t)is * lda * 2], lda, Ap);
                    }

                    mblas_ygemm_kernel_store(min_i, min_j, min_l, dp1r, dp1i,
                                             Ap, Bp,
                                             &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb);
                }

                for (int is = ls; is < ls + min_l; is += min_i) {
                    min_i = ls + min_l - is;
                    if (min_i > MC) min_i = MC;
                    if (min_i > MR) min_i = (min_i / MR) * MR;

                    pack_trmm_a(1, upper, trans, unit, conj, min_l, min_i, a, lda, ls, is, Ap);

                    mblas_ytrmm_kernel(/*left=*/1, kt,
                                       min_i, min_j, min_l, dp1r, dp1i,
                                       Ap, Bp,
                                       &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb,
                                       is - ls);
                }
            }
        } else {
            const int kt = 1;   /* TRMM_KERNEL_T */
            int min_l = m;
            if (min_l > KC) min_l = KC;
            int min_i = min_l;
            if (min_i > MC) min_i = MC;
            if (min_i > MR) min_i = (min_i / MR) * MR;

            pack_trmm_a(1, upper, trans, unit, conj, min_l, min_i, a, lda,
                        m - min_l, m - min_l, Ap);

            for (int jjs = js; jjs < js + min_j; jjs += NR) {
                int min_jj = js + min_j - jjs;
                if (min_jj > NR) min_jj = NR;

                mblas_ygemm_ncopy(min_l, min_jj, /*conj=*/0,
                                  &b[(size_t)(m - min_l) * 2 + (size_t)jjs * ldb * 2], ldb,
                                  Bp + (size_t)min_l * (jjs - js) * 2);

                mblas_ytrmm_kernel(/*left=*/1, kt,
                                   min_i, min_jj, min_l, dp1r, dp1i,
                                   Ap, Bp + (size_t)min_l * (jjs - js) * 2,
                                   &b[(size_t)(m - min_l) * 2 + (size_t)jjs * ldb * 2], ldb, 0);
            }

            for (int is = m - min_l + min_i; is < m; is += min_i) {
                min_i = m - is;
                if (min_i > MC) min_i = MC;
                if (min_i > MR) min_i = (min_i / MR) * MR;

                pack_trmm_a(1, upper, trans, unit, conj, min_l, min_i, a, lda,
                            m - min_l, is, Ap);

                mblas_ytrmm_kernel(/*left=*/1, kt,
                                   min_i, min_j, min_l, dp1r, dp1i,
                                   Ap, Bp,
                                   &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb,
                                   is - m + min_l);
            }

            for (int ls = m - min_l; ls > 0; ls -= KC) {
                min_l = ls;
                if (min_l > KC) min_l = KC;
                min_i = min_l;
                if (min_i > MC) min_i = MC;
                if (min_i > MR) min_i = (min_i / MR) * MR;

                pack_trmm_a(1, upper, trans, unit, conj, min_l, min_i, a, lda,
                            ls - min_l, ls - min_l, Ap);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;

                    mblas_ygemm_ncopy(min_l, min_jj, /*conj=*/0,
                                      &b[(size_t)(ls - min_l) * 2 + (size_t)jjs * ldb * 2], ldb,
                                      Bp + (size_t)min_l * (jjs - js) * 2);

                    mblas_ytrmm_kernel(/*left=*/1, kt,
                                       min_i, min_jj, min_l, dp1r, dp1i,
                                       Ap, Bp + (size_t)min_l * (jjs - js) * 2,
                                       &b[(size_t)(ls - min_l) * 2 + (size_t)jjs * ldb * 2], ldb, 0);
                }

                for (int is = ls - min_l + min_i; is < ls; is += min_i) {
                    min_i = ls - is;
                    if (min_i > MC) min_i = MC;
                    if (min_i > MR) min_i = (min_i / MR) * MR;

                    pack_trmm_a(1, upper, trans, unit, conj, min_l, min_i, a, lda,
                                ls - min_l, is, Ap);

                    mblas_ytrmm_kernel(/*left=*/1, kt,
                                       min_i, min_j, min_l, dp1r, dp1i,
                                       Ap, Bp,
                                       &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb,
                                       is - ls + min_l);
                }

                for (int is = ls; is < m; is += min_i) {
                    min_i = m - is;
                    if (min_i > MC) min_i = MC;
                    if (min_i > MR) min_i = (min_i / MR) * MR;

                    if (!trans) {
                        mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                          &a[(size_t)is * 2 + (size_t)(ls - min_l) * lda * 2], lda, Ap);
                    } else {
                        mblas_ygemm_ncopy(min_l, min_i, /*conj=*/conj,
                                          &a[(size_t)(ls - min_l) * 2 + (size_t)is * lda * 2], lda, Ap);
                    }

                    mblas_ygemm_kernel_store(min_i, min_j, min_l, dp1r, dp1i,
                                             Ap, Bp,
                                             &b[(size_t)is * 2 + (size_t)js * ldb * 2], ldb);
                }
            }
        }
    }
}


/* ── SIDE='R' driver — complex twin of trmm_R_band ──────────────────── */
static void trmm_R_band(int upper, int trans, int unit, int conj,
                        int N, int m_lo, int m_hi,
                        int MC, int KC, int NC,
                        const T *a, int lda,
                        T *b, int ldb,
                        T *Ap, T *Bp)
{
    (void)MC;
    const T dp1r = 1.0, dp1i = 0.0;
    int m_band = m_hi - m_lo;
    if (m_band <= 0) return;

    T *sa = Ap;
    T *sb = Bp;

    if ((!upper && !trans) || (upper && trans)) {
        const int kt = 1;
        for (int js = 0; js < N; js += NC) {
            int min_j = N - js;
            if (min_j > NC) min_j = NC;

            for (int ls = js; ls < js + min_j; ls += KC) {
                int min_l = js + min_j - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                for (int jjs = 0; jjs < ls - js; jjs += NR) {
                    int min_jj = ls - js - jjs;
                    if (min_jj > NR) min_jj = NR;

                    if (!trans) {
                        mblas_ygemm_ncopy(min_l, min_jj, /*conj=*/0,
                                          &a[(size_t)ls * 2 + (size_t)(js + jjs) * lda * 2], lda,
                                          sb + (size_t)min_l * jjs * 2);
                    } else {
                        mblas_ygemm_tcopy(min_l, min_jj, /*conj=*/conj,
                                          &a[(size_t)(js + jjs) * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * jjs * 2);
                    }

                    mblas_ygemm_kernel_store(min_i, min_jj, min_l, dp1r, dp1i,
                                             sa, sb + (size_t)min_l * jjs * 2,
                                             &b[(size_t)m_lo * 2 + (size_t)(js + jjs) * ldb * 2], ldb);
                }

                for (int jjs = 0; jjs < min_l; jjs += NR) {
                    int min_jj = min_l - jjs;
                    if (min_jj > NR) min_jj = NR;

                    pack_trmm_a(0, upper, trans, unit, conj, min_l, min_jj, a, lda,
                                ls, ls + jjs,
                                sb + (size_t)min_l * (ls - js + jjs) * 2);

                    mblas_ytrmm_kernel(/*left=*/0, kt,
                                       min_i, min_jj, min_l, dp1r, dp1i,
                                       sa, sb + (size_t)min_l * (ls - js + jjs) * 2,
                                       &b[(size_t)m_lo * 2 + (size_t)(ls + jjs) * ldb * 2], ldb,
                                       -jjs);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);

                    mblas_ygemm_kernel_store(min_i, ls - js, min_l, dp1r, dp1i,
                                             sa, sb,
                                             &b[(size_t)(m_lo + is) * 2 + (size_t)js * ldb * 2], ldb);

                    mblas_ytrmm_kernel(/*left=*/0, kt,
                                       min_i, min_l, min_l, dp1r, dp1i,
                                       sa, sb + (size_t)(ls - js) * min_l * 2,
                                       &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, 0);
                }
            }

            for (int ls = js + min_j; ls < N; ls += KC) {
                int min_l = N - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = js + min_j - jjs;
                    if (min_jj > NR) min_jj = NR;

                    if (!trans) {
                        mblas_ygemm_ncopy(min_l, min_jj, /*conj=*/0,
                                          &a[(size_t)ls * 2 + (size_t)jjs * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    } else {
                        mblas_ygemm_tcopy(min_l, min_jj, /*conj=*/conj,
                                          &a[(size_t)jjs * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    }

                    mblas_ygemm_kernel_store(min_i, min_jj, min_l, dp1r, dp1i,
                                             sa, sb + (size_t)min_l * (jjs - js) * 2,
                                             &b[(size_t)m_lo * 2 + (size_t)jjs * ldb * 2], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);

                    mblas_ygemm_kernel_store(min_i, min_j, min_l, dp1r, dp1i,
                                             sa, sb,
                                             &b[(size_t)(m_lo + is) * 2 + (size_t)js * ldb * 2], ldb);
                }
            }
        }
    } else {
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

                mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                for (int jjs = 0; jjs < min_l; jjs += NR) {
                    int min_jj = min_l - jjs;
                    if (min_jj > NR) min_jj = NR;

                    pack_trmm_a(0, upper, trans, unit, conj, min_l, min_jj, a, lda,
                                ls, ls + jjs,
                                sb + (size_t)min_l * jjs * 2);

                    mblas_ytrmm_kernel(/*left=*/0, kt,
                                       min_i, min_jj, min_l, dp1r, dp1i,
                                       sa, sb + (size_t)min_l * jjs * 2,
                                       &b[(size_t)m_lo * 2 + (size_t)(ls + jjs) * ldb * 2], ldb,
                                       -jjs);
                }

                for (int jjs = 0; jjs < js - ls - min_l; jjs += NR) {
                    int min_jj = js - ls - min_l - jjs;
                    if (min_jj > NR) min_jj = NR;

                    if (!trans) {
                        mblas_ygemm_ncopy(min_l, min_jj, /*conj=*/0,
                                          &a[(size_t)ls * 2 + (size_t)(ls + min_l + jjs) * lda * 2], lda,
                                          sb + (size_t)min_l * (min_l + jjs) * 2);
                    } else {
                        mblas_ygemm_tcopy(min_l, min_jj, /*conj=*/conj,
                                          &a[(size_t)(ls + min_l + jjs) * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * (min_l + jjs) * 2);
                    }

                    mblas_ygemm_kernel_store(min_i, min_jj, min_l, dp1r, dp1i,
                                             sa, sb + (size_t)min_l * (min_l + jjs) * 2,
                                             &b[(size_t)m_lo * 2 + (size_t)(ls + min_l + jjs) * ldb * 2], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);

                    mblas_ytrmm_kernel(/*left=*/0, kt,
                                       min_i, min_l, min_l, dp1r, dp1i,
                                       sa, sb,
                                       &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, 0);

                    if (js - ls - min_l > 0) {
                        mblas_ygemm_kernel_store(min_i, js - ls - min_l, min_l, dp1r, dp1i,
                                                 sa, sb + (size_t)min_l * min_l * 2,
                                                 &b[(size_t)(m_lo + is) * 2 + (size_t)(ls + min_l) * ldb * 2], ldb);
                    }
                }
            }

            for (int ls = 0; ls < js - min_j; ls += KC) {
                int min_l = js - min_j - ls;
                if (min_l > KC) min_l = KC;
                int min_i = m_band;
                if (min_i > MC) min_i = MC;

                mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                  &b[(size_t)m_lo * 2 + (size_t)ls * ldb * 2], ldb, sa);

                for (int jjs = js; jjs < js + min_j; jjs += NR) {
                    int min_jj = min_j + js - jjs;
                    if (min_jj > NR) min_jj = NR;

                    if (!trans) {
                        mblas_ygemm_ncopy(min_l, min_jj, /*conj=*/0,
                                          &a[(size_t)ls * 2 + (size_t)(jjs - min_j) * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    } else {
                        mblas_ygemm_tcopy(min_l, min_jj, /*conj=*/conj,
                                          &a[(size_t)(jjs - min_j) * 2 + (size_t)ls * lda * 2], lda,
                                          sb + (size_t)min_l * (jjs - js) * 2);
                    }

                    mblas_ygemm_kernel_store(min_i, min_jj, min_l, dp1r, dp1i,
                                             sa, sb + (size_t)min_l * (jjs - js) * 2,
                                             &b[(size_t)m_lo * 2 + (size_t)(jjs - min_j) * ldb * 2], ldb);
                }

                for (int is = min_i; is < m_band; is += MC) {
                    min_i = m_band - is;
                    if (min_i > MC) min_i = MC;

                    mblas_ygemm_tcopy(min_l, min_i, /*conj=*/0,
                                      &b[(size_t)(m_lo + is) * 2 + (size_t)ls * ldb * 2], ldb, sa);

                    mblas_ygemm_kernel_store(min_i, min_j, min_l, dp1r, dp1i,
                                             sa, sb,
                                             &b[(size_t)(m_lo + is) * 2 + (size_t)(js - min_j) * ldb * 2], ldb);
                }
            }
        }
    }
}


/* ── Public entry ───────────────────────────────────────────────── */
extern "C" void wtrmm_(
    const char *side_p, const char *uplo_p,
    const char *transa_p, const char *diag_p,
    const int *m_, const int *n_, const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_)
{

    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alphar = alpha_[0], alphai = alpha_[1];

    const int lside  = (toupper((unsigned char)*side_p)  == 'L');
    const int upper  = (toupper((unsigned char)*uplo_p)  == 'U');
    const char trc   = (char)toupper((unsigned char)*transa_p);
    const int trans  = (trc == 'T' || trc == 'C');
    const int conj   = (trc == 'C');
    const int nounit = (toupper((unsigned char)*diag_p)  == 'N');
    const int unit   = !nounit;

    if (M == 0 || N == 0) return;

    /* alpha == 0 zeroes B outright; no pack buffers are needed. */
    if (alphar == 0.0 && alphai == 0.0) {
        mblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, alphar, alphai, b, (ptrdiff_t)ldb);
        return;
    }

    int MC0, KC, NC;
    mblas_ygemm_blocks(&MC0, &KC, &NC);

    int K_eff = lside ? M : N;
    int MC = MC0;
    if (K_eff <= KC) {
        const long L2_TARGET_BYTES = MBLAS_L2_TARGET_BYTES;
        long target_mc = L2_TARGET_BYTES / ((long)K_eff * 2L * (long)sizeof(T));
        if (target_mc > MC) {
            if (target_mc > 4L * MC0) target_mc = 4L * MC0;
            MC = round_up((int)target_mc, MR);
            if (MC < MC0) MC = MC0;
        }
    }

    /* Buffer sizes — complex = 2 multifloats::float64x2s per element. */
    const size_t ap_bytes = (size_t)round_up(MC, MR) * (size_t)KC * 2 * sizeof(T);
    const size_t bp_bytes = (size_t)KC * (size_t)round_up(NC, NR) * 2 * sizeof(T);

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

    T **Ap_arr = (T **)calloc((size_t)nthreads, sizeof(T *));
    T **Bp_arr = (T **)calloc((size_t)nthreads, sizeof(T *));
    if (!Ap_arr || !Bp_arr) {
        free(Ap_arr); free(Bp_arr);
        fprintf(stderr, "wtrmm: pack buffer allocation failed\n");
        abort();
    }
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = (T *)aligned_alloc(64, (ap_bytes + 63) & ~(size_t)63);
        Bp_arr[t] = (T *)aligned_alloc(64, (bp_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t] || !Bp_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) {
            free(Ap_arr[t]);
            free(Bp_arr[t]);
        }
        free(Ap_arr); free(Bp_arr);
        fprintf(stderr, "wtrmm: pack buffer allocation failed\n");
        abort();
    }

    /* alpha pre-scale of B (mirrors interface/trsm.c TRMM macro path).
     * Deferred until after pack-buffer allocation so an allocation
     * failure can never leave B scaled but untransformed. */
    if (!(alphar == 1.0 && alphai == 0.0)) {
        mblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, alphar, alphai, b, (ptrdiff_t)ldb);
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
                trmm_L_band(upper, trans, unit, conj,
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
                trmm_R_band(upper, trans, unit, conj,
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
