/*
 * xgemmtr — kind16 complex (__complex128) port of OpenBLAS ZGEMMT.
 *
 *   C := alpha * op(A) * op(B) + beta * C   (only UPLO triangle of C touched)
 *   op(X) ∈ {X, X^T, X^H, conj(X)} (codes N/T/C/R; complex only).
 *
 * Port source: OpenBLAS.
 *   - interface/gemmt.c     (the algorithm is entirely here — see qgemmtr.c
 *                            for the full structural commentary; this is the
 *                            complex twin)
 *
 * Complex-only twist: op(B) with conjugation. OpenBLAS pre-conjugates B
 * in-place (IMATCOPY_K_CNC) when transb ∈ {R,C}, runs the column loop
 * with effective transb stripped of its conj bit, then restores B at
 * the end. We mirror this exactly. transa conjugation is absorbed by
 * xgemv:
 *   transa='N' → xgemv 'N'
 *   transa='T' → xgemv 'T'
 *   transa='C' → xgemv 'C'  (y += alpha * A^H * x)
 *   transa='R' → pre-conjugate A in-place; xgemv 'N'; restore.
 *
 * Fortran ABI:
 *   subroutine xgemmtr(uplo, transa, transb, n, k, alpha, a, lda,
 *                      b, ldb, beta, c, ldc)
 */

#include <stddef.h>
#include <quadmath.h>
#include <stdlib.h>
#include <ctype.h>
#include <complex.h>

typedef __complex128 C;

extern void xgemv_(const char *TRANS, const int *M, const int *N,
                   const C *ALPHA, const C *a, const int *LDA,
                   const C *x, const int *INCX,
                   const C *BETA, C *y, const int *INCY,
                   size_t trans_len);

static inline char up_(const char *p) {
    return (char)toupper((unsigned char)*p);
}

/* Conjugate `rows × cols` column-major submatrix in place. */
static void conj_in_place(C *m, int rows, int cols, int ldm)
{
    for (int j = 0; j < cols; ++j) {
        C *col = m + (size_t)j * ldm;
        for (int i = 0; i < rows; ++i) col[i] = conjq(col[i]);
    }
}

void xgemmtr_(const char *uplo, const char *transa, const char *transb,
              const int *m_, const int *k_,
              const C *alpha_,
              C *a, const int *lda_,
              C *b, const int *ldb_,
              const C *beta_,
              C *c, const int *ldc_,
              size_t uplo_len, size_t ta_len, size_t tb_len)
{
    (void)uplo_len; (void)ta_len; (void)tb_len;
    const int m = *m_, k = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const C alpha = *alpha_, beta = *beta_;
    const char UPLO = up_(uplo);
    const char ta_raw = up_(transa);
    const char tb_raw = up_(transb);

    if (m == 0) return;

    /* Decode transa: drop conj into a pre-conjugation pass for 'R'.
     * 'N','T','C' pass through to xgemv directly. */
    char ta;
    int conj_a = 0;
    if (ta_raw == 'R')      { ta = 'N'; conj_a = 1; }
    else                     { ta = ta_raw; }   /* 'N','T','C' */

    /* Decode transb: drop conj into a pre-conjugation pass for 'R'/'C'. */
    char tb;
    int conj_b = 0;
    if (tb_raw == 'R')      { tb = 'N'; conj_b = 1; }
    else if (tb_raw == 'C') { tb = 'T'; conj_b = 1; }
    else                     { tb = tb_raw; }   /* 'N','T' */

    /* Dimensions of A and B for the in-place conjugation passes. */
    const int rows_a = (ta_raw == 'N' || ta_raw == 'R') ? m : k;
    const int cols_a = (ta_raw == 'N' || ta_raw == 'R') ? k : m;
    const int rows_b = (tb_raw == 'N' || tb_raw == 'R') ? k : m;
    const int cols_b = (tb_raw == 'N' || tb_raw == 'R') ? m : k;

    if (conj_a) conj_in_place(a, rows_a, cols_a, lda);
    if (conj_b) conj_in_place(b, rows_b, cols_b, ldb);

    const C one  = (__complex128)(1.0Q);
    const int incb = (tb == 'T') ? ldb : 1;
    const int incy_one = 1;

    if (UPLO == 'L') {
        for (int i = 0; i < m; ++i) {
            const int j = m - i;
            const C *aa = (ta != 'N') ? a + (size_t)lda * i : a + i;
            const C *bb = (tb == 'T') ? b + i : b + (size_t)i * ldb;
            C *cc = c + (size_t)i * ldc + i;

            if (beta != one) {
                if (beta == 0.0Q) for (int p = 0; p < j; ++p) cc[p] = 0.0Q;
                else              for (int p = 0; p < j; ++p) cc[p] *= beta;
            }
            if (alpha == 0.0Q || k == 0) continue;

            int gm, gn;
            if (ta == 'N') { gm = j; gn = k; }
            else           { gm = k; gn = j; }
            xgemv_(&ta, &gm, &gn, &alpha, aa, &lda, bb, &incb,
                   &one, cc, &incy_one, 1);
        }
    } else {
        for (int i = 0; i < m; ++i) {
            const int j = i + 1;
            const C *bb = (tb == 'T') ? b + i : b + (size_t)i * ldb;
            C *cc = c + (size_t)i * ldc;

            if (beta != one) {
                if (beta == 0.0Q) for (int p = 0; p < j; ++p) cc[p] = 0.0Q;
                else              for (int p = 0; p < j; ++p) cc[p] *= beta;
            }
            if (alpha == 0.0Q || k == 0) continue;

            int gm, gn;
            if (ta == 'N') { gm = j; gn = k; }
            else           { gm = k; gn = j; }
            xgemv_(&ta, &gm, &gn, &alpha, a, &lda, bb, &incb,
                   &one, cc, &incy_one, 1);
        }
    }

    /* Restore A and B (in-place conjugation is its own inverse). */
    if (conj_a) conj_in_place(a, rows_a, cols_a, lda);
    if (conj_b) conj_in_place(b, rows_b, cols_b, ldb);
}
