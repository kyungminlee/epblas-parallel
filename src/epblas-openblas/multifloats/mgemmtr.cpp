/*
 * mgemmtr — kind10 (REAL(KIND=10) / 80-bit multifloats::float64x2) port of OpenBLAS DGEMMT.
 *
 *   C := alpha * op(A) * op(B) + beta * C   (only UPLO triangle of C touched)
 *   op(X) ∈ {X, X^T}; A is (n×k) or (k×n); B is (k×n) or (n×k); C is n×n.
 *
 * Port source: OpenBLAS.
 *   - interface/gemmt.c                (entire algorithm — column-by-column
 *                                        GEMV, no separate driver/level3/*)
 *
 * Unlike the other L3 routines, OpenBLAS GEMMT does NOT use the GotoBLAS
 * three-level blocked nest. It is implemented as N independent GEMV
 * calls — one per output column, with the active C-column slice
 * pre-scaled by beta (SCAL_K), then accumulated into via a single
 * GEMV against the appropriate strip of op(A) and op(B)'s i-th column.
 *
 *   uplo == 'L' (lower): for i in [0, n):
 *     length j = n - i        // active rows: [i, n)
 *     aa = (transa=='N') ? a + i : a + lda*i
 *     bb = (transb=='N') ? b + i*ldb : b + i
 *     cc = c[i, i]            // stride 1, length j
 *     scal(j, beta, cc)
 *     gemv(transa, aa, lda, bb, incb, cc, 1, alpha)  // y += alpha*op(A)*x
 *
 *   uplo == 'U' (upper): for i in [0, n):
 *     length j = i + 1        // active rows: [0, i+1)
 *     aa = a            (full top strip from row 0)
 *     bb = (transb=='N') ? b + i*ldb : b + i
 *     cc = c[0, i]
 *     scal(j, beta, cc); gemv as above
 *
 * Threading: OpenBLAS calls gemv_thread per column (SMP-parallel inside
 * the single GEMV, columns serialized). We call mgemv_ which already
 * has internal OMP; the column loop is sequential to avoid nested
 * parallelism. mgemv_ has its own work-threshold for OMP=1 fallback
 * when (j*k) is tiny.
 *
 * Fortran ABI:
 *   subroutine mgemmtr(uplo, transa, transb, n, k, alpha, a, lda,
 *                      b, ldb, beta, c, ldc)
 */

#include <stddef.h>
#include <multifloats.h>
#include <multifloats/float64x2.h>
#include <stdlib.h>
#include <ctype.h>

typedef multifloats::float64x2 T;

extern "C" void mgemv_(const char *TRANS, const int *M, const int *N,
                   const T *ALPHA, const T *a, const int *LDA,
                   const T *x, const int *INCX,
                   const T *BETA, T *y, const int *INCY);

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}
static inline char trans_code(const char *p) {
    char c = up(p);
    return (c == 'C') ? 'T' : c;
}

extern "C" void mgemmtr_(const char *uplo, const char *transa, const char *transb,
              const int *m_, const int *k_,
              const T *alpha_,
              const T *a, const int *lda_,
              const T *b, const int *ldb_,
              const T *beta_,
              T *c, const int *ldc_)
{
    const int m = *m_, k = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    const char ta = trans_code(transa);
    const char tb = trans_code(transb);

    if (m == 0) return;

    const T one = 1.0;
    const int incb = (tb == 'T') ? ldb : 1;
    const int incy_one = 1;

    if (UPLO == 'L') {
        for (int i = 0; i < m; ++i) {
            const int j = m - i;
            const T *aa = (ta == 'T') ? a + (size_t)lda * i : a + i;
            const T *bb = (tb == 'T') ? b + i : b + (size_t)i * ldb;
            T *cc = c + (size_t)i * ldc + i;

            if (beta != one) {
                if (beta == 0.0) for (int p = 0; p < j; ++p) cc[p] = 0.0;
                else              for (int p = 0; p < j; ++p) cc[p] *= beta;
            }
            if (alpha == 0.0 || k == 0) continue;

            int gm, gn;
            if (ta == 'N') { gm = j; gn = k; }
            else           { gm = k; gn = j; }
            mgemv_(&ta, &gm, &gn, &alpha, aa, &lda, bb, &incb,
                   &one, cc, &incy_one);
        }
    } else {
        for (int i = 0; i < m; ++i) {
            const int j = i + 1;
            const T *bb = (tb == 'T') ? b + i : b + (size_t)i * ldb;
            T *cc = c + (size_t)i * ldc;

            if (beta != one) {
                if (beta == 0.0) for (int p = 0; p < j; ++p) cc[p] = 0.0;
                else              for (int p = 0; p < j; ++p) cc[p] *= beta;
            }
            if (alpha == 0.0 || k == 0) continue;

            int gm, gn;
            if (ta == 'N') { gm = j; gn = k; }
            else           { gm = k; gn = j; }
            mgemv_(&ta, &gm, &gn, &alpha, a, &lda, bb, &incb,
                   &one, cc, &incy_one);
        }
    }
}
