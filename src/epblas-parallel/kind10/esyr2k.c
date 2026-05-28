/*
 * esyr2k — kind10 (REAL(KIND=10) / `long double`) symmetric rank-2k update.
 *
 *   C := alpha · (A · Bᵀ + B · Aᵀ) + beta · C         (TRANS='N')
 *   C := alpha · (Aᵀ · B + Bᵀ · A) + beta · C         (TRANS='T'/'C')
 *
 * C is N×N symmetric; only the UPLO triangle is touched.
 *
 * Same blocked pattern as esyrk: column-panel over jc, scalar rank-2k
 * on the jb×jb diagonal block (with stride-1 A,B reads matching the
 * Netlib DSYR2K reference), then two egemm calls for the off-diagonal
 * trailing update (one for A·Bᵀ half, one for B·Aᵀ half).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ESYR2K_OMP_MIN 32

typedef long double T;

static int g_esyr2k_nb = 64;

__attribute__((constructor))
static void esyr2k_init(void) {
    const char *s = getenv("ESYR2K_NB");
    if (s && *s) {
        int v = atoi(s);
        if (v > 0) g_esyr2k_nb = v;
    }
}
static int syr2k_nb(void) { return g_esyr2k_nb; }

extern void egemm_(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* Scalar rank-2k diagonal-block add. No beta scaling (caller pre-scales). */
static void syr2k_diag_add(int jc, int jb, int K, T alpha,
                           const T *restrict a, int lda,
                           const T *restrict b, int ldb,
                           T *restrict c, int ldc,
                           char UPLO, int TR)
{
    if (TR == 'N') {
        /* C(I,J) += alpha * sum_l (A(I,l)*B(J,l) + B(I,l)*A(J,l))
         * Inner i loop walks stride-1 over A(:,l) and B(:,l). */
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            for (int l = 0; l < K; ++l) {
                const T t1 = alpha * A_(j, l);
                const T t2 = alpha * B_(j, l);
                for (int i = i_lo; i < i_hi; ++i) {
                    cj[i] += B_(i, l) * t1 + A_(i, l) * t2;
                }
            }
        }
    } else {
        /* C(I,J) += alpha * sum_l (A(l,I)*B(l,J) + B(l,I)*A(l,J))
         * 2-chain dot product, same trick as esyrk TR='T'. */
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + (size_t)j * ldc;
            const T *Aj = a + (size_t)j * lda;
            const T *Bj = b + (size_t)j * ldb;
            for (int i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + (size_t)i * lda;
                const T *Bi = b + (size_t)i * ldb;
                T s0 = 0.0L, s1 = 0.0L;
                int l = 0;
                for (; l + 1 < K; l += 2) {
                    s0 += Ai[l] * Bj[l] + Bi[l] * Aj[l];
                    s1 += Ai[l + 1] * Bj[l + 1] + Bi[l + 1] * Aj[l + 1];
                }
                T s = s0 + s1;
                for (; l < K; ++l) s += Ai[l] * Bj[l] + Bi[l] * Aj[l];
                cj[i] += alpha * s;
            }
        }
    }
}

void esyr2k_(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict b, const int *ldb_,
    const T *beta_,
    T *restrict c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    const T zero = 0.0L, one = 1.0L;
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    if (alpha == zero || K == 0) {
        if (beta == one) return;
#ifdef _OPENMP
        const int use_omp = (N >= ESYR2K_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero;
            else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
        }
        return;
    }

    /* TR-aware nb. The TR='T' diag kernel already saturates the x87
     * 2-stream fadd ceiling (~2.7 GF/s on this CPU); the trailing-egemm
     * panel split adds framing overhead that the gemm-vs-scalar win
     * doesn't recover. For TR='N' the rank-2-update diag kernel is
     * slower and the trailing egemm calls are a clear win, so the
     * default 64 holds. Verified via nb sweep across N∈{128,256,512}. */
    const int nb = (TR == 'T') ? N : syr2k_nb();

#ifdef _OPENMP
    const int use_omp = (N >= ESYR2K_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;

        /* (1) Beta-scale this block's UPLO slice. */
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            if (beta == zero)      for (int i = i_lo; i < i_hi; ++i) cj[i]  = zero;
            else if (beta != one)  for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
        }

        /* (2) Diagonal block: scalar rank-2k add. */
        syr2k_diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR);

        /* (3) Off-diagonal trailing via two egemm calls (A·Bᵀ + B·Aᵀ). */
        if (UPLO == 'L') {
            const int trailing = N - jc - jb;
            if (trailing > 0) {
                const int j0 = jc + jb;
                if (TR == 'N') {
                    /* C(j0:, jc..) += alpha · A(j0:, :) · B(jc.., :)ᵀ */
                    egemm_(NN, TN, &trailing, &jb, &K, &alpha,
                           &A_(j0, 0), &lda, &B_(jc, 0), &ldb, &one,
                           &C_(j0, jc), &ldc, 1, 1);
                    /* C(j0:, jc..) += alpha · B(j0:, :) · A(jc.., :)ᵀ */
                    egemm_(NN, TN, &trailing, &jb, &K, &alpha,
                           &B_(j0, 0), &ldb, &A_(jc, 0), &lda, &one,
                           &C_(j0, jc), &ldc, 1, 1);
                } else {
                    /* C(j0:, jc..) += alpha · A(:, j0:)ᵀ · B(:, jc..) */
                    egemm_(TN, NN, &trailing, &jb, &K, &alpha,
                           &A_(0, j0), &lda, &B_(0, jc), &ldb, &one,
                           &C_(j0, jc), &ldc, 1, 1);
                    egemm_(TN, NN, &trailing, &jb, &K, &alpha,
                           &B_(0, j0), &ldb, &A_(0, jc), &lda, &one,
                           &C_(j0, jc), &ldc, 1, 1);
                }
            }
        } else {  /* UPLO == 'U' */
            if (jc > 0) {
                if (TR == 'N') {
                    egemm_(NN, TN, &jc, &jb, &K, &alpha,
                           &A_(0, 0), &lda, &B_(jc, 0), &ldb, &one,
                           &C_(0, jc), &ldc, 1, 1);
                    egemm_(NN, TN, &jc, &jb, &K, &alpha,
                           &B_(0, 0), &ldb, &A_(jc, 0), &lda, &one,
                           &C_(0, jc), &ldc, 1, 1);
                } else {
                    egemm_(TN, NN, &jc, &jb, &K, &alpha,
                           &A_(0, 0), &lda, &B_(0, jc), &ldb, &one,
                           &C_(0, jc), &ldc, 1, 1);
                    egemm_(TN, NN, &jc, &jb, &K, &alpha,
                           &B_(0, 0), &ldb, &A_(0, jc), &lda, &one,
                           &C_(0, jc), &ldc, 1, 1);
                }
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
