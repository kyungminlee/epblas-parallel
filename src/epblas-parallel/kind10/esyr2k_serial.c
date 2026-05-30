/*
 * esyr2k_serial — kind10 real (long double) symmetric rank-2k, single-thread.
 * This TU owns ALL of the esyr2k math; esyr2k_parallel.c only orchestrates the
 * diagonal-block loop across a team.
 *
 *   C := alpha·(A·Bᵀ + B·Aᵀ) + beta·C   (TRANS='N')
 *   C := alpha·(Aᵀ·B + Bᵀ·A) + beta·C   (TRANS='T'/'C')
 *
 * C is N×N symmetric (only the UPLO triangle is touched). Blocked: scalar
 * rank-2k diagonal + two egemm trailing calls per block. The trailing updates
 * run through egemm_serial (NOT egemm_): when esyr2k_block runs inside the
 * team esyr2k_parallel.c opened, a nested egemm team would trip the libgomp
 * barrier wedge (memory project-etrsm-omp4-wedge).
 */

#include "esyr2k_kernel.h"
#include <stdlib.h>
#include <ctype.h>

typedef esyr2k_T T;

static int g_esyr2k_nb = 0;
int esyr2k_nb(void) {
    if (g_esyr2k_nb == 0) {
        g_esyr2k_nb = 64;
        const char *s = getenv("ESYR2K_NB");
        if (s && *s) { int v = atoi(s); if (v > 0) g_esyr2k_nb = v; }
    }
    return g_esyr2k_nb;
}

extern void egemm_serial(
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
                           char UPLO, char TR)
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

void esyr2k_beta_scale(int j_start, int j_end, int N, T beta,
                       T *c, int ldc, char UPLO)
{
    const T zero = 0.0L;
    for (int j = j_start; j < j_end; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == zero) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero;
        else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }
}

void esyr2k_block(int jc, int jb, int N, int K, T alpha, T beta,
                  const T *a, int lda, const T *b, int ldb,
                  T *c, int ldc, char UPLO, char TR)
{
    const T zero = 0.0L, one = 1.0L;
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

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

    /* (3) Off-diagonal trailing via two egemm_serial calls (A·Bᵀ + B·Aᵀ). */
    if (UPLO == 'L') {
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            const int j0 = jc + jb;
            if (TR == 'N') {
                /* C(j0:, jc..) += alpha · A(j0:, :) · B(jc.., :)ᵀ */
                egemm_serial(NN, TN, &trailing, &jb, &K, &alpha,
                             &A_(j0, 0), &lda, &B_(jc, 0), &ldb, &one,
                             &C_(j0, jc), &ldc, 1, 1);
                /* C(j0:, jc..) += alpha · B(j0:, :) · A(jc.., :)ᵀ */
                egemm_serial(NN, TN, &trailing, &jb, &K, &alpha,
                             &B_(j0, 0), &ldb, &A_(jc, 0), &lda, &one,
                             &C_(j0, jc), &ldc, 1, 1);
            } else {
                /* C(j0:, jc..) += alpha · A(:, j0:)ᵀ · B(:, jc..) */
                egemm_serial(TN, NN, &trailing, &jb, &K, &alpha,
                             &A_(0, j0), &lda, &B_(0, jc), &ldb, &one,
                             &C_(j0, jc), &ldc, 1, 1);
                egemm_serial(TN, NN, &trailing, &jb, &K, &alpha,
                             &B_(0, j0), &ldb, &A_(0, jc), &lda, &one,
                             &C_(j0, jc), &ldc, 1, 1);
            }
        }
    } else {  /* UPLO == 'U' */
        if (jc > 0) {
            if (TR == 'N') {
                egemm_serial(NN, TN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &B_(jc, 0), &ldb, &one,
                             &C_(0, jc), &ldc, 1, 1);
                egemm_serial(NN, TN, &jc, &jb, &K, &alpha,
                             &B_(0, 0), &ldb, &A_(jc, 0), &lda, &one,
                             &C_(0, jc), &ldc, 1, 1);
            } else {
                egemm_serial(TN, NN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &B_(0, jc), &ldb, &one,
                             &C_(0, jc), &ldc, 1, 1);
                egemm_serial(TN, NN, &jc, &jb, &K, &alpha,
                             &B_(0, 0), &ldb, &A_(0, jc), &lda, &one,
                             &C_(0, jc), &ldc, 1, 1);
            }
        }
    }
}

void esyr2k_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
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

    if (alpha == zero || K == 0) {
        if (beta == one) return;
        esyr2k_beta_scale(0, N, N, beta, c, ldc, UPLO);
        return;
    }

    /* TR-aware nb (see esyr2k_kernel.h): the TR='T' diag kernel saturates the
     * x87 2-stream fadd ceiling, so run it as one full-N block (no trailing
     * egemm); TR='N' uses the tunable block size. */
    const int nb = (TR == 'T') ? N : esyr2k_nb();
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        esyr2k_block(jc, jb, N, K, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TR);
    }
}

#undef A_
#undef B_
#undef C_
