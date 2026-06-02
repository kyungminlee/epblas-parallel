/*
 * ysyr2k_serial — kind10 complex (_Complex long double) symmetric rank-2k,
 * single-thread. This TU owns ALL of the ysyr2k math; ysyr2k_parallel.c only
 * orchestrates the diagonal-block loop across a team.
 *
 * C := alpha·(A·Bᵀ + B·Aᵀ) + beta·C  (TRANS='N'), or the transposed form
 * (TRANS='T'). C is complex SYMMETRIC. Blocked: scalar rank-2k diagonal +
 * two ygemm trailing calls per block. The trailing updates run through
 * ygemm_serial (NOT ygemm_): when ysyr2k_block runs inside the team
 * ysyr2k_parallel.c opened, a nested ygemm team would trip the libgomp
 * barrier wedge (memory project-etrsm-omp4-wedge).
 */

#include "ysyr2k_kernel.h"
#include <stdlib.h>
#include <ctype.h>

typedef ysyr2k_T T;

static int g_ysyr2k_nb = 0;
int ysyr2k_nb(void) {
    if (g_ysyr2k_nb == 0) {
        g_ysyr2k_nb = 32;
        const char *s = getenv("YSYR2K_NB");
        if (s && *s) { int v = atoi(s); if (v > 0) g_ysyr2k_nb = v; }
    }
    return g_ysyr2k_nb;
}

extern void ygemm_serial(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* TRANS='T' diagonal-block update (dot form: each C element accumulates in
 * a register over the K axis). Only the Trans path uses this; the NoTrans
 * path runs a flat per-column rank-1 sweep inline in ysyr2k_block. */
static void syr2k_diag_add_t(int jc, int jb, int K, T alpha,
                             const T *restrict a, int lda,
                             const T *restrict b, int ldb,
                             T *restrict c, int ldc, char UPLO)
{
    for (int j = jc; j < jc + jb; ++j) {
        const int i_lo = (UPLO == 'L') ? j     : jc;
        const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
        T *cj = c + (size_t)j * ldc;
        const T *Aj = a + (size_t)j * lda;
        const T *Bj = b + (size_t)j * ldb;
        for (int i = i_lo; i < i_hi; ++i) {
            const T *Ai = a + (size_t)i * lda;
            const T *Bi = b + (size_t)i * ldb;
            T s = ZERO;
            for (int l = 0; l < K; ++l) s += Ai[l] * Bj[l] + Bi[l] * Aj[l];
            cj[i] += alpha * s;
        }
    }
}

void ysyr2k_beta_scale(int j_start, int j_end, int N, T beta,
                       T *c, int ldc, char UPLO)
{
    for (int j = j_start; j < j_end; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (int i = i_lo; i < i_hi; ++i) cj[i] = ZERO;
        else              for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }
}

void ysyr2k_block(int jc, int jb, int N, int K, T alpha, T beta,
                  const T *a, int lda, const T *b, int ldb,
                  T *c, int ldc, char UPLO, char TR)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    for (int j = jc; j < jc + jb; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (int i = i_lo; i < i_hi; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (int i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }

    /* TRANS='N': flat per-column rank-2k over each column's FULL stored
     * extent (diagonal triangle + trailing in one sweep), mirroring the
     * Netlib reference. The old diag-block + two-ygemm split isolated the
     * small diagonal triangle into a short inner i-loop (~jb/2 rows), so
     * its strided temp loads and loop overhead never amortized — the
     * diagonal ran ~1.8x slower per flop than the reference (which
     * dominated the serial deficit at small N). One long sweep per column
     * fixes that; columns stay disjoint across blocks so block-level
     * threading is unaffected. TRANS='T' keeps the dot diagonal + ygemm
     * trailing: its register-accumulated dot already beats the reference. */
    if (TR == 'N') {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j : 0;
            const int i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            for (int l = 0; l < K; ++l) {
                const T t1 = alpha * A_(j, l);
                const T t2 = alpha * B_(j, l);
                const T *al = a + (size_t)l * lda, *bl = b + (size_t)l * ldb;
                for (int i = i_lo; i < i_hi; ++i)
                    cj[i] += bl[i] * t1 + al[i] * t2;
            }
        }
        return;
    }

    syr2k_diag_add_t(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO);

    if (UPLO == 'L') {
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            const int j0 = jc + jb;
            ygemm_serial(TN, NN, &trailing, &jb, &K, &alpha,
                         &A_(0, j0), &lda, &B_(0, jc), &ldb, &ONE,
                         &C_(j0, jc), &ldc, 1, 1);
            ygemm_serial(TN, NN, &trailing, &jb, &K, &alpha,
                         &B_(0, j0), &ldb, &A_(0, jc), &lda, &ONE,
                         &C_(j0, jc), &ldc, 1, 1);
        }
    } else {
        if (jc > 0) {
            ygemm_serial(TN, NN, &jc, &jb, &K, &alpha,
                         &A_(0, 0), &lda, &B_(0, jc), &ldb, &ONE,
                         &C_(0, jc), &ldc, 1, 1);
            ygemm_serial(TN, NN, &jc, &jb, &K, &alpha,
                         &B_(0, 0), &ldb, &A_(0, jc), &lda, &ONE,
                         &C_(0, jc), &ldc, 1, 1);
        }
    }
}

void ysyr2k_serial(
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
    const char UPLO = (char)toupper((unsigned char)*uplo);
    char TR = (char)toupper((unsigned char)*trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    if (alpha == ZERO || K == 0) {
        if (beta == ONE) return;
        ysyr2k_beta_scale(0, N, N, beta, c, ldc, UPLO);
        return;
    }

    const int nb = ysyr2k_nb();
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        ysyr2k_block(jc, jb, N, K, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TR);
    }
}

#undef A_
#undef B_
#undef C_
