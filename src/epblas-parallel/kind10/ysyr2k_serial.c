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
 * barrier wedge.
 */

#include "ysyr2k_kernel.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef ysyr2k_T T;

static ptrdiff_t g_ysyr2k_nb = 0;
ptrdiff_t ysyr2k_nb(void) {
    if (g_ysyr2k_nb == 0) {
        g_ysyr2k_nb = 32;
        const char *s = getenv("YSYR2K_NB");
        if (s && *s) { ptrdiff_t v = atoi(s); if (v > 0) g_ysyr2k_nb = v; }
    }
    return g_ysyr2k_nb;
}

extern void ygemm_serial(
    const char *transa, const char *transb,
    const ptrdiff_t *m, const ptrdiff_t *n, const ptrdiff_t *k,
    const T *alpha,
    const T *a, const ptrdiff_t *lda,
    const T *b, const ptrdiff_t *ldb,
    const T *beta,
    T *c, const ptrdiff_t *ldc,
    size_t transa_len, size_t transb_len);

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* TRANS='T' diagonal-block update (dot form: each C element accumulates in
 * a register over the K axis). Only the Trans path uses this; the NoTrans
 * path runs a flat per-column rank-1 sweep inline in ysyr2k_block. */
static void syr2k_diag_add_t(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t K, T alpha,
                             const T *restrict a, ptrdiff_t lda,
                             const T *restrict b, ptrdiff_t ldb,
                             T *restrict c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
        const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
        T *cj = c + (size_t)j * ldc;
        const T *Aj = a + (size_t)j * lda;
        const T *Bj = b + (size_t)j * ldb;
        for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
            const T *Ai = a + (size_t)i * lda;
            const T *Bi = b + (size_t)i * ldb;
            T s = ZERO;
            for (ptrdiff_t l = 0; l < K; ++l) s += Ai[l] * Bj[l] + Bi[l] * Aj[l];
            cj[i] += alpha * s;
        }
    }
}

void ysyr2k_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t N, T beta,
                       T *c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = ZERO;
        else              for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }
}

void ysyr2k_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t N, ptrdiff_t K, T alpha, T beta,
                  const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                  T *c, ptrdiff_t ldc, char UPLO, char TR)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
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
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
            const ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
            T *cj = c + (size_t)j * ldc;
            for (ptrdiff_t l = 0; l < K; ++l) {
                const T t1 = alpha * A_(j, l);
                const T t2 = alpha * B_(j, l);
                const T *al = a + (size_t)l * lda, *bl = b + (size_t)l * ldb;
                for (ptrdiff_t i = i_lo; i < i_hi; ++i)
                    cj[i] += bl[i] * t1 + al[i] * t2;
            }
        }
        return;
    }

    syr2k_diag_add_t(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO);

    if (UPLO == 'L') {
        const ptrdiff_t trailing = N - jc - jb;
        if (trailing > 0) {
            const ptrdiff_t j0 = jc + jb;
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
    const ptrdiff_t *n_, const ptrdiff_t *k_,
    const T *alpha_,
    const T *a, const ptrdiff_t *lda_,
    const T *b, const ptrdiff_t *ldb_,
    const T *beta_,
    T *c, const ptrdiff_t *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const ptrdiff_t N = *n_, K = *k_;
    const ptrdiff_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
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

    const ptrdiff_t nb = ysyr2k_nb();
    for (ptrdiff_t jc = 0; jc < N; jc += nb) {
        const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        ysyr2k_block(jc, jb, N, K, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TR);
    }
}

#undef A_
#undef B_
#undef C_
