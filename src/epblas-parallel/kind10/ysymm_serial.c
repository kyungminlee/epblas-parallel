/*
 * ysymm_serial — kind10 complex (_Complex long double) symmetric
 * matrix-multiply, single-thread. This TU owns ALL of the ysymm math;
 * ysymm_parallel.c only orchestrates the outer panel loop across a team.
 *
 * Blocked "read A_IK once, use it twice" recipe (see ysymm_kernel.h). The
 * trailing updates run through ygemm_serial (NOT ygemm_): when a panel
 * worker runs inside the team ysymm_parallel.c opened, a nested ygemm team
 * would trip the libgomp barrier wedge.
 */

#include "ysymm_kernel.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef ysymm_T T;

/* nb is fixed at 32 (the original symm_nb_pick clamps to [32, 32]). */
ptrdiff_t ysymm_nb(void) { return 32; }

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

/* Scalar diagonal-block symm for SIDE='L' (see original ysymm for the
 * Netlib stride-1 access rationale). */
static void symm_diag_add_L(ptrdiff_t ic, ptrdiff_t ib, ptrdiff_t jc, ptrdiff_t jb, T alpha,
                            const T *restrict a, ptrdiff_t lda,
                            const T *restrict b, ptrdiff_t ldb,
                            T *restrict c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        const T *bj = b + (size_t)j * ldb;
        if (UPLO == 'L') {
            for (ptrdiff_t i = ic + ib - 1; i >= ic; --i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (ptrdiff_t k = i + 1; k < ic + ib; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * ai[k];
                }
                cj[i] += temp1 * ai[i] + alpha * temp2;
            }
        } else {
            for (ptrdiff_t i = ic; i < ic + ib; ++i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (ptrdiff_t k = ic; k < i; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * ai[k];
                }
                cj[i] += temp1 * ai[i] + alpha * temp2;
            }
        }
    }
}

/* Scalar diagonal-block symm for SIDE='R'. */
static void symm_diag_add_R(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t ic, ptrdiff_t ib, T alpha,
                            const T *restrict a, ptrdiff_t lda,
                            const T *restrict b, ptrdiff_t ldb,
                            T *restrict c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        {
            const T t = alpha * A_(j, j);
            for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, j);
        }
        if (UPLO == 'L') {
            for (ptrdiff_t k = jc; k < j; ++k) {
                const T t = alpha * A_(j, k);
                if (t != ZERO) for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
            for (ptrdiff_t k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * A_(k, j);
                if (t != ZERO) for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
        } else {
            for (ptrdiff_t k = jc; k < j; ++k) {
                const T t = alpha * A_(k, j);
                if (t != ZERO) for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
            for (ptrdiff_t k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * A_(j, k);
                if (t != ZERO) for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
        }
    }
}

void ysymm_beta_only(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M, T beta, T *c, ptrdiff_t ldc)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (ptrdiff_t i = 0; i < M; ++i) cj[i] = ZERO;
        else              for (ptrdiff_t i = 0; i < M; ++i) cj[i] *= beta;
    }
}

void ysymm_L_singleblock(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t M,
                         T alpha, T beta,
                         const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                         T *c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        T *cj = c + (size_t)j * ldc;
        const T *bj = b + (size_t)j * ldb;
        if (UPLO == 'L') {
            for (ptrdiff_t i = M - 1; i >= 0; --i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (ptrdiff_t k = i + 1; k < M; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * ai[k];
                }
                const T diag = temp1 * ai[i] + alpha * temp2;
                if (beta == ZERO)     cj[i] = diag;
                else if (beta == ONE) cj[i] += diag;
                else                  cj[i] = beta * cj[i] + diag;
            }
        } else {
            for (ptrdiff_t i = 0; i < M; ++i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (ptrdiff_t k = 0; k < i; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * ai[k];
                }
                const T diag = temp1 * ai[i] + alpha * temp2;
                if (beta == ZERO)     cj[i] = diag;
                else if (beta == ONE) cj[i] += diag;
                else                  cj[i] = beta * cj[i] + diag;
            }
        }
    }
}

void ysymm_L_panel(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t M, T alpha, T beta,
                   const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                   T *c, ptrdiff_t ldc, char UPLO, ptrdiff_t nb)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (ptrdiff_t i = 0; i < M; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (ptrdiff_t i = 0; i < M; ++i) cj[i] *= beta;
    }

    for (ptrdiff_t ic = 0; ic < M; ic += nb) {
        const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;

        if (UPLO == 'L') {
            for (ptrdiff_t kc = 0; kc < ic; kc += nb) {
                const ptrdiff_t kb = (ic - kc < nb) ? (ic - kc) : nb;
                ygemm_serial(TN, NN, &kb, &jb, &ib, &alpha,
                             &A_(ic, kc), &lda, &B_(ic, jc), &ldb, &ONE,
                             &C_(kc, jc), &ldc, 1, 1);
                ygemm_serial(NN, NN, &ib, &jb, &kb, &alpha,
                             &A_(ic, kc), &lda, &B_(kc, jc), &ldb, &ONE,
                             &C_(ic, jc), &ldc, 1, 1);
            }
        } else {
            for (ptrdiff_t kc = ic + ib; kc < M; kc += nb) {
                const ptrdiff_t kb = (M - kc < nb) ? (M - kc) : nb;
                ygemm_serial(TN, NN, &kb, &jb, &ib, &alpha,
                             &A_(ic, kc), &lda, &B_(ic, jc), &ldb, &ONE,
                             &C_(kc, jc), &ldc, 1, 1);
                ygemm_serial(NN, NN, &ib, &jb, &kb, &alpha,
                             &A_(ic, kc), &lda, &B_(kc, jc), &ldb, &ONE,
                             &C_(ic, jc), &ldc, 1, 1);
            }
        }

        symm_diag_add_L(ic, ib, jc, jb, alpha, a, lda, b, ldb, c, ldc, UPLO);
    }
}

void ysymm_R_panel(ptrdiff_t ic, ptrdiff_t ib, ptrdiff_t N, T alpha, T beta,
                   const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                   T *c, ptrdiff_t ldc, char UPLO, ptrdiff_t nb)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    for (ptrdiff_t j = 0; j < N; ++j) {
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] *= beta;
    }

    for (ptrdiff_t jc = 0; jc < N; jc += nb) {
        const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;

        if (UPLO == 'L') {
            for (ptrdiff_t kc = jc + jb; kc < N; kc += nb) {
                const ptrdiff_t kb = (N - kc < nb) ? (N - kc) : nb;
                ygemm_serial(NN, NN, &ib, &jb, &kb, &alpha,
                             &B_(ic, kc), &ldb, &A_(kc, jc), &lda, &ONE,
                             &C_(ic, jc), &ldc, 1, 1);
                ygemm_serial(NN, TN, &ib, &kb, &jb, &alpha,
                             &B_(ic, jc), &ldb, &A_(kc, jc), &lda, &ONE,
                             &C_(ic, kc), &ldc, 1, 1);
            }
        } else {
            for (ptrdiff_t kc = 0; kc < jc; kc += nb) {
                const ptrdiff_t kb = (jc - kc < nb) ? (jc - kc) : nb;
                ygemm_serial(NN, NN, &ib, &jb, &kb, &alpha,
                             &B_(ic, kc), &ldb, &A_(kc, jc), &lda, &ONE,
                             &C_(ic, jc), &ldc, 1, 1);
                ygemm_serial(NN, TN, &ib, &kb, &jb, &alpha,
                             &B_(ic, jc), &ldb, &A_(kc, jc), &lda, &ONE,
                             &C_(ic, kc), &ldc, 1, 1);
            }
        }

        symm_diag_add_R(jc, jb, ic, ib, alpha, a, lda, b, ldb, c, ldc, UPLO);
    }
}

void ysymm_serial(
    const char *side, const char *uplo,
    const ptrdiff_t *m_, const ptrdiff_t *n_,
    const T *alpha_,
    const T *a, const ptrdiff_t *lda_,
    const T *b, const ptrdiff_t *ldb_,
    const T *beta_,
    T *c, const ptrdiff_t *ldc_,
    size_t side_len, size_t uplo_len)
{
    (void)side_len; (void)uplo_len;
    const ptrdiff_t M = *m_, N = *n_;
    const ptrdiff_t lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = (char)toupper((unsigned char)*side);
    const char UPLO = (char)toupper((unsigned char)*uplo);

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
        ysymm_beta_only(0, N, M, beta, c, ldc);
        return;
    }

    const ptrdiff_t nb = ysymm_nb();

    if (SIDE == 'L' && M <= nb) {
        ysymm_L_singleblock(0, N, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }

    if (SIDE == 'L') {
        for (ptrdiff_t jc = 0; jc < N; jc += nb) {
            const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
            ysymm_L_panel(jc, jb, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    } else {
        for (ptrdiff_t ic = 0; ic < M; ic += nb) {
            const ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            ysymm_R_panel(ic, ib, N, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    }
}

#undef A_
#undef B_
#undef C_
