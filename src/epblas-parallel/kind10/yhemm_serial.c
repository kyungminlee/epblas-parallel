/*
 * yhemm_serial — kind10 complex (_Complex long double) Hermitian
 * matrix-multiply, single-thread. This TU owns ALL of the yhemm math;
 * yhemm_parallel.c only orchestrates the outer panel loop across a team.
 *
 * Same "read A_IK once, use it twice" recipe as ysymm, but Hermitian: the
 * reflection gemm uses 'C' and the scalar diagonal block keeps the diagonal
 * real. The trailing updates run through ygemm_serial (NOT ygemm_): when a
 * panel worker runs inside the team yhemm_parallel.c opened, a nested ygemm
 * team would trip the libgomp barrier wedge (project-etrsm-omp4-wedge).
 */

#include "yhemm_kernel.h"
#include <stdlib.h>
#include <ctype.h>

typedef yhemm_T T;

static int g_yhemm_nb_override = -1;
int yhemm_nb(void) {
    if (g_yhemm_nb_override < 0) {
        g_yhemm_nb_override = 0;
        const char *s = getenv("YHEMM_NB");
        if (s && *s) { int v = atoi(s); if (v > 0) g_yhemm_nb_override = v; }
    }
    /* The original hemm_nb_pick clamps to [32, 32], so nb is always 32
     * unless explicitly overridden. */
    return (g_yhemm_nb_override > 0) ? g_yhemm_nb_override : 32;
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

static inline T cconj(T z) { return ~z; }

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* Scalar Hermitian diagonal block, SIDE='L' (see original yhemm for the
 * Netlib stride-1 access rationale). */
static void hemm_diag_add_L(int ic, int ib, int jc, int jb, T alpha,
                            const T *restrict a, int lda,
                            const T *restrict b, int ldb,
                            T *restrict c, int ldc, char UPLO)
{
    for (int j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        const T *bj = b + (size_t)j * ldb;
        if (UPLO == 'L') {
            for (int i = ic + ib - 1; i >= ic; --i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (int k = i + 1; k < ic + ib; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * cconj(ai[k]);
                }
                cj[i] += temp1 * __real__ ai[i] + alpha * temp2;
            }
        } else {
            for (int i = ic; i < ic + ib; ++i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (int k = ic; k < i; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * cconj(ai[k]);
                }
                cj[i] += temp1 * __real__ ai[i] + alpha * temp2;
            }
        }
    }
}

/* Scalar Hermitian diagonal block, SIDE='R'. */
static void hemm_diag_add_R(int jc, int jb, int ic, int ib, T alpha,
                            const T *restrict a, int lda,
                            const T *restrict b, int ldb,
                            T *restrict c, int ldc, char UPLO)
{
    for (int j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        {
            const T t = alpha * (__real__ A_(j, j));
            for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, j);
        }
        if (UPLO == 'L') {
            for (int k = jc; k < j; ++k) {
                const T t = alpha * cconj(A_(j, k));
                if (t != ZERO) for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
            for (int k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * A_(k, j);
                if (t != ZERO) for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
        } else {
            for (int k = jc; k < j; ++k) {
                const T t = alpha * A_(k, j);
                if (t != ZERO) for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
            for (int k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * cconj(A_(j, k));
                if (t != ZERO) for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
        }
    }
}

void yhemm_beta_only(int j_start, int j_end, int M, T beta, T *c, int ldc)
{
    for (int j = j_start; j < j_end; ++j) {
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (int i = 0; i < M; ++i) cj[i] = ZERO;
        else              for (int i = 0; i < M; ++i) cj[i] *= beta;
    }
}

void yhemm_L_singleblock(int j_start, int j_end, int M,
                         T alpha, T beta,
                         const T *a, int lda, const T *b, int ldb,
                         T *c, int ldc, char UPLO)
{
    for (int j = j_start; j < j_end; ++j) {
        T *cj = c + (size_t)j * ldc;
        const T *bj = b + (size_t)j * ldb;
        if (UPLO == 'L') {
            for (int i = M - 1; i >= 0; --i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (int k = i + 1; k < M; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * cconj(ai[k]);
                }
                const T diag = temp1 * __real__ ai[i] + alpha * temp2;
                if (beta == ZERO)     cj[i] = diag;
                else if (beta == ONE) cj[i] += diag;
                else                  cj[i] = beta * cj[i] + diag;
            }
        } else {
            for (int i = 0; i < M; ++i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (int k = 0; k < i; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * cconj(ai[k]);
                }
                const T diag = temp1 * __real__ ai[i] + alpha * temp2;
                if (beta == ZERO)     cj[i] = diag;
                else if (beta == ONE) cj[i] += diag;
                else                  cj[i] = beta * cj[i] + diag;
            }
        }
    }
}

void yhemm_L_panel(int jc, int jb, int M, T alpha, T beta,
                   const T *a, int lda, const T *b, int ldb,
                   T *c, int ldc, char UPLO, int nb)
{
    const char NN[1] = {'N'};
    const char CN[1] = {'C'};

    for (int j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (int i = 0; i < M; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (int i = 0; i < M; ++i) cj[i] *= beta;
    }

    for (int ic = 0; ic < M; ic += nb) {
        const int ib = (M - ic < nb) ? (M - ic) : nb;

        if (UPLO == 'L') {
            for (int kc = 0; kc < ic; kc += nb) {
                const int kb = (ic - kc < nb) ? (ic - kc) : nb;
                ygemm_serial(CN, NN, &kb, &jb, &ib, &alpha,
                             &A_(ic, kc), &lda, &B_(ic, jc), &ldb, &ONE,
                             &C_(kc, jc), &ldc, 1, 1);
                ygemm_serial(NN, NN, &ib, &jb, &kb, &alpha,
                             &A_(ic, kc), &lda, &B_(kc, jc), &ldb, &ONE,
                             &C_(ic, jc), &ldc, 1, 1);
            }
        } else {
            for (int kc = ic + ib; kc < M; kc += nb) {
                const int kb = (M - kc < nb) ? (M - kc) : nb;
                ygemm_serial(CN, NN, &kb, &jb, &ib, &alpha,
                             &A_(ic, kc), &lda, &B_(ic, jc), &ldb, &ONE,
                             &C_(kc, jc), &ldc, 1, 1);
                ygemm_serial(NN, NN, &ib, &jb, &kb, &alpha,
                             &A_(ic, kc), &lda, &B_(kc, jc), &ldb, &ONE,
                             &C_(ic, jc), &ldc, 1, 1);
            }
        }

        hemm_diag_add_L(ic, ib, jc, jb, alpha, a, lda, b, ldb, c, ldc, UPLO);
    }
}

void yhemm_R_panel(int ic, int ib, int N, T alpha, T beta,
                   const T *a, int lda, const T *b, int ldb,
                   T *c, int ldc, char UPLO, int nb)
{
    const char NN[1] = {'N'};
    const char CN[1] = {'C'};

    for (int j = 0; j < N; ++j) {
        T *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (int i = ic; i < ic + ib; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (int i = ic; i < ic + ib; ++i) cj[i] *= beta;
    }

    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;

        if (UPLO == 'L') {
            for (int kc = jc + jb; kc < N; kc += nb) {
                const int kb = (N - kc < nb) ? (N - kc) : nb;
                ygemm_serial(NN, NN, &ib, &jb, &kb, &alpha,
                             &B_(ic, kc), &ldb, &A_(kc, jc), &lda, &ONE,
                             &C_(ic, jc), &ldc, 1, 1);
                ygemm_serial(NN, CN, &ib, &kb, &jb, &alpha,
                             &B_(ic, jc), &ldb, &A_(kc, jc), &lda, &ONE,
                             &C_(ic, kc), &ldc, 1, 1);
            }
        } else {
            for (int kc = 0; kc < jc; kc += nb) {
                const int kb = (jc - kc < nb) ? (jc - kc) : nb;
                ygemm_serial(NN, NN, &ib, &jb, &kb, &alpha,
                             &B_(ic, kc), &ldb, &A_(kc, jc), &lda, &ONE,
                             &C_(ic, jc), &ldc, 1, 1);
                ygemm_serial(NN, CN, &ib, &kb, &jb, &alpha,
                             &B_(ic, jc), &ldb, &A_(kc, jc), &lda, &ONE,
                             &C_(ic, kc), &ldc, 1, 1);
            }
        }

        hemm_diag_add_R(jc, jb, ic, ib, alpha, a, lda, b, ldb, c, ldc, UPLO);
    }
}

void yhemm_serial(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t side_len, size_t uplo_len)
{
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = (char)toupper((unsigned char)*side);
    const char UPLO = (char)toupper((unsigned char)*uplo);

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
        yhemm_beta_only(0, N, M, beta, c, ldc);
        return;
    }

    const int nb = yhemm_nb();

    if (SIDE == 'L' && M <= nb) {
        yhemm_L_singleblock(0, N, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }

    if (SIDE == 'L') {
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            yhemm_L_panel(jc, jb, M, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    } else {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            yhemm_R_panel(ic, ib, N, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    }
}

#undef A_
#undef B_
#undef C_
