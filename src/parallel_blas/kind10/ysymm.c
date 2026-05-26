/*
 * ysymm — kind10 complex (`_Complex long double`) symmetric matrix mul.
 *
 * Blocked algorithm with the "read A_IK once, use it twice" pattern
 * lifted from the Netlib scalar reference. For each off-diagonal A
 * block at row-block I, col-block K (with I > K for UPLO='L'), we
 * fire TWO back-to-back ygemm calls reading the same A_IK block:
 *
 *   USE 1:  C(K, J) += alpha · A_IK^T · B(I, J)    (symmetric reflection)
 *   USE 2:  C(I, J) += alpha · A_IK   · B(K, J)    (direct row I contribution)
 *
 * A_IK is fetched from main memory once (call 1's pack), the second
 * call finds it hot in L2. That halves main-memory bandwidth on A
 * vs the prior "two stripes per block" approach and matches Netlib's
 * "load A(i,k) once, use temp1 and temp2 with it" pattern at block
 * granularity.
 *
 * OMP parallelism is over J column panels of B/C — each thread owns
 * a disjoint slice of C's columns, so the I/K loops run serial inside
 * the thread with no cross-thread races.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YSYMM_OMP_MIN 32

typedef _Complex long double T;

static int g_ysymm_nb_override = 0;

__attribute__((constructor))
static void ysymm_init(void) {
    const char *s = getenv("YSYMM_NB");
    if (s && *s) {
        int v = atoi(s);
        if (v > 0) g_ysymm_nb_override = v;
    }
}

/* Panel size: nb ≈ N/nt for thread balance, capped at ~256 so the
 * A_IK block stays cache-friendly between the two ygemm calls of
 * the "read once, use twice" inner pair. nb is also floored at 64
 * to keep per-call ygemm setup costs amortized. */
static int symm_nb_pick(int M_or_N, int nt) {
    if (g_ysymm_nb_override > 0) return g_ysymm_nb_override;
    int nt_eff = (nt > 0) ? nt : 1;
    int nb = (M_or_N + nt_eff - 1) / nt_eff;
    if (nb > 32) nb = 32;
    if (nb < 32) nb = 32;
    return nb;
}

extern void ygemm_(
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

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* Scalar diagonal-block symm for SIDE='L'.
 *
 * Mirrors the Netlib reference ZSYMM access pattern: i is iterated in
 * the direction that lets the inner k loop read A column-major
 * stride-1. Reading A_(i, k) at fixed i across k would stride by lda;
 * reading A_(k, i) walks rows of a single column contiguously. A is
 * complex symmetric so A_(k,i) == A_(i,k) numerically.
 *
 *   UPLO='L': iterate i backward, k = i+1..ic+ib-1, A_(k,i) — k>i in
 *             lower-stored half.
 *   UPLO='U': iterate i forward,  k = ic..i-1,      A_(k,i) — k<i in
 *             upper-stored half.
 */
static void symm_diag_add_L(int ic, int ib, int jc, int jb, T alpha,
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
                    temp2  += bj[k] * ai[k];
                }
                cj[i] += temp1 * ai[i] + alpha * temp2;
            }
        } else {
            for (int i = ic; i < ic + ib; ++i) {
                const T temp1 = alpha * bj[i];
                T temp2 = ZERO;
                const T *ai = &A_(0, i);
                for (int k = ic; k < i; ++k) {
                    cj[k]  += temp1 * ai[k];
                    temp2  += bj[k] * ai[k];
                }
                cj[i] += temp1 * ai[i] + alpha * temp2;
            }
        }
    }
}

/* Scalar diagonal-block symm for SIDE='R'. */
static void symm_diag_add_R(int jc, int jb, int ic, int ib, T alpha,
                            const T *restrict a, int lda,
                            const T *restrict b, int ldb,
                            T *restrict c, int ldc, char UPLO)
{
    for (int j = jc; j < jc + jb; ++j) {
        T *cj = c + (size_t)j * ldc;
        {
            const T t = alpha * A_(j, j);
            for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, j);
        }
        if (UPLO == 'L') {
            for (int k = jc; k < j; ++k) {
                const T t = alpha * A_(j, k);
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
                const T t = alpha * A_(j, k);
                if (t != ZERO) for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
        }
    }
}

void ysymm_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *restrict a, const int *lda_,
    const T *restrict b, const int *ldb_,
    const T *beta_,
    T *restrict c, const int *ldc_,
    size_t side_len, size_t uplo_len)
{
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = up(side);
    const char UPLO = up(uplo);

    if (M == 0 || N == 0) return;

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    if (alpha == ZERO) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const int axis = (SIDE == 'L') ? N : M;
        const int use_omp = (axis >= YSYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T *cj = c + (size_t)j * ldc;
            if (beta == ZERO) for (int i = 0; i < M; ++i) cj[i] = ZERO;
            else              for (int i = 0; i < M; ++i) cj[i] *= beta;
        }
        return;
    }

    int nt = 1;
#ifdef _OPENMP
    nt = blas_omp_max_threads();
#endif
    const int nb = symm_nb_pick((SIDE == 'L') ? N : M, nt);

    /* SIDE='L' single-block fast path: inlines Netlib ZSYMM directly,
     * bypassing the panel-framing wrapper and folding BETA into the
     * diagonal write. Only fires when M fits in one diagonal block
     * under the current nb (no ygemm trailing updates would fire). */
    if (SIDE == 'L' && M <= nb) {
#ifdef _OPENMP
        const int use_omp = (N >= YSYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T *cj = c + (size_t)j * ldc;
            const T *bj = b + (size_t)j * ldb;
            if (UPLO == 'L') {
                for (int i = M - 1; i >= 0; --i) {
                    const T temp1 = alpha * bj[i];
                    T temp2 = ZERO;
                    const T *ai = &A_(0, i);
                    for (int k = i + 1; k < M; ++k) {
                        cj[k]  += temp1 * ai[k];
                        temp2  += bj[k] * ai[k];
                    }
                    const T diag = temp1 * ai[i] + alpha * temp2;
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
                        temp2  += bj[k] * ai[k];
                    }
                    const T diag = temp1 * ai[i] + alpha * temp2;
                    if (beta == ZERO)     cj[i] = diag;
                    else if (beta == ONE) cj[i] += diag;
                    else                  cj[i] = beta * cj[i] + diag;
                }
            }
        }
        return;
    }

    if (SIDE == 'L') {
        /* Parallel over J column panels of C. Each thread runs the
         * full I/K loops serial on its J panel — A_IK loaded once per
         * (I, K) pair, used by both ygemm calls (L2 hit on call 2). */
#ifdef _OPENMP
        const int use_omp = (N >= YSYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;

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
                        /* A(ic..ic+ib, kc..kc+kb) lower stored.
                         * USE 1: C(K, J) += alpha · A^T · B(I, J) */
                        ygemm_(TN, NN, &kb, &jb, &ib, &alpha,
                               &A_(ic, kc), &lda, &B_(ic, jc), &ldb, &ONE,
                               &C_(kc, jc), &ldc, 1, 1);
                        /* USE 2: C(I, J) += alpha · A · B(K, J) */
                        ygemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &A_(ic, kc), &lda, &B_(kc, jc), &ldb, &ONE,
                               &C_(ic, jc), &ldc, 1, 1);
                    }
                } else {  /* UPLO == 'U' */
                    for (int kc = ic + ib; kc < M; kc += nb) {
                        const int kb = (M - kc < nb) ? (M - kc) : nb;
                        ygemm_(TN, NN, &kb, &jb, &ib, &alpha,
                               &A_(ic, kc), &lda, &B_(ic, jc), &ldb, &ONE,
                               &C_(kc, jc), &ldc, 1, 1);
                        ygemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &A_(ic, kc), &lda, &B_(kc, jc), &ldb, &ONE,
                               &C_(ic, jc), &ldc, 1, 1);
                    }
                }

                symm_diag_add_L(ic, ib, jc, jb, alpha, a, lda, b, ldb, c, ldc, UPLO);
            }
        }
    } else {  /* SIDE = 'R' */
#ifdef _OPENMP
        const int use_omp = (M >= YSYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;

            for (int j = 0; j < N; ++j) {
                T *cj = c + (size_t)j * ldc;
                if (beta == ZERO)      for (int i = ic; i < ic + ib; ++i) cj[i]  = ZERO;
                else if (beta != ONE)  for (int i = ic; i < ic + ib; ++i) cj[i] *= beta;
            }

            for (int jc = 0; jc < N; jc += nb) {
                const int jb = (N - jc < nb) ? (N - jc) : nb;

                if (UPLO == 'L') {
                    /* A block at row block kc..kc+kb (kc > jc), col block jc..jc+jb — lower stored. */
                    for (int kc = jc + jb; kc < N; kc += nb) {
                        const int kb = (N - kc < nb) ? (N - kc) : nb;
                        /* USE 1: C(:, jc..jc+jb) += alpha · B(:, kc..kc+kb) · A(kc, jc) */
                        ygemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &B_(ic, kc), &ldb, &A_(kc, jc), &lda, &ONE,
                               &C_(ic, jc), &ldc, 1, 1);
                        /* USE 2: C(:, kc..kc+kb) += alpha · B(:, jc..jc+jb) · A(kc, jc)^T */
                        ygemm_(NN, TN, &ib, &kb, &jb, &alpha,
                               &B_(ic, jc), &ldb, &A_(kc, jc), &lda, &ONE,
                               &C_(ic, kc), &ldc, 1, 1);
                    }
                } else {  /* UPLO == 'U' */
                    for (int kc = 0; kc < jc; kc += nb) {
                        const int kb = (jc - kc < nb) ? (jc - kc) : nb;
                        /* A block at row block kc..kc+kb, col block jc..jc+jb — upper stored (kc < jc). */
                        ygemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &B_(ic, kc), &ldb, &A_(kc, jc), &lda, &ONE,
                               &C_(ic, jc), &ldc, 1, 1);
                        ygemm_(NN, TN, &ib, &kb, &jb, &alpha,
                               &B_(ic, jc), &ldb, &A_(kc, jc), &lda, &ONE,
                               &C_(ic, kc), &ldc, 1, 1);
                    }
                }

                symm_diag_add_R(jc, jb, ic, ib, alpha, a, lda, b, ldb, c, ldc, UPLO);
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
