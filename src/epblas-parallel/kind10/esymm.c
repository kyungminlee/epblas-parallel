/*
 * esymm — kind10 (REAL(KIND=10) / `long double`) symmetric matrix multiply.
 *
 * Blocked "read A_IK once, use twice" pattern (Netlib SYMM logic
 * lifted to block granularity): per off-diagonal block A_IK we fire
 * two consecutive egemm calls that both read the same A_IK from
 * still-warm L2 cache:
 *
 *   USE 1:  C(K, J) += alpha · A_IK^T · B(I, J)   (symmetric reflection)
 *   USE 2:  C(I, J) += alpha · A_IK   · B(K, J)   (direct row I)
 *
 * x87 long double has no SIMD, so peak per-core throughput is bound by
 * the x87 FPU. We size the macro-block so A_IK + an accumulator fit in
 * the per-core L2 between the two gemm calls (16 B/element → 128×128
 * block ≈ 256 KiB), and choose nb so threads each get a J panel.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define ESYMM_OMP_MIN 32

typedef long double T;

/* Pre-resolved ESYMM_NB override, populated at library load. 0 = no
 * override; symm_nb_pick uses its built-in adaptive sizing. */
static int g_esymm_nb_override = 0;

__attribute__((constructor))
static void esymm_init(void) {
    const char *s = getenv("ESYMM_NB");
    if (s && *s) {
        int v = atoi(s);
        if (v > 0) g_esymm_nb_override = v;
    }
}

/* nb ≈ N/nt for thread balance, capped at 128 so A_IK fits in L2
 * per core (16-byte long double × 128² = 256 KiB), floored at 64
 * to amortize egemm per-call setup. */
static int symm_nb_pick(int M_or_N, int nt) {
    if (g_esymm_nb_override > 0) return g_esymm_nb_override;
    int nt_eff = (nt > 0) ? nt : 1;
    int nb = (M_or_N + nt_eff - 1) / nt_eff;
    if (nb > 64) nb = 64;
    if (nb < 64) nb = 64;
    return nb;
}

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

/* Scalar diagonal-block symm for SIDE='L'.
 *
 * Mirrors the Netlib reference DSYMM access pattern: i is iterated in
 * the direction that lets the inner k loop read A column-major
 * stride-1. Reading A_(i, k) at fixed i across k would stride by lda
 * on every iteration; reading A_(k, i) walks rows of a single column
 * contiguously. A is symmetric so A_(k,i) == A_(i,k) numerically.
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
                T temp2 = 0.0L;
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
                T temp2 = 0.0L;
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
                if (t != 0.0L) for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
            for (int k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * A_(k, j);
                if (t != 0.0L) for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
        } else {
            for (int k = jc; k < j; ++k) {
                const T t = alpha * A_(k, j);
                if (t != 0.0L) for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
            for (int k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * A_(j, k);
                if (t != 0.0L) for (int i = ic; i < ic + ib; ++i) cj[i] += t * B_(i, k);
            }
        }
    }
}

void esymm_(
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

    const T zero = 0.0L, one = 1.0L;
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    if (alpha == zero) {
        if (beta == one) return;
#ifdef _OPENMP
        const int axis = (SIDE == 'L') ? N : M;
        const int use_omp = (axis >= ESYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T *cj = c + (size_t)j * ldc;
            if (beta == zero) for (int i = 0; i < M; ++i) cj[i] = zero;
            else              for (int i = 0; i < M; ++i) cj[i] *= beta;
        }
        return;
    }

    int nt = 1;
#ifdef _OPENMP
    nt = blas_omp_max_threads();
#endif
    const int nb = symm_nb_pick((SIDE == 'L') ? N : M, nt);

    /* SIDE='L' single-block fast path: inlines Netlib DSYMM directly,
     * bypassing the panel-framing wrapper and folding BETA into the
     * diagonal write. Only fires when M fits in one diagonal block
     * under the current nb (no egemm trailing updates would fire). */
    if (SIDE == 'L' && M <= nb) {
#ifdef _OPENMP
        const int use_omp = (N >= ESYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T *cj = c + (size_t)j * ldc;
            const T *bj = b + (size_t)j * ldb;
            if (UPLO == 'L') {
                for (int i = M - 1; i >= 0; --i) {
                    const T temp1 = alpha * bj[i];
                    T temp2 = 0.0L;
                    const T *ai = &A_(0, i);
                    for (int k = i + 1; k < M; ++k) {
                        cj[k]  += temp1 * ai[k];
                        temp2  += bj[k] * ai[k];
                    }
                    const T diag = temp1 * ai[i] + alpha * temp2;
                    if (beta == zero)     cj[i] = diag;
                    else if (beta == one) cj[i] += diag;
                    else                  cj[i] = beta * cj[i] + diag;
                }
            } else {
                for (int i = 0; i < M; ++i) {
                    const T temp1 = alpha * bj[i];
                    T temp2 = 0.0L;
                    const T *ai = &A_(0, i);
                    for (int k = 0; k < i; ++k) {
                        cj[k]  += temp1 * ai[k];
                        temp2  += bj[k] * ai[k];
                    }
                    const T diag = temp1 * ai[i] + alpha * temp2;
                    if (beta == zero)     cj[i] = diag;
                    else if (beta == one) cj[i] += diag;
                    else                  cj[i] = beta * cj[i] + diag;
                }
            }
        }
        return;
    }

    if (SIDE == 'L') {
#ifdef _OPENMP
        const int use_omp = (N >= ESYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;

            for (int j = jc; j < jc + jb; ++j) {
                T *cj = c + (size_t)j * ldc;
                if (beta == zero)      for (int i = 0; i < M; ++i) cj[i]  = zero;
                else if (beta != one)  for (int i = 0; i < M; ++i) cj[i] *= beta;
            }

            for (int ic = 0; ic < M; ic += nb) {
                const int ib = (M - ic < nb) ? (M - ic) : nb;

                if (UPLO == 'L') {
                    for (int kc = 0; kc < ic; kc += nb) {
                        const int kb = (ic - kc < nb) ? (ic - kc) : nb;
                        /* USE 1: C(K, J) += alpha · A_IK^T · B(I, J) */
                        egemm_(TN, NN, &kb, &jb, &ib, &alpha,
                               &A_(ic, kc), &lda, &B_(ic, jc), &ldb, &one,
                               &C_(kc, jc), &ldc, 1, 1);
                        /* USE 2: C(I, J) += alpha · A_IK   · B(K, J) */
                        egemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &A_(ic, kc), &lda, &B_(kc, jc), &ldb, &one,
                               &C_(ic, jc), &ldc, 1, 1);
                    }
                } else {
                    for (int kc = ic + ib; kc < M; kc += nb) {
                        const int kb = (M - kc < nb) ? (M - kc) : nb;
                        egemm_(TN, NN, &kb, &jb, &ib, &alpha,
                               &A_(ic, kc), &lda, &B_(ic, jc), &ldb, &one,
                               &C_(kc, jc), &ldc, 1, 1);
                        egemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &A_(ic, kc), &lda, &B_(kc, jc), &ldb, &one,
                               &C_(ic, jc), &ldc, 1, 1);
                    }
                }

                symm_diag_add_L(ic, ib, jc, jb, alpha, a, lda, b, ldb, c, ldc, UPLO);
            }
        }
    } else {  /* SIDE = 'R' */
#ifdef _OPENMP
        const int use_omp = (M >= ESYMM_OMP_MIN && nt > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;

            for (int j = 0; j < N; ++j) {
                T *cj = c + (size_t)j * ldc;
                if (beta == zero)      for (int i = ic; i < ic + ib; ++i) cj[i]  = zero;
                else if (beta != one)  for (int i = ic; i < ic + ib; ++i) cj[i] *= beta;
            }

            for (int jc = 0; jc < N; jc += nb) {
                const int jb = (N - jc < nb) ? (N - jc) : nb;

                if (UPLO == 'L') {
                    for (int kc = jc + jb; kc < N; kc += nb) {
                        const int kb = (N - kc < nb) ? (N - kc) : nb;
                        egemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &B_(ic, kc), &ldb, &A_(kc, jc), &lda, &one,
                               &C_(ic, jc), &ldc, 1, 1);
                        egemm_(NN, TN, &ib, &kb, &jb, &alpha,
                               &B_(ic, jc), &ldb, &A_(kc, jc), &lda, &one,
                               &C_(ic, kc), &ldc, 1, 1);
                    }
                } else {
                    for (int kc = 0; kc < jc; kc += nb) {
                        const int kb = (jc - kc < nb) ? (jc - kc) : nb;
                        egemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &B_(ic, kc), &ldb, &A_(kc, jc), &lda, &one,
                               &C_(ic, jc), &ldc, 1, 1);
                        egemm_(NN, TN, &ib, &kb, &jb, &alpha,
                               &B_(ic, jc), &ldb, &A_(kc, jc), &lda, &one,
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
