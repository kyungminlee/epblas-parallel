/*
 * yhemm — kind10 complex (`_Complex long double`) Hermitian matrix mul.
 *
 * Same "read A_IK once, use twice" blocked structure as ysymm, but
 * USE 1's reflection gemm uses 'C' (conjugate transpose) and the
 * scalar diagonal-block kernel keeps the diagonal real per Hermitian
 * convention.
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YHEMM_OMP_MIN 32

typedef _Complex long double T;

static int g_yhemm_nb_override = 0;

__attribute__((constructor))
static void yhemm_init(void) {
    const char *s = getenv("YHEMM_NB");
    if (s && *s) {
        int v = atoi(s);
        if (v > 0) g_yhemm_nb_override = v;
    }
}

/* Cap=32 chosen to match ysymm: small blocks fragment the diagonal
 * kernel work so that ygemm picks up the off-diagonal trailing
 * updates aggressively. Complex long-double ops are heavy enough
 * that ygemm tolerates small K. */
static int hemm_nb_pick(int M_or_N, int nt) {
    if (g_yhemm_nb_override > 0) return g_yhemm_nb_override;
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

static inline T cconj(T z) { return ~z; }

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* Scalar Hermitian diagonal block, SIDE='L'.
 *
 * Mirrors the Netlib reference ZHEMM access pattern: i is iterated in
 * the direction that lets the inner k loop read A column-major
 * stride-1. The Hermitian conjugate moves from the cj[k] update onto
 * the temp2 accumulation (since the stored half flips when iteration
 * direction flips):
 *
 *   UPLO='L': i backward, k = i+1..ic+ib-1, A_(k,i) in lower (k>i,
 *             stored). cj[k] gets direct A; temp2 accumulates conj(A).
 *   UPLO='U': i forward,  k = ic..i-1,      A_(k,i) in upper (k<i,
 *             stored). Same direct-vs-conj split.
 */
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
                /* A_sym(k, j) = conj(A_stored(j, k)) for k<j (lower). */
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

void yhemm_(
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
    const char CN[1] = {'C'};

    if (alpha == ZERO) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const int axis = (SIDE == 'L') ? N : M;
        const int use_omp = (axis >= YHEMM_OMP_MIN && blas_omp_max_threads() > 1);
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
    const int nb = hemm_nb_pick((SIDE == 'L') ? N : M, nt);

    /* SIDE='L' single-block fast path: inlines Netlib ZHEMM directly,
     * bypassing panel framing and folding BETA into the diagonal
     * write. Only fires when M fits in one diagonal block. */
    if (SIDE == 'L' && M <= nb) {
#ifdef _OPENMP
        const int use_omp = (N >= YHEMM_OMP_MIN && nt > 1);
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
        return;
    }

    if (SIDE == 'L') {
#ifdef _OPENMP
        const int use_omp = (N >= YHEMM_OMP_MIN && nt > 1);
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
                        /* USE 1: conj-transpose of A_IK contributes to row K. */
                        ygemm_(CN, NN, &kb, &jb, &ib, &alpha,
                               &A_(ic, kc), &lda, &B_(ic, jc), &ldb, &ONE,
                               &C_(kc, jc), &ldc, 1, 1);
                        /* USE 2: A_IK directly contributes to row I. */
                        ygemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &A_(ic, kc), &lda, &B_(kc, jc), &ldb, &ONE,
                               &C_(ic, jc), &ldc, 1, 1);
                    }
                } else {
                    for (int kc = ic + ib; kc < M; kc += nb) {
                        const int kb = (M - kc < nb) ? (M - kc) : nb;
                        ygemm_(CN, NN, &kb, &jb, &ib, &alpha,
                               &A_(ic, kc), &lda, &B_(ic, jc), &ldb, &ONE,
                               &C_(kc, jc), &ldc, 1, 1);
                        ygemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &A_(ic, kc), &lda, &B_(kc, jc), &ldb, &ONE,
                               &C_(ic, jc), &ldc, 1, 1);
                    }
                }

                hemm_diag_add_L(ic, ib, jc, jb, alpha, a, lda, b, ldb, c, ldc, UPLO);
            }
        }
    } else {
#ifdef _OPENMP
        const int use_omp = (M >= YHEMM_OMP_MIN && nt > 1);
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
                    /* For SIDE='R' UPLO='L': A_(kc, jc) block is lower stored when kc > jc.
                     * Its sym reflection at (jc, kc) = conj(A_(kc, jc))^T = A_(kc, jc)^H. */
                    for (int kc = jc + jb; kc < N; kc += nb) {
                        const int kb = (N - kc < nb) ? (N - kc) : nb;
                        /* USE 1 (uses A_(kc, jc) directly via 'N'): C(:, jc..jc+jb) += alpha · B(:, kc..) · A(kc, jc) */
                        ygemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &B_(ic, kc), &ldb, &A_(kc, jc), &lda, &ONE,
                               &C_(ic, jc), &ldc, 1, 1);
                        /* USE 2 (reflection via 'C'): C(:, kc..kc+kb) += alpha · B(:, jc..) · A(kc, jc)^H */
                        ygemm_(NN, CN, &ib, &kb, &jb, &alpha,
                               &B_(ic, jc), &ldb, &A_(kc, jc), &lda, &ONE,
                               &C_(ic, kc), &ldc, 1, 1);
                    }
                } else {
                    for (int kc = 0; kc < jc; kc += nb) {
                        const int kb = (jc - kc < nb) ? (jc - kc) : nb;
                        ygemm_(NN, NN, &ib, &jb, &kb, &alpha,
                               &B_(ic, kc), &ldb, &A_(kc, jc), &lda, &ONE,
                               &C_(ic, jc), &ldc, 1, 1);
                        ygemm_(NN, CN, &ib, &kb, &jb, &alpha,
                               &B_(ic, jc), &ldb, &A_(kc, jc), &lda, &ONE,
                               &C_(ic, kc), &ldc, 1, 1);
                    }
                }

                hemm_diag_add_R(jc, jb, ic, ib, alpha, a, lda, b, ldb, c, ldc, UPLO);
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
