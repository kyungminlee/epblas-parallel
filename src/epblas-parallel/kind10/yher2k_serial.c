/*
 * yher2k_serial — kind10 complex (_Complex long double) Hermitian rank-2k,
 * single-thread. This TU owns ALL of the yher2k math; yher2k_parallel.c only
 * orchestrates the diagonal-block loop across a team.
 *
 * C := alpha·A·Bᴴ + conj(alpha)·B·Aᴴ + beta·C  (TRANS='N'), or the
 * conjugate-transposed form (TRANS='C'). alpha is COMPLEX, beta is REAL;
 * the diagonal of C stays real. Blocked: scalar Hermitian rank-2 diagonal +
 * two ygemm trailing calls per block. The trailing updates run through
 * ygemm_serial (NOT ygemm_): when yher2k_block runs inside the team
 * yher2k_parallel.c opened, a nested ygemm team would trip the libgomp
 * barrier wedge (memory project-etrsm-omp4-wedge).
 */

#include "yher2k_kernel.h"
#include <stdlib.h>
#include <ctype.h>

typedef yher2k_TC TC;
typedef yher2k_TR TR;

static int g_yher2k_nb = 0;
int yher2k_nb(void) {
    if (g_yher2k_nb == 0) {
        g_yher2k_nb = 32;
        const char *s = getenv("YHER2K_NB");
        if (s && *s) { int v = atoi(s); if (v > 0) g_yher2k_nb = v; }
    }
    return g_yher2k_nb;
}

extern void ygemm_serial(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const TC *alpha,
    const TC *a, const int *lda,
    const TC *b, const int *ldb,
    const TC *beta,
    TC *c, const int *ldc,
    size_t transa_len, size_t transb_len);

static inline TC cconj(TC z) { return ~z; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const TC czero = 0.0L + 0.0Li;
static const TR rzero = 0.0L, rone = 1.0L;

/* Scalar Hermitian rank-2 diagonal block. Caller pre-scales β. */
static void her2k_diag_add(int jc, int jb, int K, TC alpha,
                           const TC *restrict a, int lda,
                           const TC *restrict b, int ldb,
                           TC *restrict c, int ldc,
                           char UPLO, char TR_c)
{
    if (TR_c == 'N') {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            for (int l = 0; l < K; ++l) {
                const TC t1 = alpha        * cconj(B_(j, l));
                const TC t2 = cconj(alpha) * cconj(A_(j, l));
                for (int i = i_lo; i < i_hi; ++i) {
                    if (i == j) cj[i] += __real__ (A_(i, l) * t1 + B_(i, l) * t2);
                    else        cj[i] += A_(i, l) * t1 + B_(i, l) * t2;
                }
            }
        }
    } else {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            const TC *Aj = a + (size_t)j * lda;
            const TC *Bj = b + (size_t)j * ldb;
            for (int i = i_lo; i < i_hi; ++i) {
                const TC *Ai = a + (size_t)i * lda;
                const TC *Bi = b + (size_t)i * ldb;
                TC s1 = czero;
                TC s2 = czero;
                for (int l = 0; l < K; ++l) {
                    s1 += cconj(Ai[l]) * Bj[l];
                    s2 += cconj(Bi[l]) * Aj[l];
                }
                if (i == j) cj[i] += __real__ (alpha * s1 + cconj(alpha) * s2);
                else        cj[i] += alpha * s1 + cconj(alpha) * s2;
            }
        }
    }
}

void yher2k_beta_scale(int j_start, int j_end, int N, TR beta,
                       TC *c, int ldc, char UPLO)
{
    for (int j = j_start; j < j_end; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        TC *cj = c + (size_t)j * ldc;
        if (beta == rzero) {
            for (int i = i_lo; i < i_hi; ++i) cj[i] = czero;
        } else if (beta != rone) {
            for (int i = i_lo; i < i_hi; ++i) {
                if (i == j) cj[i] = beta * __real__ cj[i];
                else        cj[i] = beta * cj[i];
            }
        } else {
            cj[j] = __real__ cj[j];
        }
    }
}

void yher2k_block(int jc, int jb, int N, int K, TC alpha, TR beta,
                  const TC *a, int lda, const TC *b, int ldb,
                  TC *c, int ldc, char UPLO, char TR_c)
{
    const TC cone       = 1.0L + 0.0Li;
    const TC alpha_conj = cconj(alpha);
    const char NN[1] = {'N'};
    const char CN[1] = {'C'};

    yher2k_beta_scale(jc, jc + jb, N, beta, c, ldc, UPLO);

    her2k_diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR_c);

    if (UPLO == 'L') {
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            const int j0 = jc + jb;
            if (TR_c == 'N') {
                ygemm_serial(NN, CN, &trailing, &jb, &K, &alpha,
                             &A_(j0, 0), &lda, &B_(jc, 0), &ldb,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
                ygemm_serial(NN, CN, &trailing, &jb, &K, &alpha_conj,
                             &B_(j0, 0), &ldb, &A_(jc, 0), &lda,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
            } else {
                ygemm_serial(CN, NN, &trailing, &jb, &K, &alpha,
                             &A_(0, j0), &lda, &B_(0, jc), &ldb,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
                ygemm_serial(CN, NN, &trailing, &jb, &K, &alpha_conj,
                             &B_(0, j0), &ldb, &A_(0, jc), &lda,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
            }
        }
    } else {
        if (jc > 0) {
            if (TR_c == 'N') {
                ygemm_serial(NN, CN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &B_(jc, 0), &ldb,
                             &cone, &C_(0, jc), &ldc, 1, 1);
                ygemm_serial(NN, CN, &jc, &jb, &K, &alpha_conj,
                             &B_(0, 0), &ldb, &A_(jc, 0), &lda,
                             &cone, &C_(0, jc), &ldc, 1, 1);
            } else {
                ygemm_serial(CN, NN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &B_(0, jc), &ldb,
                             &cone, &C_(0, jc), &ldc, 1, 1);
                ygemm_serial(CN, NN, &jc, &jb, &K, &alpha_conj,
                             &B_(0, 0), &ldb, &A_(0, jc), &lda,
                             &cone, &C_(0, jc), &ldc, 1, 1);
            }
        }
    }
}

void yher2k_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const TC *alpha_,
    const TC *a, const int *lda_,
    const TC *b, const int *ldb_,
    const TR *beta_,
    TC *c, const int *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const TC alpha = *alpha_;
    const TR beta  = *beta_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const char TR_c = (char)toupper((unsigned char)*trans);

    if (N == 0) return;

    if ((alpha == czero) || K == 0) {
        if (beta == rone) {
            for (int j = 0; j < N; ++j) c[(size_t)j * ldc + j] = __real__ c[(size_t)j * ldc + j];
            return;
        }
        yher2k_beta_scale(0, N, N, beta, c, ldc, UPLO);
        return;
    }

    const int nb = yher2k_nb();
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        yher2k_block(jc, jb, N, K, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TR_c);
    }
}

#undef A_
#undef B_
#undef C_
