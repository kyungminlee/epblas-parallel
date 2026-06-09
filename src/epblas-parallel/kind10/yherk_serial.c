/*
 * yherk_serial — kind10 complex (_Complex long double) Hermitian rank-k,
 * single-thread. This TU owns ALL of the yherk math; yherk_parallel.c only
 * orchestrates the diagonal-block loop across a team.
 *
 * TRANS ∈ {N, C}. alpha/beta are REAL; the diagonal of C stays real.
 * Blocked: beta pre-scale + scalar Hermitian diagonal add + ygemm trailing
 * with conjugate transpose. The trailing update runs through ygemm_serial
 * (NOT ygemm_): when yherk_block runs inside the team yherk_parallel.c
 * opened, a nested ygemm team would trip the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include "yherk_kernel.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef yherk_TC TC;
typedef yherk_TR TR;

ptrdiff_t yherk_nb(void) { return 32; }

extern void ygemm_serial(
    const char *transa, const char *transb,
    const ptrdiff_t *m, const ptrdiff_t *n, const ptrdiff_t *k,
    const TC *alpha,
    const TC *a, const ptrdiff_t *lda,
    const TC *b, const ptrdiff_t *ldb,
    const TC *beta,
    TC *c, const ptrdiff_t *ldc,
    size_t transa_len, size_t transb_len);

static inline TC cconj(TC z) { return ~z; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const TC czero = 0.0L + 0.0Li;
static const TR rzero = 0.0L, rone = 1.0L;

/* Diagonal jb×jb block rank-k add, keeping diagonal entries real.
 * No beta scaling (caller pre-scales). */
static void herk_diag_add(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t K, TR alpha,
                          const TC *restrict a, ptrdiff_t lda,
                          TC *restrict c, ptrdiff_t ldc,
                          char UPLO, char TR_c)
{
    if (TR_c == 'N') {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            for (ptrdiff_t l = 0; l < K; ++l) {
                const TC ajl = A_(j, l);
                if (ajl != czero) {
                    const TC t = alpha * cconj(ajl);
                    for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                        if (i == j) cj[i] += __real__ (t * A_(i, l));
                        else        cj[i] += t * A_(i, l);
                    }
                }
            }
        }
    } else {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            const TC *Aj = a + (size_t)j * lda;
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                const TC *Ai = a + (size_t)i * lda;
                TC s = czero;
                for (ptrdiff_t l = 0; l < K; ++l) s += cconj(Ai[l]) * Aj[l];
                if (i == j) cj[i] += alpha * __real__ s;
                else        cj[i] += alpha * s;
            }
        }
    }
}

void yherk_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t N, TR beta,
                      TC *c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
        TC *cj = c + (size_t)j * ldc;
        if (beta == rzero) {
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = czero;
        } else if (beta != rone) {
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                if (i == j) cj[i] = beta * __real__ cj[i];
                else        cj[i] = beta * cj[i];
            }
        } else {
            cj[j] = __real__ cj[j];
        }
    }
}

void yherk_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t N, ptrdiff_t K, TR alpha, TR beta,
                 const TC *a, ptrdiff_t lda, TC *c, ptrdiff_t ldc, char UPLO, char TR_c)
{
    const TC cone    = 1.0L + 0.0Li;
    const TC alpha_c = alpha + 0.0Li;
    const char NN[1] = {'N'};
    const char CN[1] = {'C'};

    yherk_beta_scale(jc, jc + jb, N, beta, c, ldc, UPLO);

    herk_diag_add(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR_c);

    if (UPLO == 'L') {
        const ptrdiff_t trailing = N - jc - jb;
        if (trailing > 0) {
            const ptrdiff_t j0 = jc + jb;
            if (TR_c == 'N') {
                ygemm_serial(NN, CN, &trailing, &jb, &K, &alpha_c,
                             &A_(j0, 0), &lda, &A_(jc, 0), &lda,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
            } else {
                ygemm_serial(CN, NN, &trailing, &jb, &K, &alpha_c,
                             &A_(0, j0), &lda, &A_(0, jc), &lda,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
            }
        }
    } else {
        if (jc > 0) {
            if (TR_c == 'N') {
                ygemm_serial(NN, CN, &jc, &jb, &K, &alpha_c,
                             &A_(0, 0), &lda, &A_(jc, 0), &lda,
                             &cone, &C_(0, jc), &ldc, 1, 1);
            } else {
                ygemm_serial(CN, NN, &jc, &jb, &K, &alpha_c,
                             &A_(0, 0), &lda, &A_(0, jc), &lda,
                             &cone, &C_(0, jc), &ldc, 1, 1);
            }
        }
    }
}

void yherk_serial(
    const char *uplo, const char *trans,
    const ptrdiff_t *n_, const ptrdiff_t *k_,
    const TR *alpha_,
    const TC *a, const ptrdiff_t *lda_,
    const TR *beta_,
    TC *c, const ptrdiff_t *ldc_,
    size_t uplo_len, size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const ptrdiff_t N = *n_, K = *k_;
    const ptrdiff_t lda = *lda_, ldc = *ldc_;
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = (char)toupper((unsigned char)*uplo);
    const char TR_c = (char)toupper((unsigned char)*trans);

    if (N == 0) return;

    if (alpha == rzero || K == 0) {
        if (beta == rone) {
            for (ptrdiff_t j = 0; j < N; ++j) c[(size_t)j * ldc + j] = __real__ c[(size_t)j * ldc + j];
            return;
        }
        yherk_beta_scale(0, N, N, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = yherk_nb();
    for (ptrdiff_t jc = 0; jc < N; jc += nb) {
        const ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        yherk_block(jc, jb, N, K, alpha, beta, a, lda, c, ldc, UPLO, TR_c);
    }
}

#undef A_
#undef C_
