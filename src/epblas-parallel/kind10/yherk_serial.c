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
    char transa, char transb,
    ptrdiff_t M, ptrdiff_t N, ptrdiff_t K,
    const TC *alpha,
    const TC *a, ptrdiff_t lda,
    const TC *b, ptrdiff_t ldb,
    const TC *beta,
    TC *c, ptrdiff_t ldc);

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
        /* Conjugate dot, scalar re/im chains (bit-identical to
         * cconj(Ai[l])*Aj[l]; folds the conj sign into the products so
         * the imag chain has no fchs on its critical path — same ~8% win
         * as ygemm_tn_core's 'C' path; this diagonal block is ~half the
         * work at the small N where yherk UC/LC trails gfortran). */
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            const long double *Aj = (const long double *)(a + (size_t)j * lda);
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                const long double *Ai = (const long double *)(a + (size_t)i * lda);
                long double sr = 0.0L, si = 0.0L;
                for (ptrdiff_t l = 0; l < K; ++l) {
                    const long double ar = Ai[2*l], aim = Ai[2*l+1];
                    const long double br = Aj[2*l], bim = Aj[2*l+1];
                    sr += ar * br + aim * bim;
                    si += ar * bim - aim * br;
                }
                const TC s = sr + si * 1.0Li;
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
    yherk_beta_scale(jc, jc + jb, N, beta, c, ldc, UPLO);

    herk_diag_add(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR_c);

    if (UPLO == 'L') {
        const ptrdiff_t trailing = N - jc - jb;
        if (trailing > 0) {
            const ptrdiff_t j0 = jc + jb;
            if (TR_c == 'N') {
                ygemm_serial('N', 'C', trailing, jb, K, &alpha_c,
                             &A_(j0, 0), lda, &A_(jc, 0), lda,
                             &cone, &C_(j0, jc), ldc);
            } else {
                ygemm_serial('C', 'N', trailing, jb, K, &alpha_c,
                             &A_(0, j0), lda, &A_(0, jc), lda,
                             &cone, &C_(j0, jc), ldc);
            }
        }
    } else {
        if (jc > 0) {
            if (TR_c == 'N') {
                ygemm_serial('N', 'C', jc, jb, K, &alpha_c,
                             &A_(0, 0), lda, &A_(jc, 0), lda,
                             &cone, &C_(0, jc), ldc);
            } else {
                ygemm_serial('C', 'N', jc, jb, K, &alpha_c,
                             &A_(0, 0), lda, &A_(0, jc), lda,
                             &cone, &C_(0, jc), ldc);
            }
        }
    }
}

void yherk_serial(
    char uplo, char trans,
    ptrdiff_t N, ptrdiff_t K,
    const TR *alpha_,
    const TC *a, ptrdiff_t lda,
    const TR *beta_,
    TC *c, ptrdiff_t ldc)
{
    const TR alpha = *alpha_, beta = *beta_;
    const char UPLO = (char)toupper((unsigned char)uplo);
    const char TR_c = (char)toupper((unsigned char)trans);

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
