/*
 * yher2k_serial — kind10 complex (_Complex long double) Hermitian rank-2k,
 * single-thread. This TU owns ALL of the yher2k math; yher2k_parallel.c only
 * orchestrates the diagonal-block loop across a team.
 *
 * C := alpha·A·Bᴴ + conj(alpha)·B·Aᴴ + beta·C  (TRANS='N'), or the
 * conjugate-transposed form (TRANS='C'). alpha is COMPLEX, beta is REAL;
 * the diagonal of C stays real. Blocked: scalar Hermitian rank-2 diagonal +
 * two ygemm trailing calls per block. beta is FUSED into the block instead
 * of a separate prescale pass over the whole output panel: her2k_diag_add
 * carries beta for the diagonal block, and the trailing strip's beta rides
 * the FIRST ygemm call's own prepass (the second gets beta=1). (A
 * full-column scalar Trans pass — the ysyrk_block 553cc23 pattern — was
 * tried and REGRESSED ~15-35%: packed ygemm reuses each A/B column across
 * the panel; a per-(i,j) dot re-streams all of A and B per output column
 * and blows L2 at N≥128. Complex ≠ real here.) The trailing updates run
 * through ygemm_serial (NOT ygemm_): when yher2k_block runs inside the
 * team yher2k_parallel.c opened, a nested ygemm team would trip the
 * libgomp barrier wedge.
 */

#include "yher2k_kernel.h"
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef yher2k_TC TC;
typedef yher2k_TR TR;

ptrdiff_t yher2k_nb(void) { return 32; }

extern void ygemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const TC *alpha,
    const TC *a, ptrdiff_t lda,
    const TC *b, ptrdiff_t ldb,
    const TC *beta,
    TC *c, ptrdiff_t ldc);

static inline TC cconj(TC z) { return ~z; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const TC czero = 0.0L + 0.0Li;
static const TR rzero = 0.0L, rone = 1.0L;

/* Scalar Hermitian rank-2 diagonal block. beta is FUSED here (no separate
 * caller prescale): the 'N' arm scales the column's block range once right
 * before accumulating into it (same element order and case split as
 * yher2k_beta_scale restricted to [i_lo,i_hi), so bit-identical); the 'C'
 * arm folds beta into the store — beta·C(i,j) is rounded to fp80 either
 * way and IEEE add commutes, so bit-identical too. The i==j stores assign
 * REAL values, keeping the diagonal exactly real for every beta. */
static void her2k_diag_add(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t k, TC alpha, TR beta,
                           const TC *restrict a, ptrdiff_t lda,
                           const TC *restrict b, ptrdiff_t ldb,
                           TC *restrict c, ptrdiff_t ldc,
                           char UPLO, char TRANS)
{
    if (TRANS == 'N') {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
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
            for (ptrdiff_t l = 0; l < k; ++l) {
                const TC t1 = alpha        * cconj(B_(j, l));
                const TC t2 = cconj(alpha) * cconj(A_(j, l));
                for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                    if (i == j) cj[i] += __real__ (A_(i, l) * t1 + B_(i, l) * t2);
                    else        cj[i] += A_(i, l) * t1 + B_(i, l) * t2;
                }
            }
        }
    } else {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            const TC *Aj = a + (size_t)j * lda;
            const TC *Bj = b + (size_t)j * ldb;
            for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
                const TC *Ai = a + (size_t)i * lda;
                const TC *Bi = b + (size_t)i * ldb;
                TC s1 = czero;
                TC s2 = czero;
                for (ptrdiff_t l = 0; l < k; ++l) {
                    s1 += cconj(Ai[l]) * Bj[l];
                    s2 += cconj(Bi[l]) * Aj[l];
                }
                const TC v = alpha * s1 + cconj(alpha) * s2;
                if (i == j) {
                    const TR d = __real__ v;
                    cj[i] = (beta == rzero) ? d : d + beta * __real__ cj[i];
                } else {
                    cj[i] = (beta == rzero) ? v : v + beta * cj[i];
                }
            }
        }
    }
}

void yher2k_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, TR beta,
                       TC *c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
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

void yher2k_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t n, ptrdiff_t k, TC alpha, TR beta,
                  const TC *a, ptrdiff_t lda, const TC *b, ptrdiff_t ldb,
                  TC *c, ptrdiff_t ldc, char UPLO, char TRANS)
{
    const TC cone       = 1.0L + 0.0Li;
    const TC beta_c     = beta + 0.0Li;
    const TC alpha_conj = cconj(alpha);

    /* beta is fused: the diagonal block scales inside her2k_diag_add, the
     * trailing strip scales inside the FIRST ygemm's own beta prepass (the
     * second gemm of the pair accumulates with beta=1) — no separate
     * whole-panel prescale pass. The strip has no diagonal elements, so
     * the Hermitian real projection lives entirely in her2k_diag_add. */
    her2k_diag_add(jc, jb, k, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TRANS);

    if (UPLO == 'L') {
        const ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            const ptrdiff_t j0 = jc + jb;
            if (TRANS == 'N') {
                ygemm_serial('N', 'C', trailing, jb, k, &alpha,
                             &A_(j0, 0), lda, &B_(jc, 0), ldb,
                             &beta_c, &C_(j0, jc), ldc);
                ygemm_serial('N', 'C', trailing, jb, k, &alpha_conj,
                             &B_(j0, 0), ldb, &A_(jc, 0), lda,
                             &cone, &C_(j0, jc), ldc);
            } else {
                ygemm_serial('C', 'N', trailing, jb, k, &alpha,
                             &A_(0, j0), lda, &B_(0, jc), ldb,
                             &beta_c, &C_(j0, jc), ldc);
                ygemm_serial('C', 'N', trailing, jb, k, &alpha_conj,
                             &B_(0, j0), ldb, &A_(0, jc), lda,
                             &cone, &C_(j0, jc), ldc);
            }
        }
    } else {
        if (jc > 0) {
            if (TRANS == 'N') {
                ygemm_serial('N', 'C', jc, jb, k, &alpha,
                             &A_(0, 0), lda, &B_(jc, 0), ldb,
                             &beta_c, &C_(0, jc), ldc);
                ygemm_serial('N', 'C', jc, jb, k, &alpha_conj,
                             &B_(0, 0), ldb, &A_(jc, 0), lda,
                             &cone, &C_(0, jc), ldc);
            } else {
                ygemm_serial('C', 'N', jc, jb, k, &alpha,
                             &A_(0, 0), lda, &B_(0, jc), ldb,
                             &beta_c, &C_(0, jc), ldc);
                ygemm_serial('C', 'N', jc, jb, k, &alpha_conj,
                             &B_(0, 0), ldb, &A_(0, jc), lda,
                             &cone, &C_(0, jc), ldc);
            }
        }
    }
}

void yher2k_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TC *alpha_,
    const TC *a, ptrdiff_t lda,
    const TC *b, ptrdiff_t ldb,
    const TR *beta_,
    TC *c, ptrdiff_t ldc)
{
    const TC alpha = *alpha_;
    const TR beta  = *beta_;
    const char UPLO = blas_up(uplo);
    const char TRANS = blas_up(trans);

    if (n == 0) return;

    if ((alpha == czero) || k == 0) {
        if (beta == rone) {
            for (ptrdiff_t j = 0; j < n; ++j) c[(size_t)j * ldc + j] = __real__ c[(size_t)j * ldc + j];
            return;
        }
        yher2k_beta_scale(0, n, n, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = yher2k_nb();
    for (ptrdiff_t jc = 0; jc < n; jc += nb) {
        const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        yher2k_block(jc, jb, n, k, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TRANS);
    }
}

#undef A_
#undef B_
#undef C_
