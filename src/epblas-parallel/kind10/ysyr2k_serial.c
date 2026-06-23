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
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef ysyr2k_TC TC;

ptrdiff_t ysyr2k_nb(void) { return 32; }

extern void ygemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const TC *alpha,
    const TC *a, ptrdiff_t lda,
    const TC *b, ptrdiff_t ldb,
    const TC *beta,
    TC *c, ptrdiff_t ldc);

static const TC ZERO = 0.0L + 0.0Li;
static const TC ONE  = 1.0L + 0.0Li;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* TRANS='T' diagonal-block update (dot form: each C element accumulates in
 * a register over the K axis). Only the Trans path uses this; the NoTrans
 * path runs a flat per-column rank-1 sweep inline in ysyr2k_block. */
static void syr2k_diag_add_t(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t k, TC alpha,
                             const TC *restrict a, ptrdiff_t lda,
                             const TC *restrict b, ptrdiff_t ldb,
                             TC *restrict c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
        const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
        TC *cj = c + (size_t)j * ldc;
        const TC *Aj = a + (size_t)j * lda;
        const TC *Bj = b + (size_t)j * ldb;
        for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
            const TC *Ai = a + (size_t)i * lda;
            const TC *Bi = b + (size_t)i * ldb;
            TC s = ZERO;
            for (ptrdiff_t l = 0; l < k; ++l) s += Ai[l] * Bj[l] + Bi[l] * Aj[l];
            cj[i] += alpha * s;
        }
    }
}

void ysyr2k_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, TC beta,
                       TC *c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
        TC *cj = c + (size_t)j * ldc;
        if (beta == ZERO) for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = ZERO;
        else              for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }
}

void ysyr2k_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t n, ptrdiff_t k, TC alpha, TC beta,
                  const TC *a, ptrdiff_t lda, const TC *b, ptrdiff_t ldb,
                  TC *c, ptrdiff_t ldc, char UPLO, char TRANS)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
        TC *cj = c + (size_t)j * ldc;
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
    if (TRANS == 'N') {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
            const ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
            TC *cj = c + (size_t)j * ldc;
            for (ptrdiff_t l = 0; l < k; ++l) {
                const TC t1 = alpha * A_(j, l);
                const TC t2 = alpha * B_(j, l);
                const TC *al = a + (size_t)l * lda, *bl = b + (size_t)l * ldb;
                for (ptrdiff_t i = i_lo; i < i_hi; ++i)
                    cj[i] += bl[i] * t1 + al[i] * t2;
            }
        }
        return;
    }

    syr2k_diag_add_t(jc, jb, k, alpha, a, lda, b, ldb, c, ldc, UPLO);

    if (UPLO == 'L') {
        const ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            const ptrdiff_t j0 = jc + jb;
            ygemm_serial('T', 'N', trailing, jb, k, &alpha,
                         &A_(0, j0), lda, &B_(0, jc), ldb, &ONE,
                         &C_(j0, jc), ldc);
            ygemm_serial('T', 'N', trailing, jb, k, &alpha,
                         &B_(0, j0), ldb, &A_(0, jc), lda, &ONE,
                         &C_(j0, jc), ldc);
        }
    } else {
        if (jc > 0) {
            ygemm_serial('T', 'N', jc, jb, k, &alpha,
                         &A_(0, 0), lda, &B_(0, jc), ldb, &ONE,
                         &C_(0, jc), ldc);
            ygemm_serial('T', 'N', jc, jb, k, &alpha,
                         &B_(0, 0), ldb, &A_(0, jc), lda, &ONE,
                         &C_(0, jc), ldc);
        }
    }
}

void ysyr2k_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TC *alpha_,
    const TC *a, ptrdiff_t lda,
    const TC *b, ptrdiff_t ldb,
    const TC *beta_,
    TC *c, ptrdiff_t ldc)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    char TRANS = blas_up(trans);
    if (TRANS == 'C') TRANS = 'T';

    if (n == 0) return;

    if (alpha == ZERO || k == 0) {
        if (beta == ONE) return;
        ysyr2k_beta_scale(0, n, n, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = ysyr2k_nb();
    for (ptrdiff_t jc = 0; jc < n; jc += nb) {
        const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        ysyr2k_block(jc, jb, n, k, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, TRANS);
    }
}

#undef A_
#undef B_
#undef C_
