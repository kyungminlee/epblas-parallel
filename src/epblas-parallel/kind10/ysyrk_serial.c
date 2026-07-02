/*
 * ysyrk_serial — kind10 complex (_Complex long double) symmetric rank-k,
 * single-thread. This TU owns ALL of the ysyrk math; ysyrk_parallel.c only
 * orchestrates the diagonal-block loop across a team.
 *
 * TRANS ∈ {N, T}. Complex syrk does NOT conjugate (see yherk). Blocked:
 * scalar diagonal + ygemm trailing. The trailing update runs through
 * ygemm_serial (NOT ygemm_): when ysyrk_block runs inside the team
 * ysyrk_parallel.c opened, a nested ygemm team would trip the libgomp
 * barrier wedge.
 */

#include "ysyrk_kernel.h"
#include "../common/blas_char.h"
#include <stdlib.h>
#include <ctype.h>
#include <stddef.h>

typedef ysyrk_TC TC;

ptrdiff_t ysyrk_nb(void) { return 32; }

extern void ygemm_serial(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const TC *alpha,
    const TC *a, ptrdiff_t lda,
    const TC *b, ptrdiff_t ldb,
    const TC *beta,
    TC *c, ptrdiff_t ldc);

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const TC ZERO = 0.0L + 0.0Li;
static const TC ONE  = 1.0L + 0.0Li;

static void syrk_diag_add(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t k, TC alpha,
                          const TC *restrict a, ptrdiff_t lda,
                          TC *restrict c, ptrdiff_t ldc,
                          char UPLO, char TRANS)
{
    if (TRANS == 'N') {
        for (ptrdiff_t j = jc; j < jc + jb; ++j) {
            const ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + (size_t)j * ldc;
            for (ptrdiff_t l = 0; l < k; ++l) {
                const TC ajl = A_(j, l);
                if (ajl != ZERO) {
                    const TC t = alpha * ajl;
                    for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] += t * A_(i, l);
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
                TC s = ZERO;
                for (ptrdiff_t l = 0; l < k; ++l) s += Ai[l] * Aj[l];
                cj[i] += alpha * s;
            }
        }
    }
}

/* Full TRANS column update — netlib's single-pass unpacked stride-1 column dot
 * C(i,j) += alpha·Σ_l A(l,i)·A(l,j) over the WHOLE uplo range of each column,
 * diagonal and off-diagonal together (Upper: i∈[0,j]; Lower: i∈[j,n)). Both A
 * columns are contiguous (Trans ⇒ A is K×N), so the dot is cache-friendly and
 * skips the pack/microkernel scaffolding.
 *
 * Merging the diagonal triangle into this pass (vs. a separate syrk_diag_add
 * call over [jc,j] plus an off-diagonal call over the rest) is what matches
 * netlib's memory pattern: Aj = A(:,j) is read ONCE per column and streamed
 * across the full i-range, instead of the K-long Aj vector being re-read by two
 * sub-passes. That redundant Aj re-read was the entire Upper-Trans par/mig ~1.03
 * gap — netlib's single-pass Upper streams each Aj once and beat par's split;
 * par's per-dot was already optimal (decomposed, register-resident) so U≈L in
 * absolute terms (par/ob ties), the flag was purely netlib-U being faster than
 * netlib-L while par was uniform. Bit-identical to the old split: each C(i,j) is
 * still written exactly once and the per-element += is order-independent.
 *
 * The complex MAC is decomposed into scalar real/imag long-double accumulators
 * over A reinterpreted as 2·K interleaved reals (the yerot/ygemv x87 trick,
 * trigger 8): a `_Complex` `s += Ai[l]*Aj[l]` makes gcc-15 reload each operand
 * for both the real and imag sub-products (it can't prove Ai≠Aj≠cj don't
 * alias), spilling the fp80 stack. Hoisting air/aii/ajr/aji into locals forces
 * one load each and keeps the two accumulators (sr, si — the natural re/im parts
 * a complex MAC already computes, NOT a latency-hiding 2nd chain, so no
 * trigger-6 spill) register-resident, matching gfortran.
 *
 * beta is FUSED into the store — C(i,j) = alpha·s + beta·C(i,j) in a single
 * pass — exactly as netlib does, instead of a separate beta prescale pass over
 * C followed by a += here. That prescale (only live for beta∉{0,1}, and the
 * bench uses beta=0.3) was par's fixed per-call overhead: it re-reads and
 * re-writes the whole output triangle, cheap per element but a measurable
 * small-N tax netlib never pays (its Upper-Trans is a few % faster purely from
 * this single-pass fusion). Bit-identical to prescale-then-add: beta·C(i,j) is
 * rounded to fp80 either way (stored vs. register), and IEEE add commutes. */
static void syrk_trans_full(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t n, ptrdiff_t k,
                            TC alpha, TC beta, const TC *restrict a, ptrdiff_t lda,
                            TC *restrict c, ptrdiff_t ldc, char UPLO)
{
    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j     : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? n     : j + 1;
        TC *cj = c + (size_t)j * ldc;
        const long double *Ajr = (const long double *)(a + (size_t)j * lda);
        for (ptrdiff_t i = i_lo; i < i_hi; ++i) {
            const long double *Air = (const long double *)(a + (size_t)i * lda);
            long double sr = 0.0L, si = 0.0L;
            for (ptrdiff_t l = 0; l < k; ++l) {
                const long double air = Air[2*l + 0], aii = Air[2*l + 1];
                const long double ajr = Ajr[2*l + 0], aji = Ajr[2*l + 1];
                sr += air * ajr - aii * aji;
                si += air * aji + aii * ajr;
            }
            const TC s = alpha * __builtin_complex(sr, si);
            cj[i] = (beta == ZERO) ? s : s + beta * cj[i];
        }
    }
}

void ysyrk_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, TC beta,
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

void ysyrk_block(ptrdiff_t jc, ptrdiff_t jb, ptrdiff_t n, ptrdiff_t k, TC alpha, TC beta,
                 const TC *a, ptrdiff_t lda, TC *c, ptrdiff_t ldc, char UPLO, char TRANS)
{
    /* TRANS='T': syrk_trans_full does the WHOLE column (diagonal + trailing) in
     * one Aj-streaming pass with beta FUSED into the store — no separate beta
     * prescale, and do NOT call syrk_diag_add (the diagonal is already in the
     * full column, adding it again would double-count). TRANS='N' keeps the
     * beta-prescale + scalar-diagonal + ygemm-trailing split (blocked NoTrans). */
    if (TRANS != 'N') {
        syrk_trans_full(jc, jb, n, k, alpha, beta, a, lda, c, ldc, UPLO);
        return;
    }

    for (ptrdiff_t j = jc; j < jc + jb; ++j) {
        const ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
        TC *cj = c + (size_t)j * ldc;
        if (beta == ZERO)      for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i]  = ZERO;
        else if (beta != ONE)  for (ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] *= beta;
    }

    syrk_diag_add(jc, jb, k, alpha, a, lda, c, ldc, UPLO, TRANS);

    if (UPLO == 'L') {
        const ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            const ptrdiff_t j0 = jc + jb;
            ygemm_serial('N', 'T', trailing, jb, k, &alpha,
                         &A_(j0, 0), lda, &A_(jc, 0), lda,
                         &ONE, &C_(j0, jc), ldc);
        }
    } else {
        if (jc > 0) {
            ygemm_serial('N', 'T', jc, jb, k, &alpha,
                         &A_(0, 0), lda, &A_(jc, 0), lda,
                         &ONE, &C_(0, jc), ldc);
        }
    }
}

void ysyrk_serial(
    char uplo, char trans,
    ptrdiff_t n, ptrdiff_t k,
    const TC *alpha_,
    const TC *a, ptrdiff_t lda,
    const TC *beta_,
    TC *c, ptrdiff_t ldc)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char UPLO = blas_up(uplo);
    const char TRANS   = blas_up(trans);

    if (n == 0) return;

    if (alpha == ZERO || k == 0) {
        if (beta == ONE) return;
        ysyrk_beta_scale(0, n, n, beta, c, ldc, UPLO);
        return;
    }

    const ptrdiff_t nb = ysyrk_nb();
    for (ptrdiff_t jc = 0; jc < n; jc += nb) {
        const ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        ysyrk_block(jc, jb, n, k, alpha, beta, a, lda, c, ldc, UPLO, TRANS);
    }
}

#undef A_
#undef C_
