/*
 * ygemmtr_serial — kind10 complex (_Complex long double) triangular
 * GEMM-update, single-thread. This TU owns ALL of the ygemmtr math;
 * ygemmtr_parallel.c only orchestrates the column loop across a team.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only the UPLO triangle of C)
 *
 * `~z` is the GCC-extension complex conjugate.
 */

#include "ygemmtr_kernel.h"
#include "../common/blas_char.h"
#include <ctype.h>
#include <stdbool.h>
#include <stddef.h>

typedef ygemmtr_TC TC;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const TC zero = 0.0L + 0.0iL;
static const TC one  = 1.0L + 0.0iL;

void ygemmtr_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t n, bool upper,
                        TC beta, TC *c, ptrdiff_t ldc)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t is = upper ? 0 : j;
        const ptrdiff_t ie = upper ? (j + 1) : n;
        TC *cj = &C_(0, j);
        if (beta == zero) for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
        else              for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
    }
}

void ygemmtr_col(ptrdiff_t j, ptrdiff_t n, ptrdiff_t k, bool upper,
                 TC alpha, TC beta,
                 const TC *a, ptrdiff_t lda, const TC *b, ptrdiff_t ldb,
                 TC *c, ptrdiff_t ldc,
                 bool trans_a, bool conj_a, bool trans_b, bool conj_b)
{
    const ptrdiff_t is = upper ? 0 : j;
    const ptrdiff_t ie = upper ? (j + 1) : n;
    TC *cj = &C_(0, j);

    if (!trans_a) {
        if (beta == zero)      for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
        else if (beta != one)  for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
        /* K-unroll by 2 — expose two independent FMA chains per i to mask
         * x87 fmul latency. conj_b is hoisted out of the hot loop; a runtime
         * branch inside the K-unrolled body defeats gcc's scheduling for
         * this kind10 complex pattern. */
        ptrdiff_t l = 0;
        if (!trans_b) {
            for (; l + 1 < k; l += 2) {
                const TC t0 = alpha * B_(l,     j);
                const TC t1 = alpha * B_(l + 1, j);
                const TC *al0 = &A_(0, l);
                const TC *al1 = &A_(0, l + 1);
                for (ptrdiff_t i = is; i < ie; ++i)
                    cj[i] += t0 * al0[i] + t1 * al1[i];
            }
        } else if (!conj_b) {
            for (; l + 1 < k; l += 2) {
                const TC t0 = alpha * B_(j, l);
                const TC t1 = alpha * B_(j, l + 1);
                const TC *al0 = &A_(0, l);
                const TC *al1 = &A_(0, l + 1);
                for (ptrdiff_t i = is; i < ie; ++i)
                    cj[i] += t0 * al0[i] + t1 * al1[i];
            }
        } else {
            for (; l + 1 < k; l += 2) {
                const TC t0 = alpha * ~B_(j, l);
                const TC t1 = alpha * ~B_(j, l + 1);
                const TC *al0 = &A_(0, l);
                const TC *al1 = &A_(0, l + 1);
                for (ptrdiff_t i = is; i < ie; ++i)
                    cj[i] += t0 * al0[i] + t1 * al1[i];
            }
        }
        for (; l < k; ++l) {
            TC bl;
            if (!trans_b)      bl = B_(l, j);
            else if (!conj_b)  bl = B_(j, l);
            else               bl = ~B_(j, l);
            const TC t = alpha * bl;
            const TC *al = &A_(0, l);
            for (ptrdiff_t i = is; i < ie; ++i) cj[i] += t * al[i];
        }
    } else {
        for (ptrdiff_t i = is; i < ie; ++i) {
            TC s = zero;
            if (!trans_b) {
                if (!conj_a) for (ptrdiff_t l = 0; l < k; ++l) s += A_(l, i) * B_(l, j);
                else         for (ptrdiff_t l = 0; l < k; ++l) s += ~A_(l, i) * B_(l, j);
            } else if (!conj_b) {
                if (!conj_a) for (ptrdiff_t l = 0; l < k; ++l) s += A_(l, i) * B_(j, l);
                else         for (ptrdiff_t l = 0; l < k; ++l) s += ~A_(l, i) * B_(j, l);
            } else {
                if (!conj_a) for (ptrdiff_t l = 0; l < k; ++l) s += A_(l, i) * ~B_(j, l);
                else         for (ptrdiff_t l = 0; l < k; ++l) s += ~A_(l, i) * ~B_(j, l);
            }
            cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
        }
    }
}

void ygemmtr_serial(char uplo, char transa, char transb,
                    ptrdiff_t n, ptrdiff_t k,
                    const TC *alpha_,
                    const TC *a, ptrdiff_t lda,
                    const TC *b, ptrdiff_t ldb,
                    const TC *beta_,
                    TC *c, ptrdiff_t ldc)
{
    const TC alpha = *alpha_, beta = *beta_;
    const bool upper = (blas_up(uplo) == 'U');
    const char ta = blas_up(transa);
    const char tb = blas_up(transb);

    if (n <= 0) return;

    const bool conj_a = (ta == 'C');
    const bool conj_b = (tb == 'C');
    const bool trans_a = (ta != 'N');
    const bool trans_b = (tb != 'N');

    if (alpha == zero || k == 0) {
        if (beta == one) return;
        ygemmtr_beta_scale(0, n, n, upper, beta, c, ldc);
        return;
    }

    for (ptrdiff_t j = 0; j < n; ++j)
        ygemmtr_col(j, n, k, upper, alpha, beta, a, lda, b, ldb, c, ldc,
                    trans_a, conj_a, trans_b, conj_b);
}

#undef A_
#undef B_
#undef C_
