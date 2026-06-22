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
#include <ctype.h>
#include <stddef.h>

typedef ygemmtr_T T;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

static const T zero = 0.0L + 0.0iL;
static const T one  = 1.0L + 0.0iL;

void ygemmtr_beta_scale(ptrdiff_t j_start, ptrdiff_t j_end, ptrdiff_t N, ptrdiff_t upper,
                        T beta, T *c, ptrdiff_t ldc)
{
    for (ptrdiff_t j = j_start; j < j_end; ++j) {
        const ptrdiff_t is = upper ? 0 : j;
        const ptrdiff_t ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);
        if (beta == zero) for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
        else              for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
    }
}

void ygemmtr_col(ptrdiff_t j, ptrdiff_t N, ptrdiff_t K, ptrdiff_t upper,
                 T alpha, T beta,
                 const T *a, ptrdiff_t lda, const T *b, ptrdiff_t ldb,
                 T *c, ptrdiff_t ldc,
                 ptrdiff_t trans_a, ptrdiff_t conj_a, ptrdiff_t trans_b, ptrdiff_t conj_b)
{
    const ptrdiff_t is = upper ? 0 : j;
    const ptrdiff_t ie = upper ? (j + 1) : N;
    T *cj = &C_(0, j);

    if (!trans_a) {
        if (beta == zero)      for (ptrdiff_t i = is; i < ie; ++i) cj[i]  = zero;
        else if (beta != one)  for (ptrdiff_t i = is; i < ie; ++i) cj[i] *= beta;
        /* K-unroll by 2 — expose two independent FMA chains per i to mask
         * x87 fmul latency. conj_b is hoisted out of the hot loop; a runtime
         * branch inside the K-unrolled body defeats gcc's scheduling for
         * this kind10 complex pattern. */
        ptrdiff_t l = 0;
        if (!trans_b) {
            for (; l + 1 < K; l += 2) {
                const T t0 = alpha * B_(l,     j);
                const T t1 = alpha * B_(l + 1, j);
                const T *al0 = &A_(0, l);
                const T *al1 = &A_(0, l + 1);
                for (ptrdiff_t i = is; i < ie; ++i)
                    cj[i] += t0 * al0[i] + t1 * al1[i];
            }
        } else if (!conj_b) {
            for (; l + 1 < K; l += 2) {
                const T t0 = alpha * B_(j, l);
                const T t1 = alpha * B_(j, l + 1);
                const T *al0 = &A_(0, l);
                const T *al1 = &A_(0, l + 1);
                for (ptrdiff_t i = is; i < ie; ++i)
                    cj[i] += t0 * al0[i] + t1 * al1[i];
            }
        } else {
            for (; l + 1 < K; l += 2) {
                const T t0 = alpha * ~B_(j, l);
                const T t1 = alpha * ~B_(j, l + 1);
                const T *al0 = &A_(0, l);
                const T *al1 = &A_(0, l + 1);
                for (ptrdiff_t i = is; i < ie; ++i)
                    cj[i] += t0 * al0[i] + t1 * al1[i];
            }
        }
        for (; l < K; ++l) {
            T bl;
            if (!trans_b)      bl = B_(l, j);
            else if (!conj_b)  bl = B_(j, l);
            else               bl = ~B_(j, l);
            const T t = alpha * bl;
            const T *al = &A_(0, l);
            for (ptrdiff_t i = is; i < ie; ++i) cj[i] += t * al[i];
        }
    } else {
        for (ptrdiff_t i = is; i < ie; ++i) {
            T s = zero;
            if (!trans_b) {
                if (!conj_a) for (ptrdiff_t l = 0; l < K; ++l) s += A_(l, i) * B_(l, j);
                else         for (ptrdiff_t l = 0; l < K; ++l) s += ~A_(l, i) * B_(l, j);
            } else if (!conj_b) {
                if (!conj_a) for (ptrdiff_t l = 0; l < K; ++l) s += A_(l, i) * B_(j, l);
                else         for (ptrdiff_t l = 0; l < K; ++l) s += ~A_(l, i) * B_(j, l);
            } else {
                if (!conj_a) for (ptrdiff_t l = 0; l < K; ++l) s += A_(l, i) * ~B_(j, l);
                else         for (ptrdiff_t l = 0; l < K; ++l) s += ~A_(l, i) * ~B_(j, l);
            }
            cj[i] = (beta == zero) ? alpha * s : alpha * s + beta * cj[i];
        }
    }
}

void ygemmtr_serial(char uplo, char transa, char transb,
                    ptrdiff_t N, ptrdiff_t K,
                    const T *alpha_,
                    const T *a, ptrdiff_t lda,
                    const T *b, ptrdiff_t ldb,
                    const T *beta_,
                    T *c, ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const ptrdiff_t upper = ((char)toupper((unsigned char)uplo) == 'U');
    const ptrdiff_t ta = (char)toupper((unsigned char)transa);
    const ptrdiff_t tb = (char)toupper((unsigned char)transb);

    if (N <= 0) return;

    const ptrdiff_t conj_a = (ta == 'C');
    const ptrdiff_t conj_b = (tb == 'C');
    const ptrdiff_t trans_a = (ta != 'N');
    const ptrdiff_t trans_b = (tb != 'N');

    if (alpha == zero || K == 0) {
        if (beta == one) return;
        ygemmtr_beta_scale(0, N, N, upper, beta, c, ldc);
        return;
    }

    for (ptrdiff_t j = 0; j < N; ++j)
        ygemmtr_col(j, N, K, upper, alpha, beta, a, lda, b, ldb, c, ldc,
                    trans_a, conj_a, trans_b, conj_b);
}

#undef A_
#undef B_
#undef C_
