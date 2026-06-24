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
        /* K-unroll by 2 — expose two independent products t0*al0, t1*al1 into
         * one accumulator per i, halving the passes over cj. conj_b is hoisted
         * out of the hot loop into the three l-loops below; a runtime branch
         * inside the unrolled body defeats gcc's scheduling here.
         *
         * WHY 2-CHAIN AND NOT A PLAIN SINGLE CHAIN (do not "simplify" this):
         * the trans_a=='N' keys (UNN/UNT/UNC/LNN/LNT/LNC — note transb is
         * irrelevant, uplo is irrelevant; it is purely the !trans_a axpy path
         * here vs the trans_a dot path below) sit ~8% behind the gfortran
         * (migrated) reference. trans_a=='T'/'C' (the dot path) already ties
         * mig. The 8% is a genuine gcc-vs-gfortran x87 codegen gap, quantified
         * 2026-06-24 by isolated perf at UNN/256, OMP=1, 2000 calls:
         *
         *     leg            instr/call   IPC     wall ratio
         *     par (2-chain)   148.9M       1.359   1.08  (reproduces harness)
         *     mig (gfortran)  145.2M       1.424   1.00
         *
         *   So the 2-chain MATCHES mig's instruction count (+2.5%); the residual
         *   is ~5% lower IPC — gcc schedules the fp80 complex MAC across the
         *   8-deep x87 stack slightly worse than gfortran. That is below the
         *   source level; every structural alternative was measured WORSE:
         *     - plain single chain: 1.45. Its inner i-loop disassembles
         *       op-for-op identical to gfortran's REMAINDER loop, but gcc does
         *       NOT i-unroll it, so it retires ~29% MORE instructions/call
         *       (187M vs mig's 145M — mig i-unrolls its main loop x2). NOT a
         *       placement effect: -falign-loops=8/16/32 all 1.45.
         *     - register-resident dot for NN (A(i,l) strided by lda): 1.22 for
         *       N-N, 1.55+ for N-T/N-C — strided A kills it.
         *     - explicit i-unroll x2, single temp: 1.36 (gcc's i-unroll codegen
         *       is worse than gfortran's).
         *     - 3rd l-chain to out-ILP mig: spills the 8-deep x87 stack
         *       (project-x87-accumulator-spill trigger 6 — complex fp80).
         *   The 2-chain is gcc's best and beats OpenBLAS ~26% (par/ob ~0.74).
         * Same floor class as yhemm LL and yher2 LOWER. The ygemm TT levers
         * (conj-unswitch; strided-B transpose) target failure modes absent
         * here (no per-element conj branch; B is stride-1; omp4 already wins). */
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
