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

/* 2x2 register tile for the !trans_a axpy path: L-unroll by 2 (two temps
 * t0,t1, already done by the caller) AND I-unroll by 2 (two independent C
 * accumulators cj[i], cj[i+1]). The two i-chains are independent RMWs into
 * distinct C rows, so their faddp latency overlaps — the lever the single-i
 * 2-chain can't pull. noinline isolates the x87 register schedule from the
 * caller's scaffold (project-x87-accumulator-spill trigger 3). restrict: cj is
 * a C column, al0/al1 are A columns — provably distinct arrays. */
static void __attribute__((noinline))
ygemmtr_axpy2(TC *restrict cj, const TC *restrict al0, const TC *restrict al1,
              TC t0, TC t1, ptrdiff_t is, ptrdiff_t ie)
{
    ptrdiff_t i = is;
    for (; i + 1 < ie; i += 2) {
        cj[i]     += t0 * al0[i]     + t1 * al1[i];
        cj[i + 1] += t0 * al0[i + 1] + t1 * al1[i + 1];
    }
    for (; i < ie; ++i)
        cj[i] += t0 * al0[i] + t1 * al1[i];
}

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
        /* The !trans_a axpy path: L-unroll by 2 (two temps t0,t1, computed
         * here) feeding a 2x2 register tile (ygemmtr_axpy2 also I-unrolls by 2).
         * conj_b is hoisted out of the hot loop into the three l-loops below; a
         * runtime branch inside the body defeats gcc's scheduling here.
         *
         * WHY 2x2 (do not "simplify" back to a single-i or single-l loop):
         * the trans_a=='N' keys (UNN/UNT/UNC/LNN/LNT/LNC — transb and uplo are
         * irrelevant; it is purely this axpy path vs the trans_a dot path below)
         * trail the gfortran (migrated) reference, which i-unrolls its main loop
         * x2 and schedules the fp80 complex MAC across the 8-deep x87 stack a bit
         * better than gcc. trans_a=='T'/'C' (the dot path) already ties mig.
         * Progression of structural alternatives, all MEASURED on the dual
         * harness (REPS=40), worst cell = UNC/256 unless noted:
         *     - plain single chain: par/mig 1.45 — gcc won't i-unroll it,
         *       retires ~29% more instrs/call (187M vs mig 145M). NOT placement
         *       (-falign-loops=8/16/32 all 1.45).
         *     - explicit i-unroll x2, single temp: 1.36 (gcc's lone-axis i-unroll
         *       codegen is worse than gfortran's).
         *     - L-unroll x2, single i (the old shipped "2-chain"): 1.08. Matches
         *       mig's instr count (+2.5%), residual ~5% lower IPC.
         *     - register-resident dot for NN (A(i,l) strided by lda): 1.22 (N-N),
         *       1.55+ (N-T/N-C) — strided A kills it.
         *     - 2x2 tile = L-unroll x2 AND I-unroll x2 (SHIPPED, 2026-06-24):
         *       1.041. Two independent C-accumulators (cj[i], cj[i+1]) overlap
         *       their faddp latency while t0,t1 stay register-resident (4 x87
         *       slots, NO spill — verified in the ygemmtr_axpy2 disasm). Halves
         *       the gap the single-axis unrolls left (~8% -> ~4%); helps all 6
         *       cells (e.g. UNC/256 1.080->1.041, UNN/64 1.068->1.063).
         *     - 3rd l-chain (L-unroll x3): spills the 8-deep x87 stack
         *       (project-x87-accumulator-spill trigger 6 — complex fp80).
         *   The 2x2 is gcc's best and beats OpenBLAS ~28% (par/ob ~0.72). The
         *   residual ~4% vs mig (binding at small N) is the genuine gcc-vs-
         *   gfortran x87 scheduling gap — same floor class as yhemm UPPER and
         *   yher2 LOWER. The ygemm TT levers (conj-unswitch; strided-B transpose)
         *   target failure modes absent here (no per-element conj branch; B is
         *   stride-1; omp4 already wins). */
        ptrdiff_t l = 0;
        if (!trans_b) {
            for (; l + 1 < k; l += 2) {
                const TC t0 = alpha * B_(l,     j);
                const TC t1 = alpha * B_(l + 1, j);
                ygemmtr_axpy2(cj, &A_(0, l), &A_(0, l + 1), t0, t1, is, ie);
            }
        } else if (!conj_b) {
            for (; l + 1 < k; l += 2) {
                const TC t0 = alpha * B_(j, l);
                const TC t1 = alpha * B_(j, l + 1);
                ygemmtr_axpy2(cj, &A_(0, l), &A_(0, l + 1), t0, t1, is, ie);
            }
        } else {
            for (; l + 1 < k; l += 2) {
                const TC t0 = alpha * ~B_(j, l);
                const TC t1 = alpha * ~B_(j, l + 1);
                ygemmtr_axpy2(cj, &A_(0, l), &A_(0, l + 1), t0, t1, is, ie);
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
