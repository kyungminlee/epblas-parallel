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

/* !trans_a axpy core: cj[i] += t0*al0[i] + t1*al1[i] (L-unroll-2, two temps
 * t0,t1 from the caller), SCALAR re/im decomposed and I-unrolled by 4.
 *
 * Keeping the accumulator as `_Complex long double` forces gcc to treat each
 * cj[i] as one indivisible re+im pair on the 8-deep x87 stack — that caps how
 * many independent fadd chains can overlap, so the loop is bottlenecked on the
 * ~5-cycle fp80 fadd latency (par/mig ~1.04-1.06, pure scheduling: par and the
 * gfortran ref emit the IDENTICAL 4.5-fldt/MAC instruction mix, see
 * task 12 / 01-disasm). Decomposing into explicit long double re/im arithmetic
 * — and reinterpreting the complex cj/al0/al1 as 2n-real arrays (the yerot /
 * ygemv trick, project-x87-accumulator-spill trigger 8) — lets gcc interleave
 * the real and imaginary fadd chains and hide that latency. I-unroll 4 gives it
 * four independent C accumulators to schedule across. Measured: par absolute ns
 * drops ~4-6% (mig, a separate gfortran archive, is unchanged) -> par/mig
 * ~0.99-1.00, native -march tune, no per-file flag needed.
 *
 * noinline isolates the x87 schedule from the caller scaffold (trigger 3).
 * restrict: cj is a C column, al0/al1 are A columns — provably distinct. */
static void __attribute__((noinline))
ygemmtr_axpy2(TC *restrict cj, const TC *restrict al0, const TC *restrict al1,
              TC t0, TC t1, ptrdiff_t is, ptrdiff_t ie)
{
    const long double t0r = __real__ t0, t0i = __imag__ t0;
    const long double t1r = __real__ t1, t1i = __imag__ t1;
    const long double *restrict a0 = (const long double *)al0;
    const long double *restrict a1 = (const long double *)al1;
    long double *restrict c = (long double *)cj;
    ptrdiff_t i = is;
    for (; i + 3 < ie; i += 4) {
        for (ptrdiff_t k = 2 * i; k < 2 * i + 8; k += 2) {
            c[k]     += (t0r * a0[k]     - t0i * a0[k + 1]) + (t1r * a1[k]     - t1i * a1[k + 1]);
            c[k + 1] += (t0r * a0[k + 1] + t0i * a0[k])     + (t1r * a1[k + 1] + t1i * a1[k]);
        }
    }
    for (; i < ie; ++i) {
        c[2 * i]     += (t0r * a0[2 * i]     - t0i * a0[2 * i + 1]) + (t1r * a1[2 * i]     - t1i * a1[2 * i + 1]);
        c[2 * i + 1] += (t0r * a0[2 * i + 1] + t0i * a0[2 * i])     + (t1r * a1[2 * i + 1] + t1i * a1[2 * i]);
    }
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
         * here) feeding ygemmtr_axpy2 (scalar re/im decompose, I-unroll 4).
         * conj_b is hoisted out of the hot loop into the three l-loops below; a
         * runtime branch inside the body defeats gcc's scheduling here.
         *
         * The trans_a=='N' keys (UNN/UNT/UNC/LNN/LNT/LNC — transb and uplo are
         * irrelevant; it is purely this axpy path vs the trans_a dot path below)
         * used to trail the gfortran (migrated) reference. trans_a=='T'/'C' (the
         * dot path) already ties mig.
         * Progression of structural alternatives, all MEASURED on the dual
         * harness, worst NoTrans cell unless noted:
         *     - plain single chain: par/mig 1.45 — gcc won't unroll it,
         *       retires ~29% more instrs/call (187M vs mig 145M). NOT placement
         *       (-falign-loops=8/16/32 all 1.45).
         *     - explicit i-unroll x2, single temp: 1.36 (lone-axis i-unroll
         *       codegen is poor; the winning axis is L, matching mig).
         *     - L-unroll x2, single i (old "2-chain"): 1.08. Matches mig instr
         *       count (+2.5%), residual ~5% lower IPC.
         *     - 2x2 _Complex tile (L-unroll x2 AND I-unroll x2): 1.04. The
         *       per-complex-MAC x87 op mix here is IDENTICAL to mig (4.5 fldt,
         *       4 fmul, 3 faddp, 1 fsubrp, 0.5 fxch, 1 fstpt — disasm, task 12).
         *       0% instruction difference, ~4% lower IPC: gcc just scheduled the
         *       _Complex MAC a hair worse. Looked like a pure x87 floor.
         *     - per-file -mtune=core2 on the _Complex 2x2: 1.039. -march=native
         *       (Skylake) over-schedules legacy x87; an x87-era tune model orders
         *       it ~2% better. Real but small; superseded below.
         *     - SCALAR re/im decompose + I-unroll 4 (SHIPPED, 2026-06-24): ~0.99.
         *       The breakthrough. Keeping cj[i] as `_Complex long double` forced
         *       gcc to treat re+im as one indivisible stack pair, capping chain
         *       overlap so the loop sat on ~5-cyc fp80 fadd latency. Splitting
         *       into explicit long double re/im (+ reinterpreting cj/al0/al1 as
         *       2n reals, the yerot/ygemv trick, x87-spill trigger 8) lets gcc
         *       interleave the real and imag fadd chains; I-unroll 4 gives four
         *       independent C accumulators. par ABSOLUTE ns drops ~4-6% (mig, a
         *       separate gfortran archive, is byte-identical across the A/B) ->
         *       par/mig ~0.99-1.00 on native tune, so NO per-file flag needed.
         *       This was a real instruction-level-parallelism win, not a
         *       reference artifact (mig flat in absolute ns; cf. enrm2).
         *   Now beats OpenBLAS ~28% (par/ob ~0.72) AND ties/beats mig. NOT a
         *   floor after all — the _Complex type was hiding the ILP from gcc.
         *   The ygemm TT levers (conj-unswitch; strided-B transpose) target
         *   failure modes absent here (no per-element conj branch; B stride-1). */
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
                /* transb=N: B_(l,j) is contiguous in l. Scalar sr/si decompose,
                 * same lever as the strided-B branches below — frees gcc to
                 * interleave the real/imag faddp chains (the `_Complex s` type
                 * pins re+im together on the x87 stack) and, for conj_a, turns
                 * the `~A` per-element fchs into an fadd/fsub swap. Bit-identical
                 * to `A * B` / `~A * B` (same products, same (ac∓bd, ad±bc)
                 * order, same per-part accumulation). */
                long double sr = 0.0L, si = 0.0L;
                if (!conj_a) {
                    for (ptrdiff_t l = 0; l < k; ++l) {
                        const TC av = A_(l, i), bv = B_(l, j);
                        const long double ar = __real__ av, ai = __imag__ av;
                        const long double br = __real__ bv, bi = __imag__ bv;
                        sr += ar * br - ai * bi;   /* a·b real */
                        si += ar * bi + ai * br;   /* a·b imag */
                    }
                } else {
                    for (ptrdiff_t l = 0; l < k; ++l) {
                        const TC av = A_(l, i), bv = B_(l, j);
                        const long double ar = __real__ av, ai = __imag__ av;
                        const long double br = __real__ bv, bi = __imag__ bv;
                        sr += ar * br + ai * bi;   /* conj(a)·b real */
                        si += ar * bi - ai * br;   /* conj(a)·b imag */
                    }
                }
                __real__ s = sr; __imag__ s = si;
            } else if (!conj_b) {
                /* T·T = A·B, transb=T so B_(j,l) is strided by ldb. Keeping the
                 * accumulator as `_Complex s` pins each (re,im) together on the
                 * 8-deep x87 stack and serializes the faddp chain (par/mig
                 * ~1.013-1.014). Decompose into scalar sr/si — the same lever as
                 * the C·T / T·C branches below and the !trans_a axpy core — so
                 * gcc interleaves the real and imag fadd chains (par BEATS mig,
                 * matching the single-conj twins). Bit-identical: same products,
                 * same (ac-bd, ad+bc) order as -fcx-fortran-rules, same per-part
                 * accumulation order as `A * B`. */
                if (!conj_a) {
                    long double sr = 0.0L, si = 0.0L;
                    for (ptrdiff_t l = 0; l < k; ++l) {
                        const TC av = A_(l, i), bv = B_(j, l);
                        const long double ar = __real__ av, ai = __imag__ av;
                        const long double br = __real__ bv, bi = __imag__ bv;
                        sr += ar * br - ai * bi;   /* a·b real */
                        si += ar * bi + ai * br;   /* a·b imag */
                    }
                    __real__ s = sr; __imag__ s = si;
                }
                /* C·T = conj(A)·B. A SINGLE operand conjugate is the awkward case
                 * for gcc -fcx-fortran-rules: it emits a per-element `fchs` that
                 * serializes the x87 multiply (zero- and two-conj cases fold —
                 * conj(a)·conj(b)=conj(a·b) — so only this one and T·C below lose
                 * ~2-4% to the ob clone). Fuse by hand: decompose into scalar
                 * re/im so the sign flip becomes an fadd/fsub swap (no fchs), and
                 * accumulate two independent fp80 chains (sr,si) to overlap faddp
                 * latency. Bit-identical: a-(-b)==a+b exactly on x87, same
                 * products, same per-part accumulation order as `~A * B`. */
                else {
                    long double sr = 0.0L, si = 0.0L;
                    for (ptrdiff_t l = 0; l < k; ++l) {
                        const TC av = A_(l, i), bv = B_(j, l);
                        const long double ar = __real__ av, ai = __imag__ av;
                        const long double br = __real__ bv, bi = __imag__ bv;
                        sr += ar * br + ai * bi;   /* conj(a)·b real */
                        si += ar * bi - ai * br;   /* conj(a)·b imag */
                    }
                    __real__ s = sr; __imag__ s = si;
                }
            } else {
                /* T·C = A·conj(B). Same single-conj fchs trap as C·T — fuse it. */
                if (!conj_a) {
                    long double sr = 0.0L, si = 0.0L;
                    for (ptrdiff_t l = 0; l < k; ++l) {
                        const TC av = A_(l, i), bv = B_(j, l);
                        const long double ar = __real__ av, ai = __imag__ av;
                        const long double br = __real__ bv, bi = __imag__ bv;
                        sr += ar * br + ai * bi;   /* a·conj(b) real */
                        si += ai * br - ar * bi;   /* a·conj(b) imag */
                    }
                    __real__ s = sr; __imag__ s = si;
                }
                /* C·C = conj(A)·conj(B) = conj(A·B). Two conjugates fold (no
                 * fchs trap), so this stayed a `_Complex s` dot — but the type
                 * still serializes the faddp chain (par/mig ~1.019-1.027, the
                 * worst DOT cell). Scalar sr/si decompose, same as T·T above:
                 * conj(a·b) = (ar·br − ai·bi) − i(ar·bi + ai·br). Bit-identical:
                 * negation is exact on x87 and −(x+y)==(−x)+(−y), same products
                 * and per-part order as `~A * ~B`. */
                else {
                    long double sr = 0.0L, si = 0.0L;
                    for (ptrdiff_t l = 0; l < k; ++l) {
                        const TC av = A_(l, i), bv = B_(j, l);
                        const long double ar = __real__ av, ai = __imag__ av;
                        const long double br = __real__ bv, bi = __imag__ bv;
                        sr += ar * br - ai * bi;     /* conj(a·b) real */
                        si += -ar * bi - ai * br;    /* conj(a·b) imag */
                    }
                    __real__ s = sr; __imag__ s = si;
                }
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
