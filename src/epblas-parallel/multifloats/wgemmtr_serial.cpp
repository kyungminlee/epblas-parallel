/*
 * wgemmtr_serial.cpp — multifloats complex DD (complex64x2) triangular GEMM
 * update, single-thread core.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the block-
 * size policy, the scalar diagonal-triangle update, and the two per-block
 * cores (declared in wgemmtr_kernel.h), plus the public `wgemmtr_serial`
 * entry. The off-diagonal rectangle is routed through `wgemm_serial` (the
 * complex-DD SIMD kernel), so there is no nested OpenMP on this path.
 *
 * Structure mirrors mgemmtr, but all 9 (ta, tb) ∈ {N,T,C}² combinations are
 * real branches because T and C differ for complex.
 */
#include "wgemmtr_kernel.h"
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#include "wgemm_kernel.h"
#include <cstdlib>
#include <cctype>

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {

const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
const T one_cdd { R{1.0, 0.0}, R{0.0, 0.0} };
using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;

#define A_(i, j)  a[(std::size_t)(j) * lda + (i)]
#define B_(i, j)  b[(std::size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(std::size_t)(j) * ldc + (i)]

/* Scalar update of the jb×jb diagonal triangle at (jc, jc).
 * Assumes beta-scaling on C[is..ie, j] already done. */
inline void diag_add(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t K, T alpha,
                     const T *a, std::ptrdiff_t lda,
                     const T *b, std::ptrdiff_t ldb,
                     T *c, std::ptrdiff_t ldc,
                     bool upper, char ta, char tb)
{
    const bool trans_a = (ta != 'N');
    const bool trans_b = (tb != 'N');
    const bool conj_a  = (ta == 'C');
    const bool conj_b  = (tb == 'C');

    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        const std::ptrdiff_t is = upper ? jc        : j;
        const std::ptrdiff_t ie = upper ? (j + 1)   : (jc + jb);
        T *cj = c + (std::size_t)j * ldc;

        if (!trans_a) {
            for (std::ptrdiff_t l = 0; l < K; ++l) {
                T bl;
                if (!trans_b)      bl = B_(l, j);
                else if (!conj_b)  bl = B_(j, l);
                else               bl = cconj(B_(j, l));
                const T t = cmul(alpha, bl);
                if (ceq0(t)) continue;
                const T *al = &A_(0, l);
                for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = cadd(cj[i], cmul(t, al[i]));
            }
        } else {
            for (std::ptrdiff_t i = is; i < ie; ++i) {
                T s = zero_cdd;
                if (!trans_b) {
                    if (!conj_a) for (std::ptrdiff_t l = 0; l < K; ++l) s = cadd(s, cmul(A_(l, i),        B_(l, j)));
                    else         for (std::ptrdiff_t l = 0; l < K; ++l) s = cadd(s, cmul(cconj(A_(l, i)), B_(l, j)));
                } else if (!conj_b) {
                    if (!conj_a) for (std::ptrdiff_t l = 0; l < K; ++l) s = cadd(s, cmul(A_(l, i),        B_(j, l)));
                    else         for (std::ptrdiff_t l = 0; l < K; ++l) s = cadd(s, cmul(cconj(A_(l, i)), B_(j, l)));
                } else {
                    if (!conj_a) for (std::ptrdiff_t l = 0; l < K; ++l) s = cadd(s, cmul(A_(l, i),        cconj(B_(j, l))));
                    else         for (std::ptrdiff_t l = 0; l < K; ++l) s = cadd(s, cmul(cconj(A_(l, i)), cconj(B_(j, l))));
                }
                cj[i] = cadd(cj[i], cmul(alpha, s));
            }
        }
    }
}

} /* anonymous */

/* Column block size. Small by design: the jb×jb diagonal triangle of each
 * block is handled by the SCALAR diag_add (no SIMD), while the off-diagonal
 * rectangle routes through the AVX2 complex-DD wgemm_serial kernel. The scalar
 * fraction is ≈ nb/N, so a small nb minimises slow scalar work and pushes the
 * bulk through SIMD. At nb=64 a single-block solve (N≤64) never reaches the
 * SIMD off-diagonal at all. 8 is the knee: smallest multiple of the kernel's
 * NR=4 that still fills two SIMD column-tiles per block. Env-overridable via
 * WGEMMTR_NB for tuning. */
std::ptrdiff_t wgemmtr_block_nb(void) {
    static const std::ptrdiff_t nb = [] {
        const char *e = std::getenv("WGEMMTR_NB");
        std::ptrdiff_t v = e ? std::atoi(e) : 0;
        return (v > 0) ? v : 8;
    }();
    return nb;
}

void wgemmtr_beta_core(std::ptrdiff_t j0, std::ptrdiff_t j1, std::ptrdiff_t N, bool upper,
                       T beta, T *c, std::ptrdiff_t ldc)
{
    for (std::ptrdiff_t j = j0; j < j1; ++j) {
        const std::ptrdiff_t is = upper ? 0 : j;
        const std::ptrdiff_t ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);
        if (ceq0(beta)) for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = zero_cdd;
        else                  for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = cmul(cj[i], beta);
    }
}

void wgemmtr_block_core(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t N, std::ptrdiff_t K,
                        T alpha, T beta,
                        const T *a, std::ptrdiff_t lda,
                        const T *b, std::ptrdiff_t ldb,
                        T *c, std::ptrdiff_t ldc,
                        bool upper, char ta, char tb)
{
    const char ta_s[1] = { ta };
    const char tb_s[1] = { tb };

    /* Beta-scale the triangle slice for cols [jc, jc+jb). */
    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        const std::ptrdiff_t is = upper ? 0 : j;
        const std::ptrdiff_t ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);
        if (ceq0(beta))      for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = zero_cdd;
        else if (!ceq1(beta)) for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = cmul(cj[i], beta);
    }

    /* Diagonal jb×jb triangle: scalar. */
    diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, upper, ta, tb);

    /* Off-diagonal rectangle: routed through wgemm_serial (SIMD). */
    if (upper) {
        if (jc > 0) {
            const std::ptrdiff_t m = jc;
            const T *ablk = (ta == 'N') ? &A_(0, 0) : &A_(0, 0);
            const T *bblk = (tb == 'N') ? &B_(0, jc) : &B_(jc, 0);
            wgemm_serial(ta_s[0], tb_s[0], m, jb, K, &alpha, ablk, lda, bblk, ldb, &one_cdd, &C_(0, jc), ldc);
        }
    } else {
        const std::ptrdiff_t trailing = N - jc - jb;
        if (trailing > 0) {
            const std::ptrdiff_t r0 = jc + jb;
            const T *ablk = (ta == 'N') ? &A_(r0, 0) : &A_(0, r0);
            const T *bblk = (tb == 'N') ? &B_(0, jc) : &B_(jc, 0);
            wgemm_serial(ta_s[0], tb_s[0], trailing, jb, K, &alpha, ablk, lda, bblk, ldb, &one_cdd, &C_(r0, jc), ldc);
        }
    }
}

extern "C" void wgemmtr_serial(
    char uplo, char transa, char transb,
    std::ptrdiff_t N, std::ptrdiff_t K,
    const T *alpha_,
    const T *a, std::ptrdiff_t lda,
    const T *b, std::ptrdiff_t ldb,
    const T *beta_,
    T *c, std::ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const bool upper = (up(&uplo) == 'U');
    const char ta = up(&transa);
    const char tb = up(&transb);

    if (N <= 0) return;

    if (ceq0(alpha) || K == 0) {
        if (ceq1(beta)) return;
        wgemmtr_beta_core(0, N, N, upper, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = wgemmtr_block_nb();
    for (std::ptrdiff_t jc = 0; jc < N; jc += nb) {
        const std::ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        wgemmtr_block_core(jc, jb, N, K, alpha, beta,
                           a, lda, b, ldb, c, ldc, upper, ta, tb);
    }
}

#undef A_
#undef B_
#undef C_
