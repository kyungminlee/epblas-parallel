/*
 * mgemmtr_serial.cpp — multifloats real (DD / float64x2) triangular GEMM
 * update, single-thread core.
 *
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 *
 * Owns ALL the numerics shared by the serial and parallel entries: the block-
 * size policy, the scalar diagonal-triangle update, and the two per-block
 * cores (declared in mgemmtr_kernel.h), plus the public `mgemmtr_serial`
 * entry. The off-diagonal rectangle is routed through `mgemm_serial` (the SIMD
 * kernel), so there is no nested OpenMP on this path.
 *
 * Structure mirrors msyrk: blocked over jc with block size nb.
 *   - Per jc-block: scale the triangle slice in cols [jc, jc+jb).
 *   - jb×jb diagonal triangle handled by a scalar local update.
 *   - Off-diagonal rect (UPPER: rows above; LOWER: rows below) goes through
 *     mgemm_serial, which carries the SIMD kernel.
 *
 * Both mgemmtr_serial and the parallel mgemmtr_ drive numerics through these
 * cores, so the two paths are bitwise-identical.
 */
#include "mgemmtr_kernel.h"
#include "mf_util.h"
#include "mf_pred.h"
#include "mgemm_kernel.h"
#include <cstdlib>
#include <cctype>

namespace mf = multifloats;
using TR = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {

const TR zero_dd{0.0, 0.0};
const TR one_dd {1.0, 0.0};

#define A_(i, j)  a[(std::size_t)(j) * lda + (i)]
#define B_(i, j)  b[(std::size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(std::size_t)(j) * ldc + (i)]

/* Scalar update of the jb×jb diagonal triangle at (jc, jc).
 * Assumes beta-scaling on C[is..ie, j] already done. */
inline void diag_add(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TR alpha,
                     const TR *a, std::ptrdiff_t lda,
                     const TR *b, std::ptrdiff_t ldb,
                     TR *c, std::ptrdiff_t ldc,
                     bool upper, char ta, char tb)
{
    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        const std::ptrdiff_t is = upper ? jc        : j;
        const std::ptrdiff_t ie = upper ? (j + 1)   : (jc + jb);
        TR *cj = c + (std::size_t)j * ldc;

        if (ta == 'N') {
            if (tb == 'N') {
                for (std::ptrdiff_t l = 0; l < k; ++l) {
                    const TR t = alpha * B_(l, j);
                    if (eq0(t)) continue;
                    const TR *al = &A_(0, l);
                    for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = cj[i] + t * al[i];
                }
            } else {
                for (std::ptrdiff_t l = 0; l < k; ++l) {
                    const TR t = alpha * B_(j, l);
                    if (eq0(t)) continue;
                    const TR *al = &A_(0, l);
                    for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = cj[i] + t * al[i];
                }
            }
        } else {
            if (tb == 'N') {
                for (std::ptrdiff_t i = is; i < ie; ++i) {
                    TR s = zero_dd;
                    for (std::ptrdiff_t l = 0; l < k; ++l) s = s + A_(l, i) * B_(l, j);
                    cj[i] = cj[i] + alpha * s;
                }
            } else {
                for (std::ptrdiff_t i = is; i < ie; ++i) {
                    TR s = zero_dd;
                    for (std::ptrdiff_t l = 0; l < k; ++l) s = s + A_(l, i) * B_(j, l);
                    cj[i] = cj[i] + alpha * s;
                }
            }
        }
    }
}

} /* anonymous */

/* Column block size. Small by design: the jb×jb diagonal triangle of each
 * block is handled by the SCALAR diag_add (no SIMD), while the off-diagonal
 * rectangle routes through the AVX2 mgemm_serial kernel. The scalar fraction
 * is ≈ nb/N, so a small nb minimises slow scalar work and pushes the bulk
 * through SIMD. 8 is the knee: smallest multiple of the kernel's NR=4 that
 * still fills two SIMD column-tiles per block, and it is best-or-tied across
 * N∈[64,512] for every (uplo,transa,transb). (Larger nb — e.g. the 64 a
 * dense GEMM would use — is pessimal here precisely because the diagonal is
 * scalar, not SIMD.) Env-overridable via MGEMMTR_NB for tuning. */
std::ptrdiff_t mgemmtr_block_nb(void) {
    static const std::ptrdiff_t nb = [] {
        const char *e = std::getenv("MGEMMTR_NB");
        std::ptrdiff_t v = e ? std::atoi(e) : 0;
        return (v > 0) ? v : 8;
    }();
    return nb;
}

void mgemmtr_beta_core(std::ptrdiff_t j0, std::ptrdiff_t j1, std::ptrdiff_t n, bool upper,
                       TR beta, TR *c, std::ptrdiff_t ldc)
{
    for (std::ptrdiff_t j = j0; j < j1; ++j) {
        const std::ptrdiff_t is = upper ? 0 : j;
        const std::ptrdiff_t ie = upper ? (j + 1) : n;
        TR *cj = &C_(0, j);
        if (eq0(beta)) for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = zero_dd;
        else                 for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = cj[i] * beta;
    }
}

void mgemmtr_block_core(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t n, std::ptrdiff_t k,
                        TR alpha, TR beta,
                        const TR *a, std::ptrdiff_t lda,
                        const TR *b, std::ptrdiff_t ldb,
                        TR *c, std::ptrdiff_t ldc,
                        bool upper, char ta, char tb)
{
    /* Beta-scale the triangle slice for cols [jc, jc+jb). */
    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        const std::ptrdiff_t is = upper ? 0 : j;
        const std::ptrdiff_t ie = upper ? (j + 1) : n;
        TR *cj = &C_(0, j);
        if (eq0(beta))      for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = zero_dd;
        else if (!eq1(beta)) for (std::ptrdiff_t i = is; i < ie; ++i) cj[i] = cj[i] * beta;
    }

    /* Diagonal jb×jb triangle: scalar. */
    diag_add(jc, jb, k, alpha, a, lda, b, ldb, c, ldc, upper, ta, tb);

    /* Off-diagonal rectangle: routed through mgemm_serial (SIMD). */
    if (upper) {
        if (jc > 0) {
            const std::ptrdiff_t m = jc;
            const TR *ablk = (ta == 'N') ? &A_(0, 0) : &A_(0, 0);
            const TR *bblk = (tb == 'N') ? &B_(0, jc) : &B_(jc, 0);
            mgemm_serial(ta, tb, m, jb, k, &alpha,
                         ablk, lda, bblk, ldb,
                         &one_dd, &C_(0, jc), ldc);
        }
    } else {
        const std::ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            const std::ptrdiff_t r0 = jc + jb;
            const TR *ablk = (ta == 'N') ? &A_(r0, 0) : &A_(0, r0);
            const TR *bblk = (tb == 'N') ? &B_(0, jc) : &B_(jc, 0);
            mgemm_serial(ta, tb, trailing, jb, k, &alpha,
                         ablk, lda, bblk, ldb,
                         &one_dd, &C_(r0, jc), ldc);
        }
    }
}

extern "C" void mgemmtr_serial(
    char uplo, char transa, char transb,
    std::ptrdiff_t n, std::ptrdiff_t k,
    const TR *alpha_,
    const TR *a, std::ptrdiff_t lda,
    const TR *b, std::ptrdiff_t ldb,
    const TR *beta_,
    TR *c, std::ptrdiff_t ldc)
{
    const TR alpha = *alpha_, beta = *beta_;
    const bool upper = (up(&uplo) == 'U');
    char ta = up(&transa); if (ta == 'C') ta = 'T';
    char tb = up(&transb); if (tb == 'C') tb = 'T';

    if (n <= 0) return;

    if (eq0(alpha) || k == 0) {
        if (eq1(beta)) return;
        mgemmtr_beta_core(0, n, n, upper, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = mgemmtr_block_nb();
    for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
        const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        mgemmtr_block_core(jc, jb, n, k, alpha, beta,
                           a, lda, b, ldb, c, ldc, upper, ta, tb);
    }
}

#undef A_
#undef B_
#undef C_
