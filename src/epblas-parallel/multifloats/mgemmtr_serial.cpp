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
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {

const T zero_dd{0.0, 0.0};
const T one_dd {1.0, 0.0};

#define A_(i, j)  a[(std::size_t)(j) * lda + (i)]
#define B_(i, j)  b[(std::size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(std::size_t)(j) * ldc + (i)]

/* Scalar update of the jb×jb diagonal triangle at (jc, jc).
 * Assumes beta-scaling on C[is..ie, j] already done. */
inline void diag_add(int jc, int jb, int K, T alpha,
                     const T *a, int lda,
                     const T *b, int ldb,
                     T *c, int ldc,
                     bool upper, char ta, char tb)
{
    for (int j = jc; j < jc + jb; ++j) {
        const int is = upper ? jc        : j;
        const int ie = upper ? (j + 1)   : (jc + jb);
        T *cj = c + (std::size_t)j * ldc;

        if (ta == 'N') {
            if (tb == 'N') {
                for (int l = 0; l < K; ++l) {
                    const T t = alpha * B_(l, j);
                    if (eq0(t)) continue;
                    const T *al = &A_(0, l);
                    for (int i = is; i < ie; ++i) cj[i] = cj[i] + t * al[i];
                }
            } else {
                for (int l = 0; l < K; ++l) {
                    const T t = alpha * B_(j, l);
                    if (eq0(t)) continue;
                    const T *al = &A_(0, l);
                    for (int i = is; i < ie; ++i) cj[i] = cj[i] + t * al[i];
                }
            }
        } else {
            if (tb == 'N') {
                for (int i = is; i < ie; ++i) {
                    T s = zero_dd;
                    for (int l = 0; l < K; ++l) s = s + A_(l, i) * B_(l, j);
                    cj[i] = cj[i] + alpha * s;
                }
            } else {
                for (int i = is; i < ie; ++i) {
                    T s = zero_dd;
                    for (int l = 0; l < K; ++l) s = s + A_(l, i) * B_(j, l);
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
int mgemmtr_block_nb(void) {
    static const int nb = [] {
        const char *e = std::getenv("MGEMMTR_NB");
        int v = e ? std::atoi(e) : 0;
        return (v > 0) ? v : 8;
    }();
    return nb;
}

void mgemmtr_beta_core(int j0, int j1, int N, bool upper,
                       T beta, T *c, int ldc)
{
    for (int j = j0; j < j1; ++j) {
        const int is = upper ? 0 : j;
        const int ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);
        if (eq0(beta)) for (int i = is; i < ie; ++i) cj[i] = zero_dd;
        else                 for (int i = is; i < ie; ++i) cj[i] = cj[i] * beta;
    }
}

void mgemmtr_block_core(int jc, int jb, int N, int K,
                        T alpha, T beta,
                        const T *a, int lda,
                        const T *b, int ldb,
                        T *c, int ldc,
                        bool upper, char ta, char tb)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char *ta_s = (ta == 'N') ? NN : TN;
    const char *tb_s = (tb == 'N') ? NN : TN;

    /* Beta-scale the triangle slice for cols [jc, jc+jb). */
    for (int j = jc; j < jc + jb; ++j) {
        const int is = upper ? 0 : j;
        const int ie = upper ? (j + 1) : N;
        T *cj = &C_(0, j);
        if (eq0(beta))      for (int i = is; i < ie; ++i) cj[i] = zero_dd;
        else if (!eq1(beta)) for (int i = is; i < ie; ++i) cj[i] = cj[i] * beta;
    }

    /* Diagonal jb×jb triangle: scalar. */
    diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, upper, ta, tb);

    /* Off-diagonal rectangle: routed through mgemm_serial (SIMD). */
    if (upper) {
        if (jc > 0) {
            const int m = jc;
            const T *ablk = (ta == 'N') ? &A_(0, 0) : &A_(0, 0);
            const T *bblk = (tb == 'N') ? &B_(0, jc) : &B_(jc, 0);
            mgemm_serial(ta_s, tb_s, &m, &jb, &K, &alpha,
                         ablk, &lda, bblk, &ldb,
                         &one_dd, &C_(0, jc), &ldc, 1, 1);
        }
    } else {
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            const int r0 = jc + jb;
            const T *ablk = (ta == 'N') ? &A_(r0, 0) : &A_(0, r0);
            const T *bblk = (tb == 'N') ? &B_(0, jc) : &B_(jc, 0);
            mgemm_serial(ta_s, tb_s, &trailing, &jb, &K, &alpha,
                         ablk, &lda, bblk, &ldb,
                         &one_dd, &C_(r0, jc), &ldc, 1, 1);
        }
    }
}

extern "C" void mgemmtr_serial(
    const char *uplo, const char *transa, const char *transb,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t ta_len, std::size_t tb_len)
{
    (void)uplo_len; (void)ta_len; (void)tb_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const bool upper = (up(uplo) == 'U');
    char ta = up(transa); if (ta == 'C') ta = 'T';
    char tb = up(transb); if (tb == 'C') tb = 'T';

    if (N <= 0) return;

    if (eq0(alpha) || K == 0) {
        if (eq1(beta)) return;
        mgemmtr_beta_core(0, N, N, upper, beta, c, ldc);
        return;
    }

    const int nb = mgemmtr_block_nb();
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        mgemmtr_block_core(jc, jb, N, K, alpha, beta,
                           a, lda, b, ldb, c, ldc, upper, ta, tb);
    }
}

#undef A_
#undef B_
#undef C_
