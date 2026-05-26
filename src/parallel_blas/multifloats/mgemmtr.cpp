/*
 * mgemmtr — multifloats real (DD) triangular GEMM update.
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 *
 * Structure mirrors msyrk: blocked over jc with block size nb.
 *   - Per jc-block: scale the triangle slice in cols [jc, jc+jb).
 *   - Off-diagonal rect (UPPER: rows above; LOWER: rows below) goes
 *     through mgemm, which carries the SIMD kernel.
 *   - jb×jb diagonal triangle handled by a scalar local update.
 * Rectangular portion is where most of the work lives once N ≥ ~3·nb,
 * so the SIMD kernel in mgemm captures it. The triangle is O(nb·N·K),
 * the rect is O(½·N²·K), and mgemm runs ~3.8× the unblocked rank-1.
 */
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {

#define MGEMMTR_OMP_MIN 32

inline char up(const char *p) { return (char)std::toupper((unsigned char)*p); }
const T zero_dd{0.0, 0.0};
const T one_dd {1.0, 0.0};
inline bool dd_iszero(const T &x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (const T &x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }

int env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    int v = std::atoi(s);
    return v > 0 ? v : dflt;
}

int g_nb = 0;
int gemmtr_nb(void) {
    if (g_nb == 0) g_nb = env_int("MGEMMTR_NB", 64);
    return g_nb;
}

} /* anonymous */

#define A_(i, j)  a[(std::size_t)(j) * lda + (i)]
#define B_(i, j)  b[(std::size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(std::size_t)(j) * ldc + (i)]

extern "C" void mgemm_(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    std::size_t transa_len, std::size_t transb_len);

namespace {

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
                    if (dd_iszero(t)) continue;
                    const T *al = &A_(0, l);
                    for (int i = is; i < ie; ++i) cj[i] = cj[i] + t * al[i];
                }
            } else {
                for (int l = 0; l < K; ++l) {
                    const T t = alpha * B_(j, l);
                    if (dd_iszero(t)) continue;
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

extern "C" void mgemmtr_(
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

    if (dd_iszero(alpha) || K == 0) {
        if (dd_isone(beta)) return;
#ifdef _OPENMP
        const bool use_omp0 = (N >= MGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp0) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const int is = upper ? 0 : j;
            const int ie = upper ? (j + 1) : N;
            T *cj = &C_(0, j);
            if (dd_iszero(beta)) for (int i = is; i < ie; ++i) cj[i] = zero_dd;
            else                 for (int i = is; i < ie; ++i) cj[i] = cj[i] * beta;
        }
        return;
    }

    const int nb = gemmtr_nb();
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char *ta_s = (ta == 'N') ? NN : TN;
    const char *tb_s = (tb == 'N') ? NN : TN;

#ifdef _OPENMP
    const bool use_omp = (N >= MGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;

        /* Beta-scale the triangle slice for cols [jc, jc+jb). */
        for (int j = jc; j < jc + jb; ++j) {
            const int is = upper ? 0 : j;
            const int ie = upper ? (j + 1) : N;
            T *cj = &C_(0, j);
            if (dd_iszero(beta))      for (int i = is; i < ie; ++i) cj[i] = zero_dd;
            else if (!dd_isone(beta)) for (int i = is; i < ie; ++i) cj[i] = cj[i] * beta;
        }

        /* Diagonal jb×jb triangle: scalar. */
        diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, upper, ta, tb);

        /* Off-diagonal rectangle: routed through mgemm (SIMD). */
        if (upper) {
            if (jc > 0) {
                const int m = jc;
                /* C[0:m, jc:jc+jb] += alpha * op(A)[0:m, :] * op(B)[:, jc:jc+jb] */
                const T *ablk = (ta == 'N') ? &A_(0, 0) : &A_(0, 0);
                const T *bblk = (tb == 'N') ? &B_(0, jc) : &B_(jc, 0);
                mgemm_(ta_s, tb_s, &m, &jb, &K, &alpha,
                       ablk, &lda, bblk, &ldb,
                       &one_dd, &C_(0, jc), &ldc, 1, 1);
            }
        } else {
            const int trailing = N - jc - jb;
            if (trailing > 0) {
                const int r0 = jc + jb;
                /* C[r0:N, jc:jc+jb] += alpha * op(A)[r0:N, :] * op(B)[:, jc:jc+jb] */
                const T *ablk = (ta == 'N') ? &A_(r0, 0) : &A_(0, r0);
                const T *bblk = (tb == 'N') ? &B_(0, jc) : &B_(jc, 0);
                mgemm_(ta_s, tb_s, &trailing, &jb, &K, &alpha,
                       ablk, &lda, bblk, &ldb,
                       &one_dd, &C_(r0, jc), &ldc, 1, 1);
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
