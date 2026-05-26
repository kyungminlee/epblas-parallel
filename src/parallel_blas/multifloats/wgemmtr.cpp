/*
 * wgemmtr — multifloats complex DD triangular GEMM update.
 *   C := alpha · op(A) · op(B) + beta · C   (only UPLO triangle of C)
 *
 * Structure mirrors mgemmtr: blocked over jc; off-diagonal rect routed
 * through wgemm (carries the complex-DD SIMD kernel); the jb×jb
 * diagonal triangle uses a scalar update. Unlike the real case, all 9
 * (ta, tb) ∈ {N,T,C}² combinations are real branches because T and C
 * differ for complex.
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
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {

#define WGEMMTR_OMP_MIN 32

inline char up(const char *p) { return (char)std::toupper((unsigned char)*p); }
const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
const T one_cdd { R{1.0, 0.0}, R{0.0, 0.0} };
inline bool cdd_iszero(const T &x) {
    return x.re.limbs[0] == 0.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
inline bool cdd_isone(const T &x) {
    return x.re.limbs[0] == 1.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }

int env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    int v = std::atoi(s);
    return v > 0 ? v : dflt;
}

int g_nb = 0;
int gemmtr_nb(void) {
    if (g_nb == 0) g_nb = env_int("WGEMMTR_NB", 64);
    return g_nb;
}

} /* anonymous */

#define A_(i, j)  a[(std::size_t)(j) * lda + (i)]
#define B_(i, j)  b[(std::size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(std::size_t)(j) * ldc + (i)]

extern "C" void wgemm_(
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
    const bool trans_a = (ta != 'N');
    const bool trans_b = (tb != 'N');
    const bool conj_a  = (ta == 'C');
    const bool conj_b  = (tb == 'C');

    for (int j = jc; j < jc + jb; ++j) {
        const int is = upper ? jc        : j;
        const int ie = upper ? (j + 1)   : (jc + jb);
        T *cj = c + (std::size_t)j * ldc;

        if (!trans_a) {
            for (int l = 0; l < K; ++l) {
                T bl;
                if (!trans_b)      bl = B_(l, j);
                else if (!conj_b)  bl = B_(j, l);
                else               bl = cconj(B_(j, l));
                const T t = cmul(alpha, bl);
                if (cdd_iszero(t)) continue;
                const T *al = &A_(0, l);
                for (int i = is; i < ie; ++i) cj[i] = cadd(cj[i], cmul(t, al[i]));
            }
        } else {
            for (int i = is; i < ie; ++i) {
                T s = zero_cdd;
                if (!trans_b) {
                    if (!conj_a) for (int l = 0; l < K; ++l) s = cadd(s, cmul(A_(l, i),        B_(l, j)));
                    else         for (int l = 0; l < K; ++l) s = cadd(s, cmul(cconj(A_(l, i)), B_(l, j)));
                } else if (!conj_b) {
                    if (!conj_a) for (int l = 0; l < K; ++l) s = cadd(s, cmul(A_(l, i),        B_(j, l)));
                    else         for (int l = 0; l < K; ++l) s = cadd(s, cmul(cconj(A_(l, i)), B_(j, l)));
                } else {
                    if (!conj_a) for (int l = 0; l < K; ++l) s = cadd(s, cmul(A_(l, i),        cconj(B_(j, l))));
                    else         for (int l = 0; l < K; ++l) s = cadd(s, cmul(cconj(A_(l, i)), cconj(B_(j, l))));
                }
                cj[i] = cadd(cj[i], cmul(alpha, s));
            }
        }
    }
}

} /* anonymous */

extern "C" void wgemmtr_(
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
    const char ta = up(transa);
    const char tb = up(transb);

    if (N <= 0) return;

    if (cdd_iszero(alpha) || K == 0) {
        if (cdd_isone(beta)) return;
#ifdef _OPENMP
        const bool use_omp0 = (N >= WGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp0) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            const int is = upper ? 0 : j;
            const int ie = upper ? (j + 1) : N;
            T *cj = &C_(0, j);
            if (cdd_iszero(beta)) for (int i = is; i < ie; ++i) cj[i] = zero_cdd;
            else                  for (int i = is; i < ie; ++i) cj[i] = cmul(cj[i], beta);
        }
        return;
    }

    const int nb = gemmtr_nb();
    const char ta_s[1] = { ta };
    const char tb_s[1] = { tb };

#ifdef _OPENMP
    const bool use_omp = (N >= WGEMMTR_OMP_MIN && blas_omp_max_threads() > 1);
    #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;

        for (int j = jc; j < jc + jb; ++j) {
            const int is = upper ? 0 : j;
            const int ie = upper ? (j + 1) : N;
            T *cj = &C_(0, j);
            if (cdd_iszero(beta))      for (int i = is; i < ie; ++i) cj[i] = zero_cdd;
            else if (!cdd_isone(beta)) for (int i = is; i < ie; ++i) cj[i] = cmul(cj[i], beta);
        }

        diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, upper, ta, tb);

        if (upper) {
            if (jc > 0) {
                const int m = jc;
                const T *ablk = (ta == 'N') ? &A_(0, 0) : &A_(0, 0);
                const T *bblk = (tb == 'N') ? &B_(0, jc) : &B_(jc, 0);
                wgemm_(ta_s, tb_s, &m, &jb, &K, &alpha,
                       ablk, &lda, bblk, &ldb,
                       &one_cdd, &C_(0, jc), &ldc, 1, 1);
            }
        } else {
            const int trailing = N - jc - jb;
            if (trailing > 0) {
                const int r0 = jc + jb;
                const T *ablk = (ta == 'N') ? &A_(r0, 0) : &A_(0, r0);
                const T *bblk = (tb == 'N') ? &B_(0, jc) : &B_(jc, 0);
                wgemm_(ta_s, tb_s, &trailing, &jb, &K, &alpha,
                       ablk, &lda, bblk, &ldb,
                       &one_cdd, &C_(r0, jc), &ldc, 1, 1);
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
