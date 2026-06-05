/*
 * wsyrk_serial — multifloats complex (DD) symmetric rank-k update, pure
 * single-thread worker. Owns ALL the numerics; no OpenMP on this path.
 * TRANS ∈ {N, T}.
 *
 * AVX2 4-wide SIMD diag kernel + wgemm_serial trailing. Complex DD analog of
 * msyrk's SIMD diag (cdd_mul / cdd_add primitives, 4 SoA arrays per packed
 * matrix). Computes the full square diag block and writes only the UPLO
 * triangle on unpack.
 *
 * The trailing rank-k update routes through wgemm_serial (no nested OpenMP) so
 * wsyrk_parallel.cpp can call wsyrk_block from inside its own omp region.
 */
#include "wsyrk_kernel.h"
#include "wgemm_kernel.h"
#include <cstddef>
#include <cstdlib>
#include <cctype>

#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {

int env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    int v = std::atoi(s);
    return v > 0 ? v : dflt;
}

inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

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

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define C_(i, j)  c[static_cast<std::size_t>(j) * ldc + (i)]

#ifdef MBLAS_SIMD_DD

constexpr int kSimdLane = simd_dd::NR;
constexpr int kMaxBlockM = 128;
constexpr int kMaxK      = 512;

inline void pack_4col_cdd(int count, int row_start,
                          const T *m, int ldm, int j_start, int j_count,
                          double *rh, double *rl, double *ih, double *il)
{
    for (int j = 0; j < j_count; ++j) {
        const T *col = m + static_cast<std::size_t>(j_start + j) * ldm;
        for (int i = 0; i < count; ++i) {
            rh[i * kSimdLane + j] = col[row_start + i].re.limbs[0];
            rl[i * kSimdLane + j] = col[row_start + i].re.limbs[1];
            ih[i * kSimdLane + j] = col[row_start + i].im.limbs[0];
            il[i * kSimdLane + j] = col[row_start + i].im.limbs[1];
        }
    }
    for (int j = j_count; j < kSimdLane; ++j)
        for (int i = 0; i < count; ++i) {
            rh[i * kSimdLane + j] = 0.0; rl[i * kSimdLane + j] = 0.0;
            ih[i * kSimdLane + j] = 0.0; il[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_4col_cdd_triangle(int jc, int jb, int j_start, int j_count,
                                     char UPLO, T *c, int ldc,
                                     const double *rh, const double *rl,
                                     const double *ih, const double *il)
{
    for (int j = 0; j < j_count; ++j) {
        const int j_abs = j_start + j;
        const int i_lo = (UPLO == 'L') ? j_abs   : jc;
        const int i_hi = (UPLO == 'L') ? jc + jb : j_abs + 1;
        T *col = c + static_cast<std::size_t>(j_abs) * ldc;
        for (int i = i_lo; i < i_hi; ++i) {
            const int ir = i - jc;
            col[i].re.limbs[0] = rh[ir * kSimdLane + j];
            col[i].re.limbs[1] = rl[ir * kSimdLane + j];
            col[i].im.limbs[0] = ih[ir * kSimdLane + j];
            col[i].im.limbs[1] = il[ir * kSimdLane + j];
        }
    }
}

inline void broadcast_cdd(const T &v,
                          __m256d &rh, __m256d &rl,
                          __m256d &ih, __m256d &il)
{
    rh = _mm256_set1_pd(v.re.limbs[0]); rl = _mm256_set1_pd(v.re.limbs[1]);
    ih = _mm256_set1_pd(v.im.limbs[0]); il = _mm256_set1_pd(v.im.limbs[1]);
}

/* TR='N' rank-1: for each l, load α·A(j_panel..+4, l) and update
 * C[i, j_panel..+4] += t · A(i, l) across i ∈ diag block. */
inline void simd_syrk_diag_tn(int jc, int jb, int K, T alpha,
                              const T *a, int lda,
                              int j_panel, int j_count,
                              double *crh, double *crl,
                              double *cih, double *cil)
{
    __m256d a_rh, a_rl, a_ih, a_il;
    broadcast_cdd(alpha, a_rh, a_rl, a_ih, a_il);
    alignas(32) double bj_rh[kSimdLane], bj_rl[kSimdLane];
    alignas(32) double bj_ih[kSimdLane], bj_il[kSimdLane];
    for (int l = 0; l < K; ++l) {
        for (int j = 0; j < j_count; ++j) {
            const T v = A_(j_panel + j, l);
            bj_rh[j] = v.re.limbs[0]; bj_rl[j] = v.re.limbs[1];
            bj_ih[j] = v.im.limbs[0]; bj_il[j] = v.im.limbs[1];
        }
        for (int j = j_count; j < kSimdLane; ++j) {
            bj_rh[j] = 0.0; bj_rl[j] = 0.0;
            bj_ih[j] = 0.0; bj_il[j] = 0.0;
        }
        __m256d aj_rh = _mm256_load_pd(bj_rh);
        __m256d aj_rl = _mm256_load_pd(bj_rl);
        __m256d aj_ih = _mm256_load_pd(bj_ih);
        __m256d aj_il = _mm256_load_pd(bj_il);
        __m256d trh, trl, tih, til;
        simd_dd::cdd_mul(a_rh, a_rl, a_ih, a_il,
                         aj_rh, aj_rl, aj_ih, aj_il,
                         trh, trl, tih, til);
        for (int i = jc; i < jc + jb; ++i) {
            const int ir = i - jc;
            __m256d aih, ail, aiih, aiil;
            broadcast_cdd(A_(i, l), aih, ail, aiih, aiil);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(trh, trl, tih, til,
                             aih, ail, aiih, aiil,
                             prh, prl, pih, pil);
            __m256d ck_rh = _mm256_load_pd(&crh[ir * kSimdLane]);
            __m256d ck_rl = _mm256_load_pd(&crl[ir * kSimdLane]);
            __m256d ck_ih = _mm256_load_pd(&cih[ir * kSimdLane]);
            __m256d ck_il = _mm256_load_pd(&cil[ir * kSimdLane]);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(ck_rh, ck_rl, ck_ih, ck_il,
                             prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            _mm256_store_pd(&crh[ir * kSimdLane], nrh);
            _mm256_store_pd(&crl[ir * kSimdLane], nrl);
            _mm256_store_pd(&cih[ir * kSimdLane], nih);
            _mm256_store_pd(&cil[ir * kSimdLane], nil_);
        }
    }
}

/* TR='T' dot product: pre-packed Aj columns. For each i, run 4-wide
 * SIMD accumulator across l. */
inline void simd_syrk_diag_tt(int jc, int jb, int K, T alpha,
                              const T *a, int lda,
                              int j_panel, int j_count,
                              const double *ajrh, const double *ajrl,
                              const double *ajih, const double *ajil,
                              double *crh, double *crl,
                              double *cih, double *cil)
{
    (void)j_panel; (void)j_count;
    __m256d a_rh, a_rl, a_ih, a_il;
    broadcast_cdd(alpha, a_rh, a_rl, a_ih, a_il);
    for (int i = jc; i < jc + jb; ++i) {
        const int ir = i - jc;
        const T *Ai = a + static_cast<std::size_t>(i) * lda;
        __m256d srh = _mm256_setzero_pd();
        __m256d srl = _mm256_setzero_pd();
        __m256d sih = _mm256_setzero_pd();
        __m256d sil = _mm256_setzero_pd();
        for (int l = 0; l < K; ++l) {
            __m256d aih, ail, aiih, aiil;
            broadcast_cdd(Ai[l], aih, ail, aiih, aiil);
            __m256d ajh = _mm256_load_pd(&ajrh[l * kSimdLane]);
            __m256d ajl = _mm256_load_pd(&ajrl[l * kSimdLane]);
            __m256d ajih_v = _mm256_load_pd(&ajih[l * kSimdLane]);
            __m256d ajil_v = _mm256_load_pd(&ajil[l * kSimdLane]);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(aih, ail, aiih, aiil,
                             ajh, ajl, ajih_v, ajil_v,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(srh, srl, sih, sil,
                             prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            srh = nrh; srl = nrl; sih = nih; sil = nil_;
        }
        __m256d prh, prl, pih, pil;
        simd_dd::cdd_mul(a_rh, a_rl, a_ih, a_il,
                         srh, srl, sih, sil,
                         prh, prl, pih, pil);
        __m256d ck_rh = _mm256_load_pd(&crh[ir * kSimdLane]);
        __m256d ck_rl = _mm256_load_pd(&crl[ir * kSimdLane]);
        __m256d ck_ih = _mm256_load_pd(&cih[ir * kSimdLane]);
        __m256d ck_il = _mm256_load_pd(&cil[ir * kSimdLane]);
        __m256d nrh, nrl, nih, nil_;
        simd_dd::cdd_add(ck_rh, ck_rl, ck_ih, ck_il,
                         prh, prl, pih, pil,
                         nrh, nrl, nih, nil_);
        _mm256_store_pd(&crh[ir * kSimdLane], nrh);
        _mm256_store_pd(&crl[ir * kSimdLane], nrl);
        _mm256_store_pd(&cih[ir * kSimdLane], nih);
        _mm256_store_pd(&cil[ir * kSimdLane], nil_);
    }
}

inline void simd_syrk_diag_panels(int jc, int jb, int K, T alpha,
                                  const T *a, int lda,
                                  T *c, int ldc,
                                  char UPLO, char TR)
{
    alignas(32) double crh[kMaxBlockM * kSimdLane];
    alignas(32) double crl[kMaxBlockM * kSimdLane];
    alignas(32) double cih[kMaxBlockM * kSimdLane];
    alignas(32) double cil[kMaxBlockM * kSimdLane];
    alignas(32) static thread_local double ajrh[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajrl[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajih[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajil[kMaxK * kSimdLane];

    for (int j = jc; j < jc + jb; j += kSimdLane) {
        const int jcount = (jc + jb - j < kSimdLane) ? (jc + jb - j) : kSimdLane;
        pack_4col_cdd(jb, jc, c, ldc, j, jcount, crh, crl, cih, cil);
        if (TR == 'N') {
            simd_syrk_diag_tn(jc, jb, K, alpha, a, lda, j, jcount,
                              crh, crl, cih, cil);
        } else {
            for (int jj = 0; jj < jcount; ++jj) {
                const T *col = a + static_cast<std::size_t>(j + jj) * lda;
                for (int l = 0; l < K; ++l) {
                    ajrh[l * kSimdLane + jj] = col[l].re.limbs[0];
                    ajrl[l * kSimdLane + jj] = col[l].re.limbs[1];
                    ajih[l * kSimdLane + jj] = col[l].im.limbs[0];
                    ajil[l * kSimdLane + jj] = col[l].im.limbs[1];
                }
            }
            for (int jj = jcount; jj < kSimdLane; ++jj)
                for (int l = 0; l < K; ++l) {
                    ajrh[l * kSimdLane + jj] = 0.0; ajrl[l * kSimdLane + jj] = 0.0;
                    ajih[l * kSimdLane + jj] = 0.0; ajil[l * kSimdLane + jj] = 0.0;
                }
            simd_syrk_diag_tt(jc, jb, K, alpha, a, lda, j, jcount,
                              ajrh, ajrl, ajih, ajil,
                              crh, crl, cih, cil);
        }
        unpack_4col_cdd_triangle(jc, jb, j, jcount, UPLO, c, ldc,
                                 crh, crl, cih, cil);
    }
}

#endif  /* MBLAS_SIMD_DD */

void syrk_diag_add(int jc, int jb, int K, T alpha,
                   const T *a, int lda, T *c, int ldc,
                   char UPLO, char TR)
{
    if (TR == 'N') {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            for (int l = 0; l < K; ++l) {
                const T ajl = A_(j, l);
                if (!cdd_iszero(ajl)) {
                    const T t = cmul(alpha, ajl);
                    for (int i = i_lo; i < i_hi; ++i) cj[i] = cadd(cj[i], cmul(t, A_(i, l)));
                }
            }
        }
    } else {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            const T *Aj = a + static_cast<std::size_t>(j) * lda;
            for (int i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + static_cast<std::size_t>(i) * lda;
                T s = zero_cdd;
                for (int l = 0; l < K; ++l) s = cadd(s, cmul(Ai[l], Aj[l]));
                cj[i] = cadd(cj[i], cmul(alpha, s));
            }
        }
    }
}

inline void diag_dispatch(int jc, int jb, int K, T alpha,
                          const T *a, int lda, T *c, int ldc,
                          char UPLO, char TR)
{
#ifdef MBLAS_SIMD_DD
    if (jb <= kMaxBlockM && K <= kMaxK) {
        simd_syrk_diag_panels(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);
        return;
    }
#endif
    syrk_diag_add(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);
}

} /* anonymous namespace */

int wsyrk_block_nb(void) {
    static int nb = 0;
    if (nb == 0) nb = env_int("WSYRK_NB", 64);
    return nb;
}

void wsyrk_scale_col(int j, int N, char UPLO, T beta, T *c, int ldc) {
    const int i_lo = (UPLO == 'L') ? j : 0;
    const int i_hi = (UPLO == 'L') ? N : j + 1;
    T *cj = c + static_cast<std::size_t>(j) * ldc;
    if (cdd_iszero(beta)) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero_cdd;
    else                  for (int i = i_lo; i < i_hi; ++i) cj[i] = cmul(cj[i], beta);
}

void wsyrk_block(int jc, int jb, int N, int K, char UPLO, char TR,
                 T alpha, T beta, const T *a, int lda, T *c, int ldc)
{
    /* Beta-scale this block's own triangle columns. */
    for (int j = jc; j < jc + jb; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (cdd_iszero(beta))      for (int i = i_lo; i < i_hi; ++i) cj[i] = zero_cdd;
        else if (!cdd_isone(beta)) for (int i = i_lo; i < i_hi; ++i) cj[i] = cmul(cj[i], beta);
    }

    diag_dispatch(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    if (UPLO == 'L') {
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            const int j0 = jc + jb;
            if (TR == 'N') {
                wgemm_serial(NN, TN, &trailing, &jb, &K, &alpha,
                             &A_(j0, 0), &lda, &A_(jc, 0), &lda,
                             &one_cdd, &C_(j0, jc), &ldc, 1, 1);
            } else {
                wgemm_serial(TN, NN, &trailing, &jb, &K, &alpha,
                             &A_(0, j0), &lda, &A_(0, jc), &lda,
                             &one_cdd, &C_(j0, jc), &ldc, 1, 1);
            }
        }
    } else {
        if (jc > 0) {
            if (TR == 'N') {
                wgemm_serial(NN, TN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &A_(jc, 0), &lda,
                             &one_cdd, &C_(0, jc), &ldc, 1, 1);
            } else {
                wgemm_serial(TN, NN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &A_(0, jc), &lda,
                             &one_cdd, &C_(0, jc), &ldc, 1, 1);
            }
        }
    }
}

extern "C" void wsyrk_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *beta_,
    T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    const char TR = up(trans);

    if (N == 0) return;

    if (cdd_iszero(alpha) || K == 0) {
        if (cdd_isone(beta)) return;
        for (int j = 0; j < N; ++j) wsyrk_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const int nb = wsyrk_block_nb();
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        wsyrk_block(jc, jb, N, K, UPLO, TR, alpha, beta, a, lda, c, ldc);
    }
}

#undef A_
#undef C_
