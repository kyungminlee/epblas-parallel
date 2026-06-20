/*
 * wherk_serial — multifloats complex (DD) Hermitian rank-k update, pure
 * single-thread worker. Owns ALL the numerics; no OpenMP on this path.
 * alpha/beta REAL, A/C complex, the diagonal of C stays real. TRANS ∈ {N, C}.
 *
 * Blocked: AVX2 SIMD diagonal (real-alpha scaling, conjugated panel for TR='N',
 * conjugated Ai for TR='C', real-diag preservation on unpack) + scalar diagonal
 * fallback + wgemm_serial trailing with conjugate transpose.
 *
 * The trailing rank-k update routes through wgemm_serial (no nested OpenMP) so
 * wherk_parallel.cpp can call wherk_block from inside its own omp region.
 */
#include "wherk_kernel.h"
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

inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

const R rzero{0.0, 0.0};
const T czero{ rzero, rzero };

inline bool dd_iszero(R x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (R x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }
inline bool cdd_iszero(const T &x) { return dd_iszero(x.re) && dd_iszero(x.im); }

inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
inline T rcmul(R const &r, T const &z) { return T{ r * z.re, r * z.im }; }

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

/* Triangle unpack with diagonal-real preservation: for diagonal cells
 * (i == j_abs) overwrite only the .re limbs, keeping .im of the
 * original C[i,i] (Netlib herk preserves c.im on diagonal). */
inline void unpack_4col_herk_triangle(int jc, int jb, int j_start, int j_count,
                                      char UPLO, T *c, int ldc,
                                      const double *rh, const double *rl,
                                      const double *ih, const double *il)
{
    (void)ih; (void)il;
    for (int j = 0; j < j_count; ++j) {
        const int j_abs = j_start + j;
        const int i_lo = (UPLO == 'L') ? j_abs   : jc;
        const int i_hi = (UPLO == 'L') ? jc + jb : j_abs + 1;
        T *col = c + static_cast<std::size_t>(j_abs) * ldc;
        for (int i = i_lo; i < i_hi; ++i) {
            const int ir = i - jc;
            col[i].re.limbs[0] = rh[ir * kSimdLane + j];
            col[i].re.limbs[1] = rl[ir * kSimdLane + j];
            if (i != j_abs) {
                col[i].im.limbs[0] = ih[ir * kSimdLane + j];
                col[i].im.limbs[1] = il[ir * kSimdLane + j];
            }
            /* else: keep original col[i].im (real-diag preservation) */
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

/* TR='N' rank-1 with Hermitian-conjugated Aj panel and real alpha.
 * t = alpha · conj(A(j_panel..+4, l));
 * C[i, j_panel..+4] += t · A(i, l). */
inline void simd_herk_diag_n(int jc, int jb, int K, R alpha,
                             const T *a, int lda,
                             int j_panel, int j_count,
                             double *crh, double *crl,
                             double *cih, double *cil)
{
    const __m256d a_h = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d a_l = _mm256_set1_pd(alpha.limbs[1]);
    alignas(32) double aj_rh[kSimdLane], aj_rl[kSimdLane];
    alignas(32) double aj_ih[kSimdLane], aj_il[kSimdLane];
    for (int l = 0; l < K; ++l) {
        for (int j = 0; j < j_count; ++j) {
            const T v = A_(j_panel + j, l);
            aj_rh[j] = v.re.limbs[0]; aj_rl[j] = v.re.limbs[1];
            aj_ih[j] = -v.im.limbs[0]; aj_il[j] = -v.im.limbs[1];   /* conj */
        }
        for (int j = j_count; j < kSimdLane; ++j) {
            aj_rh[j] = 0.0; aj_rl[j] = 0.0;
            aj_ih[j] = 0.0; aj_il[j] = 0.0;
        }
        __m256d ajh = _mm256_load_pd(aj_rh);
        __m256d ajl = _mm256_load_pd(aj_rl);
        __m256d aji = _mm256_load_pd(aj_ih);
        __m256d ajli = _mm256_load_pd(aj_il);
        /* t = alpha · conj(A_jpanel,l) — alpha is real DD: scale each limb */
        __m256d trh, trl, tih, til;
        simd_dd::dd_mul(a_h, a_l, ajh, ajl, trh, trl);
        simd_dd::dd_mul(a_h, a_l, aji, ajli, tih, til);
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

/* TR='C' dot product SIMD, KC-tiled: accumulate the conjugated complex DD dot
 * Σ conj(Ai[l])·Aj over l ∈ [l0, l0+kc) into the per-row 4-wide accumulator acc.
 * ajrh/.../ajil hold this chunk's 4 packed Aj columns (un-conjugated) at
 * chunk-local rows 0..kc-1. acc is loaded/stored each call, so accumulation
 * continues across chunks in the same order as a single l=0..K-1 loop →
 * bit-identical to the untiled path. Alpha is applied once after all chunks. */
inline void simd_herk_diag_c_chunk(int jc, int jb, int kc,
                                   const T *a, int lda, int l0,
                                   const double *ajrh, const double *ajrl,
                                   const double *ajih, const double *ajil,
                                   double *acc_rh, double *acc_rl,
                                   double *acc_ih, double *acc_il)
{
    for (int i = jc; i < jc + jb; ++i) {
        const int ir = i - jc;
        const T *Ai = a + static_cast<std::size_t>(i) * lda;
        __m256d srh = _mm256_load_pd(&acc_rh[ir * kSimdLane]);
        __m256d srl = _mm256_load_pd(&acc_rl[ir * kSimdLane]);
        __m256d sih = _mm256_load_pd(&acc_ih[ir * kSimdLane]);
        __m256d sil = _mm256_load_pd(&acc_il[ir * kSimdLane]);
        for (int ll = 0; ll < kc; ++ll) {
            /* conj(Ai[l0+ll]) broadcast */
            __m256d aih = _mm256_set1_pd(Ai[l0 + ll].re.limbs[0]);
            __m256d ail_ = _mm256_set1_pd(Ai[l0 + ll].re.limbs[1]);
            __m256d aiih = _mm256_set1_pd(-Ai[l0 + ll].im.limbs[0]);
            __m256d aiil = _mm256_set1_pd(-Ai[l0 + ll].im.limbs[1]);
            __m256d ajh = _mm256_load_pd(&ajrh[ll * kSimdLane]);
            __m256d ajl = _mm256_load_pd(&ajrl[ll * kSimdLane]);
            __m256d aji = _mm256_load_pd(&ajih[ll * kSimdLane]);
            __m256d ajli = _mm256_load_pd(&ajil[ll * kSimdLane]);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(aih, ail_, aiih, aiil,
                             ajh, ajl, aji, ajli,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(srh, srl, sih, sil,
                             prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            srh = nrh; srl = nrl; sih = nih; sil = nil_;
        }
        _mm256_store_pd(&acc_rh[ir * kSimdLane], srh);
        _mm256_store_pd(&acc_rl[ir * kSimdLane], srl);
        _mm256_store_pd(&acc_ih[ir * kSimdLane], sih);
        _mm256_store_pd(&acc_il[ir * kSimdLane], sil);
    }
}

inline void simd_herk_diag_panels(int jc, int jb, int K, R alpha,
                                  const T *a, int lda,
                                  T *c, int ldc,
                                  char UPLO, char TR_c)
{
    alignas(32) double crh[kMaxBlockM * kSimdLane];
    alignas(32) double crl[kMaxBlockM * kSimdLane];
    alignas(32) double cih[kMaxBlockM * kSimdLane];
    alignas(32) double cil[kMaxBlockM * kSimdLane];
    /* TR='C' scratch: one K-chunk of 4 packed Aj columns (bounded by kMaxK)
     * plus a per-row complex-DD accumulator carried across chunks. */
    alignas(32) static thread_local double ajrh[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajrl[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajih[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajil[kMaxK * kSimdLane];
    alignas(32) double acc_rh[kMaxBlockM * kSimdLane];
    alignas(32) double acc_rl[kMaxBlockM * kSimdLane];
    alignas(32) double acc_ih[kMaxBlockM * kSimdLane];
    alignas(32) double acc_il[kMaxBlockM * kSimdLane];

    for (int j = jc; j < jc + jb; j += kSimdLane) {
        const int jcount = (jc + jb - j < kSimdLane) ? (jc + jb - j) : kSimdLane;
        pack_4col_cdd(jb, jc, c, ldc, j, jcount, crh, crl, cih, cil);
        if (TR_c == 'N') {
            /* TR='N' reads A directly per l — K-independent, no scratch cap. */
            simd_herk_diag_n(jc, jb, K, alpha, a, lda, j, jcount,
                             crh, crl, cih, cil);
        } else {
            const __m256d zv = _mm256_setzero_pd();
            for (int ir = 0; ir < jb; ++ir) {
                _mm256_store_pd(&acc_rh[ir * kSimdLane], zv);
                _mm256_store_pd(&acc_rl[ir * kSimdLane], zv);
                _mm256_store_pd(&acc_ih[ir * kSimdLane], zv);
                _mm256_store_pd(&acc_il[ir * kSimdLane], zv);
            }
            /* KC-tile over K so any K fits the bounded pre-pack scratch. */
            for (int l0 = 0; l0 < K; l0 += kMaxK) {
                const int kc = (K - l0 < kMaxK) ? (K - l0) : kMaxK;
                for (int jj = 0; jj < jcount; ++jj) {
                    const T *col = a + static_cast<std::size_t>(j + jj) * lda;
                    for (int ll = 0; ll < kc; ++ll) {
                        ajrh[ll * kSimdLane + jj] = col[l0 + ll].re.limbs[0];
                        ajrl[ll * kSimdLane + jj] = col[l0 + ll].re.limbs[1];
                        ajih[ll * kSimdLane + jj] = col[l0 + ll].im.limbs[0];
                        ajil[ll * kSimdLane + jj] = col[l0 + ll].im.limbs[1];
                    }
                }
                for (int jj = jcount; jj < kSimdLane; ++jj)
                    for (int ll = 0; ll < kc; ++ll) {
                        ajrh[ll * kSimdLane + jj] = 0.0; ajrl[ll * kSimdLane + jj] = 0.0;
                        ajih[ll * kSimdLane + jj] = 0.0; ajil[ll * kSimdLane + jj] = 0.0;
                    }
                simd_herk_diag_c_chunk(jc, jb, kc, a, lda, l0,
                                       ajrh, ajrl, ajih, ajil,
                                       acc_rh, acc_rl, acc_ih, acc_il);
            }
            /* Finalize: C[panel] += alpha · acc (alpha real, scale each limb). */
            const __m256d a_h = _mm256_set1_pd(alpha.limbs[0]);
            const __m256d a_l = _mm256_set1_pd(alpha.limbs[1]);
            for (int i = jc; i < jc + jb; ++i) {
                const int ir = i - jc;
                __m256d srh = _mm256_load_pd(&acc_rh[ir * kSimdLane]);
                __m256d srl = _mm256_load_pd(&acc_rl[ir * kSimdLane]);
                __m256d sih = _mm256_load_pd(&acc_ih[ir * kSimdLane]);
                __m256d sil = _mm256_load_pd(&acc_il[ir * kSimdLane]);
                __m256d prh, prl, pih, pil;
                simd_dd::dd_mul(a_h, a_l, srh, srl, prh, prl);
                simd_dd::dd_mul(a_h, a_l, sih, sil, pih, pil);
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
        unpack_4col_herk_triangle(jc, jb, j, jcount, UPLO, c, ldc,
                                  crh, crl, cih, cil);
    }
}

#endif  /* MBLAS_SIMD_DD */

void herk_diag_add(int jc, int jb, int K, R alpha,
                   const T *a, int lda, T *c, int ldc,
                   char UPLO, char TR_c)
{
    if (TR_c == 'N') {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            for (int l = 0; l < K; ++l) {
                const T ajl = A_(j, l);
                if (!cdd_iszero(ajl)) {
                    const T t = rcmul(alpha, cconj(ajl));
                    for (int i = i_lo; i < i_hi; ++i) {
                        T prod = cmul(t, A_(i, l));
                        if (i == j) cj[i] = T{ cj[i].re + prod.re, cj[i].im };
                        else        cj[i] = cadd(cj[i], prod);
                    }
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
                T s = czero;
                for (int l = 0; l < K; ++l) s = cadd(s, cmul(cconj(Ai[l]), Aj[l]));
                T as = rcmul(alpha, s);
                if (i == j) cj[i] = T{ cj[i].re + as.re, cj[i].im };
                else        cj[i] = cadd(cj[i], as);
            }
        }
    }
}

inline void diag_dispatch(int jc, int jb, int K, R alpha,
                          const T *a, int lda, T *c, int ldc,
                          char UPLO, char TR_c)
{
#ifdef MBLAS_SIMD_DD
    if (jb <= kMaxBlockM) {
        simd_herk_diag_panels(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR_c);
        return;
    }
#endif
    herk_diag_add(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR_c);
}

} /* anonymous namespace */

int wherk_block_nb(void) {
    static int nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void wherk_zero_diag_im(int j, T *c, int ldc) {
    c[static_cast<std::size_t>(j) * ldc + j].im = rzero;
}

void wherk_scale_col(int j, int N, char UPLO, R beta, T *c, int ldc) {
    const int i_lo = (UPLO == 'L') ? j : 0;
    const int i_hi = (UPLO == 'L') ? N : j + 1;
    T *cj = c + static_cast<std::size_t>(j) * ldc;
    if (dd_iszero(beta)) {
        for (int i = i_lo; i < i_hi; ++i) cj[i] = czero;
    } else {
        for (int i = i_lo; i < i_hi; ++i) {
            if (i == j) cj[i] = T{ beta * cj[i].re, rzero };
            else        cj[i] = rcmul(beta, cj[i]);
        }
    }
}

void wherk_block(int jc, int jb, int N, int K, char UPLO, char TR,
                 R alpha, R beta, const T *a, int lda, T *c, int ldc)
{
    /* Beta-scale this block's own triangle columns (real-diag preservation). */
    for (int j = jc; j < jc + jb; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (dd_iszero(beta)) {
            for (int i = i_lo; i < i_hi; ++i) cj[i] = czero;
        } else if (!dd_isone(beta)) {
            for (int i = i_lo; i < i_hi; ++i) {
                if (i == j) cj[i] = T{ beta * cj[i].re, rzero };
                else        cj[i] = rcmul(beta, cj[i]);
            }
        } else {
            cj[j].im = rzero;
        }
    }

    diag_dispatch(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);

    const T alpha_c = T{ alpha, rzero };
    const T cone { R{1.0, 0.0}, rzero };
    const char NN[1] = {'N'};
    const char CN[1] = {'C'};

    if (UPLO == 'L') {
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            const int j0 = jc + jb;
            if (TR == 'N') {
                wgemm_serial(NN, CN, &trailing, &jb, &K, &alpha_c,
                             &A_(j0, 0), &lda, &A_(jc, 0), &lda,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
            } else {
                wgemm_serial(CN, NN, &trailing, &jb, &K, &alpha_c,
                             &A_(0, j0), &lda, &A_(0, jc), &lda,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
            }
        }
    } else {
        if (jc > 0) {
            if (TR == 'N') {
                wgemm_serial(NN, CN, &jc, &jb, &K, &alpha_c,
                             &A_(0, 0), &lda, &A_(jc, 0), &lda,
                             &cone, &C_(0, jc), &ldc, 1, 1);
            } else {
                wgemm_serial(CN, NN, &jc, &jb, &K, &alpha_c,
                             &A_(0, 0), &lda, &A_(0, jc), &lda,
                             &cone, &C_(0, jc), &ldc, 1, 1);
            }
        }
    }
}

extern "C" void wherk_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const R *alpha_,
    const T *a, const int *lda_,
    const R *beta_,
    T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldc = *ldc_;
    const R alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    const char TR_c = up(trans);

    if (N == 0) return;

    if (dd_iszero(alpha) || K == 0) {
        if (dd_isone(beta)) {
            for (int j = 0; j < N; ++j) wherk_zero_diag_im(j, c, ldc);
            return;
        }
        for (int j = 0; j < N; ++j) wherk_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const int nb = wherk_block_nb();
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        wherk_block(jc, jb, N, K, UPLO, TR_c, alpha, beta, a, lda, c, ldc);
    }
}

#undef A_
#undef C_
