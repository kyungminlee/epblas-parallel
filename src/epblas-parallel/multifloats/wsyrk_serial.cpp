/*
 * wsyrk_serial — multifloats complex (DD) symmetric rank-k update, pure
 * single-thread worker. Owns ALL the numerics; no OpenMP on this path.
 * TRANS ∈ {N, T}.
 *
 * AVX2 4-wide SIMD diag kernel + wgemm_serial trailing. Complex DD analog of
 * msyrk's SIMD diag (cmul / cadd primitives, 4 SoA arrays per packed
 * matrix). Computes the full square diag block and writes only the UPLO
 * triangle on unpack.
 *
 * The trailing rank-k update routes through wgemm_serial (no nested OpenMP) so
 * wsyrk_parallel.cpp can call wsyrk_block from inside its own omp region.
 */
#include "wsyrk_kernel.h"
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */
#include "wgemm_kernel.h"
#include <cstddef>
#include <cstdlib>
#include <cctype>

#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {


const TC zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
const TC one_cdd { R{1.0, 0.0}, R{0.0, 0.0} };


using mf_kernels::cmul;
using mf_kernels::cadd;

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define C_(i, j)  c[static_cast<std::size_t>(j) * ldc + (i)]

#ifdef MBLAS_SIMD_DD

/* AVX2+FMA under a possibly pre-Haswell baseline -march: these SIMD kernels and
 * their helpers are compiled with the feature enabled and reached only behind
 * mf_have_avx2_fma() at the call sites below. See mf_dispatch.h. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")

constexpr std::ptrdiff_t kSimdLane = simd_fast::NR;
constexpr std::ptrdiff_t kMaxBlockM = 128;
constexpr std::ptrdiff_t kMaxK      = 512;

inline void pack_4col_cdd(std::ptrdiff_t count, std::ptrdiff_t row_start,
                          const TC *m, std::ptrdiff_t ldm, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                          double *rh, double *rl, double *ih, double *il)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        const TC *col = m + static_cast<std::size_t>(j_start + j) * ldm;
        for (std::ptrdiff_t i = 0; i < count; ++i) {
            rh[i * kSimdLane + j] = col[row_start + i].re.limbs[0];
            rl[i * kSimdLane + j] = col[row_start + i].re.limbs[1];
            ih[i * kSimdLane + j] = col[row_start + i].im.limbs[0];
            il[i * kSimdLane + j] = col[row_start + i].im.limbs[1];
        }
    }
    for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j)
        for (std::ptrdiff_t i = 0; i < count; ++i) {
            rh[i * kSimdLane + j] = 0.0; rl[i * kSimdLane + j] = 0.0;
            ih[i * kSimdLane + j] = 0.0; il[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_4col_cdd_triangle(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                                     char UPLO, TC *c, std::ptrdiff_t ldc,
                                     const double *rh, const double *rl,
                                     const double *ih, const double *il)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        const std::ptrdiff_t j_abs = j_start + j;
        const std::ptrdiff_t i_lo = (UPLO == 'L') ? j_abs   : jc;
        const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc + jb : j_abs + 1;
        TC *col = c + static_cast<std::size_t>(j_abs) * ldc;
        for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
            const std::ptrdiff_t ir = i - jc;
            col[i].re.limbs[0] = rh[ir * kSimdLane + j];
            col[i].re.limbs[1] = rl[ir * kSimdLane + j];
            col[i].im.limbs[0] = ih[ir * kSimdLane + j];
            col[i].im.limbs[1] = il[ir * kSimdLane + j];
        }
    }
}

using simd_exact::vbcast;

/* TRANS='N' rank-1: for each l, load α·A(j_panel..+4, l) and update
 * C[i, j_panel..+4] += t · A(i, l) across i ∈ diag block. */
inline void simd_syrk_diag_tn(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                              const TC *a, std::ptrdiff_t lda,
                              std::ptrdiff_t j_panel, std::ptrdiff_t j_count,
                              double *crh, double *crl,
                              double *cih, double *cil)
{
    __m256d a_rh, a_rl, a_ih, a_il;
    vbcast(alpha, a_rh, a_rl, a_ih, a_il);
    alignas(32) double bj_rh[kSimdLane], bj_rl[kSimdLane];
    alignas(32) double bj_ih[kSimdLane], bj_il[kSimdLane];
    for (std::ptrdiff_t l = 0; l < k; ++l) {
        for (std::ptrdiff_t j = 0; j < j_count; ++j) {
            const TC v = A_(j_panel + j, l);
            bj_rh[j] = v.re.limbs[0]; bj_rl[j] = v.re.limbs[1];
            bj_ih[j] = v.im.limbs[0]; bj_il[j] = v.im.limbs[1];
        }
        for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j) {
            bj_rh[j] = 0.0; bj_rl[j] = 0.0;
            bj_ih[j] = 0.0; bj_il[j] = 0.0;
        }
        __m256d aj_rh = _mm256_load_pd(bj_rh);
        __m256d aj_rl = _mm256_load_pd(bj_rl);
        __m256d aj_ih = _mm256_load_pd(bj_ih);
        __m256d aj_il = _mm256_load_pd(bj_il);
        __m256d trh, trl, tih, til;
        simd_fast::cmul(a_rh, a_rl, a_ih, a_il,
                         aj_rh, aj_rl, aj_ih, aj_il,
                         trh, trl, tih, til);
        for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
            const std::ptrdiff_t ir = i - jc;
            __m256d aih, ail, aiih, aiil;
            vbcast(A_(i, l), aih, ail, aiih, aiil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(trh, trl, tih, til,
                             aih, ail, aiih, aiil,
                             prh, prl, pih, pil);
            __m256d ck_rh = _mm256_load_pd(&crh[ir * kSimdLane]);
            __m256d ck_rl = _mm256_load_pd(&crl[ir * kSimdLane]);
            __m256d ck_ih = _mm256_load_pd(&cih[ir * kSimdLane]);
            __m256d ck_il = _mm256_load_pd(&cil[ir * kSimdLane]);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(ck_rh, ck_rl, ck_ih, ck_il,
                             prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            _mm256_store_pd(&crh[ir * kSimdLane], nrh);
            _mm256_store_pd(&crl[ir * kSimdLane], nrl);
            _mm256_store_pd(&cih[ir * kSimdLane], nih);
            _mm256_store_pd(&cil[ir * kSimdLane], nil_);
        }
    }
}

/* TRANS='T' dot product SIMD, KC-tiled: accumulate the complex DD dot over
 * l ∈ [l0, l0+kc) into the per-row 4-wide accumulator acc. ajrh/.../ajil
 * hold this chunk's 4 packed A columns at chunk-local rows 0..kc-1. acc is
 * loaded/stored each call, so accumulation continues across chunks in the
 * same order as a single l=0..K-1 loop → bit-identical to the untiled path. */
inline void simd_syrk_diag_tt_chunk(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t kc,
                                    const TC *a, std::ptrdiff_t lda, std::ptrdiff_t l0,
                                    const double *ajrh, const double *ajrl,
                                    const double *ajih, const double *ajil,
                                    double *acc_rh, double *acc_rl,
                                    double *acc_ih, double *acc_il)
{
    for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
        const std::ptrdiff_t ir = i - jc;
        const TC *Ai = a + static_cast<std::size_t>(i) * lda;
        __m256d srh = _mm256_load_pd(&acc_rh[ir * kSimdLane]);
        __m256d srl = _mm256_load_pd(&acc_rl[ir * kSimdLane]);
        __m256d sih = _mm256_load_pd(&acc_ih[ir * kSimdLane]);
        __m256d sil = _mm256_load_pd(&acc_il[ir * kSimdLane]);
        for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
            __m256d aih, ail, aiih, aiil;
            vbcast(Ai[l0 + ll], aih, ail, aiih, aiil);
            __m256d ajh = _mm256_load_pd(&ajrh[ll * kSimdLane]);
            __m256d ajl = _mm256_load_pd(&ajrl[ll * kSimdLane]);
            __m256d ajih_v = _mm256_load_pd(&ajih[ll * kSimdLane]);
            __m256d ajil_v = _mm256_load_pd(&ajil[ll * kSimdLane]);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(aih, ail, aiih, aiil,
                             ajh, ajl, ajih_v, ajil_v,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(srh, srl, sih, sil,
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

inline void simd_syrk_diag_panels(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                                  const TC *a, std::ptrdiff_t lda,
                                  TC *c, std::ptrdiff_t ldc,
                                  char UPLO, char TRANS)
{
    alignas(32) double crh[kMaxBlockM * kSimdLane];
    alignas(32) double crl[kMaxBlockM * kSimdLane];
    alignas(32) double cih[kMaxBlockM * kSimdLane];
    alignas(32) double cil[kMaxBlockM * kSimdLane];
    /* TRANS='T' scratch: one K-chunk of 4 packed A columns (bounded by kMaxK)
     * plus a per-row complex-DD accumulator carried across chunks. */
    alignas(32) static thread_local double ajrh[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajrl[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajih[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajil[kMaxK * kSimdLane];
    alignas(32) double acc_rh[kMaxBlockM * kSimdLane];
    alignas(32) double acc_rl[kMaxBlockM * kSimdLane];
    alignas(32) double acc_ih[kMaxBlockM * kSimdLane];
    alignas(32) double acc_il[kMaxBlockM * kSimdLane];

    for (std::ptrdiff_t j = jc; j < jc + jb; j += kSimdLane) {
        const std::ptrdiff_t jcount = (jc + jb - j < kSimdLane) ? (jc + jb - j) : kSimdLane;
        pack_4col_cdd(jb, jc, c, ldc, j, jcount, crh, crl, cih, cil);
        if (TRANS == 'N') {
            /* TRANS='N' reads A directly per l — K-independent, no scratch cap. */
            simd_syrk_diag_tn(jc, jb, k, alpha, a, lda, j, jcount,
                              crh, crl, cih, cil);
        } else {
            const __m256d zv = _mm256_setzero_pd();
            for (std::ptrdiff_t ir = 0; ir < jb; ++ir) {
                _mm256_store_pd(&acc_rh[ir * kSimdLane], zv);
                _mm256_store_pd(&acc_rl[ir * kSimdLane], zv);
                _mm256_store_pd(&acc_ih[ir * kSimdLane], zv);
                _mm256_store_pd(&acc_il[ir * kSimdLane], zv);
            }
            /* KC-tile over K so any K fits the bounded pre-pack scratch. */
            for (std::ptrdiff_t l0 = 0; l0 < k; l0 += kMaxK) {
                const std::ptrdiff_t kc = (k - l0 < kMaxK) ? (k - l0) : kMaxK;
                for (std::ptrdiff_t jj = 0; jj < jcount; ++jj) {
                    const TC *col = a + static_cast<std::size_t>(j + jj) * lda;
                    for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
                        ajrh[ll * kSimdLane + jj] = col[l0 + ll].re.limbs[0];
                        ajrl[ll * kSimdLane + jj] = col[l0 + ll].re.limbs[1];
                        ajih[ll * kSimdLane + jj] = col[l0 + ll].im.limbs[0];
                        ajil[ll * kSimdLane + jj] = col[l0 + ll].im.limbs[1];
                    }
                }
                for (std::ptrdiff_t jj = jcount; jj < kSimdLane; ++jj)
                    for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
                        ajrh[ll * kSimdLane + jj] = 0.0; ajrl[ll * kSimdLane + jj] = 0.0;
                        ajih[ll * kSimdLane + jj] = 0.0; ajil[ll * kSimdLane + jj] = 0.0;
                    }
                simd_syrk_diag_tt_chunk(jc, jb, kc, a, lda, l0,
                                        ajrh, ajrl, ajih, ajil,
                                        acc_rh, acc_rl, acc_ih, acc_il);
            }
            /* Finalize: C[panel] += alpha · acc (single alpha-mul, as untiled). */
            __m256d a_rh, a_rl, a_ih, a_il;
            vbcast(alpha, a_rh, a_rl, a_ih, a_il);
            for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
                const std::ptrdiff_t ir = i - jc;
                __m256d srh = _mm256_load_pd(&acc_rh[ir * kSimdLane]);
                __m256d srl = _mm256_load_pd(&acc_rl[ir * kSimdLane]);
                __m256d sih = _mm256_load_pd(&acc_ih[ir * kSimdLane]);
                __m256d sil = _mm256_load_pd(&acc_il[ir * kSimdLane]);
                __m256d prh, prl, pih, pil;
                simd_fast::cmul(a_rh, a_rl, a_ih, a_il,
                                 srh, srl, sih, sil,
                                 prh, prl, pih, pil);
                __m256d ck_rh = _mm256_load_pd(&crh[ir * kSimdLane]);
                __m256d ck_rl = _mm256_load_pd(&crl[ir * kSimdLane]);
                __m256d ck_ih = _mm256_load_pd(&cih[ir * kSimdLane]);
                __m256d ck_il = _mm256_load_pd(&cil[ir * kSimdLane]);
                __m256d nrh, nrl, nih, nil_;
                simd_fast::cadd(ck_rh, ck_rl, ck_ih, ck_il,
                                 prh, prl, pih, pil,
                                 nrh, nrl, nih, nil_);
                _mm256_store_pd(&crh[ir * kSimdLane], nrh);
                _mm256_store_pd(&crl[ir * kSimdLane], nrl);
                _mm256_store_pd(&cih[ir * kSimdLane], nih);
                _mm256_store_pd(&cil[ir * kSimdLane], nil_);
            }
        }
        unpack_4col_cdd_triangle(jc, jb, j, jcount, UPLO, c, ldc,
                                 crh, crl, cih, cil);
    }
}

#pragma GCC pop_options

#endif  /* MBLAS_SIMD_DD */

void syrk_diag_add(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                   const TC *a, std::ptrdiff_t lda, TC *c, std::ptrdiff_t ldc,
                   char UPLO, char TRANS)
{
    if (TRANS == 'N') {
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            const std::ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + static_cast<std::size_t>(j) * ldc;
            for (std::ptrdiff_t l = 0; l < k; ++l) {
                const TC ajl = A_(j, l);
                if (!ceq0(ajl)) {
                    const TC t = cmul(alpha, ajl);
                    for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = cadd(cj[i], cmul(t, A_(i, l)));
                }
            }
        }
    } else {
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            const std::ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + static_cast<std::size_t>(j) * ldc;
            const TC *Aj = a + static_cast<std::size_t>(j) * lda;
            for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                const TC *Ai = a + static_cast<std::size_t>(i) * lda;
                TC s = zero_cdd;
                for (std::ptrdiff_t l = 0; l < k; ++l) s = cadd(s, cmul(Ai[l], Aj[l]));
                cj[i] = cadd(cj[i], cmul(alpha, s));
            }
        }
    }
}

inline void diag_dispatch(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                          const TC *a, std::ptrdiff_t lda, TC *c, std::ptrdiff_t ldc,
                          char UPLO, char TRANS)
{
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma() && jb <= kMaxBlockM) {
        simd_syrk_diag_panels(jc, jb, k, alpha, a, lda, c, ldc, UPLO, TRANS);
        return;
    }
#endif
    syrk_diag_add(jc, jb, k, alpha, a, lda, c, ldc, UPLO, TRANS);
}

} /* anonymous namespace */

std::ptrdiff_t wsyrk_block_nb(void) {
    static std::ptrdiff_t nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void wsyrk_scale_col(std::ptrdiff_t j, std::ptrdiff_t n, char UPLO, TC beta, TC *c, std::ptrdiff_t ldc) {
    const std::ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
    const std::ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
    TC *cj = c + static_cast<std::size_t>(j) * ldc;
    if (ceq0(beta)) for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = zero_cdd;
    else                  for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = cmul(cj[i], beta);
}

void wsyrk_block(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t n, std::ptrdiff_t k, char UPLO, char TRANS,
                 TC alpha, TC beta, const TC *a, std::ptrdiff_t lda, TC *c, std::ptrdiff_t ldc)
{
    /* Beta-scale this block's own triangle columns. */
    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        const std::ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const std::ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
        TC *cj = c + static_cast<std::size_t>(j) * ldc;
        if (ceq0(beta))      for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = zero_cdd;
        else if (!ceq1(beta)) for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = cmul(cj[i], beta);
    }

    diag_dispatch(jc, jb, k, alpha, a, lda, c, ldc, UPLO, TRANS);

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    if (UPLO == 'L') {
        const std::ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            const std::ptrdiff_t j0 = jc + jb;
            if (TRANS == 'N') {
                wgemm_serial(NN[0], TN[0], trailing, jb, k, &alpha, &A_(j0, 0), lda, &A_(jc, 0), lda, &one_cdd, &C_(j0, jc), ldc);
            } else {
                wgemm_serial(TN[0], NN[0], trailing, jb, k, &alpha, &A_(0, j0), lda, &A_(0, jc), lda, &one_cdd, &C_(j0, jc), ldc);
            }
        }
    } else {
        if (jc > 0) {
            if (TRANS == 'N') {
                wgemm_serial(NN[0], TN[0], jc, jb, k, &alpha, &A_(0, 0), lda, &A_(jc, 0), lda, &one_cdd, &C_(0, jc), ldc);
            } else {
                wgemm_serial(TN[0], NN[0], jc, jb, k, &alpha, &A_(0, 0), lda, &A_(0, jc), lda, &one_cdd, &C_(0, jc), ldc);
            }
        }
    }
}

extern "C" void wsyrk_serial(
    char uplo, char trans,
    std::ptrdiff_t n, std::ptrdiff_t k,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    const TC *beta_,
    TC *c, std::ptrdiff_t ldc)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char UPLO = up(&uplo);
    const char TRANS = up(&trans);

    if (n == 0) return;

    if (ceq0(alpha) || k == 0) {
        if (ceq1(beta)) return;
        for (std::ptrdiff_t j = 0; j < n; ++j) wsyrk_scale_col(j, n, UPLO, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = wsyrk_block_nb();
    for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
        const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        wsyrk_block(jc, jb, n, k, UPLO, TRANS, alpha, beta, a, lda, c, ldc);
    }
}

#undef A_
#undef C_
