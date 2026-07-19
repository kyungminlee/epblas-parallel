/*
 * wsyr2k_serial — multifloats complex (DD) symmetric rank-2k update, pure
 * single-thread worker. Owns ALL the numerics; no OpenMP on this path.
 *
 *   C := alpha · (A · Bᵀ + B · Aᵀ) + beta · C        (TRANS='N')
 *   C := alpha · (Aᵀ · B + Bᵀ · A) + beta · C        (TRANS='T'/'C')
 *
 * Complex SYMMETRIC (not Hermitian — see wher2k): no conjugation, plain 'T'
 * trailing gemms. Blocked: AVX2 SIMD (or scalar) rank-2 diagonal kernel + two
 * wgemm trailing calls per off-diagonal wing. The trailing gemms route through
 * wgemm_serial (no nested OpenMP) so wsyr2k_parallel.cpp can call the block
 * worker from inside its own omp region.
 */
#include "wsyr2k_kernel.h"
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#include "wgemm_kernel.h"
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */

#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h */
namespace {


using mf_pred::zero_cdd;   /* shared DD constants — mf_pred.h */
using mf_pred::one_cdd;


using mf_kernels::cmul;
using mf_kernels::cadd;

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]
#define C_(i, j)  c[static_cast<std::size_t>(j) * ldc + (i)]

#ifdef MBLAS_SIMD_DD

/* AVX2+FMA under a possibly pre-Haswell baseline -march: these SIMD kernels and
 * their helpers are compiled with the feature enabled and reached only behind
 * mf_have_avx2_fma() at the call site below. See mf_dispatch.h. */
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

/* TRANS='N' rank-2 update: t1 = α·A(j_panel..+4, l), t2 = α·B(j_panel..+4, l);
 * C[i, panel] += B(i,l)·t1 + A(i,l)·t2 across i ∈ diag block. */
inline void simd_syr2k_diag_tn(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                               const TC *a, std::ptrdiff_t lda, const TC *b, std::ptrdiff_t ldb,
                               std::ptrdiff_t j_panel, std::ptrdiff_t j_count,
                               double *crh, double *crl,
                               double *cih, double *cil)
{
    __m256d a_rh, a_rl, a_ih, a_il;
    vbcast(alpha, a_rh, a_rl, a_ih, a_il);
    alignas(32) double aj_rh[kSimdLane], aj_rl[kSimdLane], aj_ih[kSimdLane], aj_il[kSimdLane];
    alignas(32) double bj_rh[kSimdLane], bj_rl[kSimdLane], bj_ih[kSimdLane], bj_il[kSimdLane];
    for (std::ptrdiff_t ll = 0; ll < k; ++ll) {
        for (std::ptrdiff_t j = 0; j < j_count; ++j) {
            const TC av = A_(j_panel + j, ll);
            const TC bv = B_(j_panel + j, ll);
            aj_rh[j] = av.re.limbs[0]; aj_rl[j] = av.re.limbs[1];
            aj_ih[j] = av.im.limbs[0]; aj_il[j] = av.im.limbs[1];
            bj_rh[j] = bv.re.limbs[0]; bj_rl[j] = bv.re.limbs[1];
            bj_ih[j] = bv.im.limbs[0]; bj_il[j] = bv.im.limbs[1];
        }
        for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j) {
            aj_rh[j] = 0.0; aj_rl[j] = 0.0; aj_ih[j] = 0.0; aj_il[j] = 0.0;
            bj_rh[j] = 0.0; bj_rl[j] = 0.0; bj_ih[j] = 0.0; bj_il[j] = 0.0;
        }
        __m256d ajrh = _mm256_load_pd(aj_rh), ajrl = _mm256_load_pd(aj_rl);
        __m256d ajih = _mm256_load_pd(aj_ih), ajil = _mm256_load_pd(aj_il);
        __m256d bjrh = _mm256_load_pd(bj_rh), bjrl = _mm256_load_pd(bj_rl);
        __m256d bjih = _mm256_load_pd(bj_ih), bjil = _mm256_load_pd(bj_il);
        __m256d t1rh, t1rl, t1ih, t1il, t2rh, t2rl, t2ih, t2il;
        simd_fast::cmul(a_rh, a_rl, a_ih, a_il, ajrh, ajrl, ajih, ajil, t1rh, t1rl, t1ih, t1il);
        simd_fast::cmul(a_rh, a_rl, a_ih, a_il, bjrh, bjrl, bjih, bjil, t2rh, t2rl, t2ih, t2il);
        for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
            const std::ptrdiff_t ir = i - jc;
            __m256d aih, ail_, aiih, aiil;
            __m256d bih, bil_, biih, biil;
            vbcast(A_(i, ll), aih, ail_, aiih, aiil);
            vbcast(B_(i, ll), bih, bil_, biih, biil);
            __m256d p1rh, p1rl, p1ih, p1il, p2rh, p2rl, p2ih, p2il;
            simd_fast::cmul(bih, bil_, biih, biil, t1rh, t1rl, t1ih, t1il, p1rh, p1rl, p1ih, p1il);
            simd_fast::cmul(aih, ail_, aiih, aiil, t2rh, t2rl, t2ih, t2il, p2rh, p2rl, p2ih, p2il);
            __m256d srh, srl, sih, sil;
            simd_fast::cadd(p1rh, p1rl, p1ih, p1il, p2rh, p2rl, p2ih, p2il, srh, srl, sih, sil);
            __m256d ckrh = _mm256_load_pd(&crh[ir * kSimdLane]);
            __m256d ckrl = _mm256_load_pd(&crl[ir * kSimdLane]);
            __m256d ckih = _mm256_load_pd(&cih[ir * kSimdLane]);
            __m256d ckil = _mm256_load_pd(&cil[ir * kSimdLane]);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(ckrh, ckrl, ckih, ckil, srh, srl, sih, sil, nrh, nrl, nih, nil_);
            _mm256_store_pd(&crh[ir * kSimdLane], nrh);
            _mm256_store_pd(&crl[ir * kSimdLane], nrl);
            _mm256_store_pd(&cih[ir * kSimdLane], nih);
            _mm256_store_pd(&cil[ir * kSimdLane], nil_);
        }
    }
}

/* TRANS='T' dot product SIMD, KC-tiled: accumulate Ai(l)·Bj + Bi(l)·Aj (complex
 * DD) over l ∈ [l0, l0+kc) into the per-row 4-wide accumulator acc. ajrh/.../bjil
 * hold this chunk's 4 packed A & B columns at chunk-local rows 0..kc-1. acc is
 * loaded/stored each call, so accumulation continues across chunks in the
 * same order as a single l=0..K-1 loop → bit-identical to the untiled path. */
inline void simd_syr2k_diag_tt_chunk(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t kc,
                                     const TC *a, std::ptrdiff_t lda, const TC *b, std::ptrdiff_t ldb,
                                     std::ptrdiff_t l0,
                                     const double *ajrh, const double *ajrl,
                                     const double *ajih, const double *ajil,
                                     const double *bjrh, const double *bjrl,
                                     const double *bjih, const double *bjil,
                                     double *acc_rh, double *acc_rl,
                                     double *acc_ih, double *acc_il)
{
    for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
        const std::ptrdiff_t ir = i - jc;
        const TC *Ai = a + static_cast<std::size_t>(i) * lda;
        const TC *Bi = b + static_cast<std::size_t>(i) * ldb;
        __m256d srh = _mm256_load_pd(&acc_rh[ir * kSimdLane]);
        __m256d srl = _mm256_load_pd(&acc_rl[ir * kSimdLane]);
        __m256d sih = _mm256_load_pd(&acc_ih[ir * kSimdLane]);
        __m256d sil = _mm256_load_pd(&acc_il[ir * kSimdLane]);
        for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
            const std::ptrdiff_t l = l0 + ll;
            __m256d aih, ail_, aiih, aiil;
            __m256d bih, bil_, biih, biil;
            vbcast(Ai[l], aih, ail_, aiih, aiil);
            vbcast(Bi[l], bih, bil_, biih, biil);
            __m256d ajrv = _mm256_load_pd(&ajrh[ll * kSimdLane]);
            __m256d ajrlv = _mm256_load_pd(&ajrl[ll * kSimdLane]);
            __m256d ajiv = _mm256_load_pd(&ajih[ll * kSimdLane]);
            __m256d ajilv = _mm256_load_pd(&ajil[ll * kSimdLane]);
            __m256d bjrv = _mm256_load_pd(&bjrh[ll * kSimdLane]);
            __m256d bjrlv = _mm256_load_pd(&bjrl[ll * kSimdLane]);
            __m256d bjiv = _mm256_load_pd(&bjih[ll * kSimdLane]);
            __m256d bjilv = _mm256_load_pd(&bjil[ll * kSimdLane]);
            __m256d p1rh, p1rl, p1ih, p1il, p2rh, p2rl, p2ih, p2il;
            simd_fast::cmul(aih, ail_, aiih, aiil, bjrv, bjrlv, bjiv, bjilv,
                             p1rh, p1rl, p1ih, p1il);
            simd_fast::cmul(bih, bil_, biih, biil, ajrv, ajrlv, ajiv, ajilv,
                             p2rh, p2rl, p2ih, p2il);
            __m256d sumrh, sumrl, sumih, sumil;
            simd_fast::cadd(p1rh, p1rl, p1ih, p1il, p2rh, p2rl, p2ih, p2il,
                             sumrh, sumrl, sumih, sumil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(srh, srl, sih, sil, sumrh, sumrl, sumih, sumil,
                             nrh, nrl, nih, nil_);
            srh = nrh; srl = nrl; sih = nih; sil = nil_;
        }
        _mm256_store_pd(&acc_rh[ir * kSimdLane], srh);
        _mm256_store_pd(&acc_rl[ir * kSimdLane], srl);
        _mm256_store_pd(&acc_ih[ir * kSimdLane], sih);
        _mm256_store_pd(&acc_il[ir * kSimdLane], sil);
    }
}

inline void simd_syr2k_diag_panels(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                                   const TC *a, std::ptrdiff_t lda, const TC *b, std::ptrdiff_t ldb,
                                   TC *c, std::ptrdiff_t ldc, char UPLO, char TRANS)
{
    alignas(32) double crh[kMaxBlockM * kSimdLane], crl[kMaxBlockM * kSimdLane];
    alignas(32) double cih[kMaxBlockM * kSimdLane], cil[kMaxBlockM * kSimdLane];
    /* TRANS='T' scratch: one K-chunk of 4 packed A & B columns (bounded by
     * kMaxK) plus a per-row complex-DD accumulator carried across chunks. */
    alignas(32) static thread_local double ajrh[kMaxK * kSimdLane], ajrl[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajih[kMaxK * kSimdLane], ajil[kMaxK * kSimdLane];
    alignas(32) static thread_local double bjrh[kMaxK * kSimdLane], bjrl[kMaxK * kSimdLane];
    alignas(32) static thread_local double bjih[kMaxK * kSimdLane], bjil[kMaxK * kSimdLane];
    alignas(32) double acc_rh[kMaxBlockM * kSimdLane], acc_rl[kMaxBlockM * kSimdLane];
    alignas(32) double acc_ih[kMaxBlockM * kSimdLane], acc_il[kMaxBlockM * kSimdLane];

    for (std::ptrdiff_t j = jc; j < jc + jb; j += kSimdLane) {
        const std::ptrdiff_t jcount = (jc + jb - j < kSimdLane) ? (jc + jb - j) : kSimdLane;
        pack_4col_cdd(jb, jc, c, ldc, j, jcount, crh, crl, cih, cil);
        if (TRANS == 'N') {
            simd_syr2k_diag_tn(jc, jb, k, alpha, a, lda, b, ldb, j, jcount,
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
                    const TC *acol = a + static_cast<std::size_t>(j + jj) * lda;
                    const TC *bcol = b + static_cast<std::size_t>(j + jj) * ldb;
                    for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
                        ajrh[ll * kSimdLane + jj] = acol[l0 + ll].re.limbs[0];
                        ajrl[ll * kSimdLane + jj] = acol[l0 + ll].re.limbs[1];
                        ajih[ll * kSimdLane + jj] = acol[l0 + ll].im.limbs[0];
                        ajil[ll * kSimdLane + jj] = acol[l0 + ll].im.limbs[1];
                        bjrh[ll * kSimdLane + jj] = bcol[l0 + ll].re.limbs[0];
                        bjrl[ll * kSimdLane + jj] = bcol[l0 + ll].re.limbs[1];
                        bjih[ll * kSimdLane + jj] = bcol[l0 + ll].im.limbs[0];
                        bjil[ll * kSimdLane + jj] = bcol[l0 + ll].im.limbs[1];
                    }
                }
                for (std::ptrdiff_t jj = jcount; jj < kSimdLane; ++jj)
                    for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
                        ajrh[ll * kSimdLane + jj] = 0.0; ajrl[ll * kSimdLane + jj] = 0.0;
                        ajih[ll * kSimdLane + jj] = 0.0; ajil[ll * kSimdLane + jj] = 0.0;
                        bjrh[ll * kSimdLane + jj] = 0.0; bjrl[ll * kSimdLane + jj] = 0.0;
                        bjih[ll * kSimdLane + jj] = 0.0; bjil[ll * kSimdLane + jj] = 0.0;
                    }
                simd_syr2k_diag_tt_chunk(jc, jb, kc, a, lda, b, ldb, l0,
                                         ajrh, ajrl, ajih, ajil,
                                         bjrh, bjrl, bjih, bjil,
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
                simd_fast::cmul(a_rh, a_rl, a_ih, a_il, srh, srl, sih, sil,
                                 prh, prl, pih, pil);
                __m256d ckrh = _mm256_load_pd(&crh[ir * kSimdLane]);
                __m256d ckrl = _mm256_load_pd(&crl[ir * kSimdLane]);
                __m256d ckih = _mm256_load_pd(&cih[ir * kSimdLane]);
                __m256d ckil = _mm256_load_pd(&cil[ir * kSimdLane]);
                __m256d nrh, nrl, nih, nil_;
                simd_fast::cadd(ckrh, ckrl, ckih, ckil, prh, prl, pih, pil,
                                 nrh, nrl, nih, nil_);
                _mm256_store_pd(&crh[ir * kSimdLane], nrh);
                _mm256_store_pd(&crl[ir * kSimdLane], nrl);
                _mm256_store_pd(&cih[ir * kSimdLane], nih);
                _mm256_store_pd(&cil[ir * kSimdLane], nil_);
            }
        }
        unpack_4col_cdd_triangle(jc, jb, j, jcount, UPLO, c, ldc, crh, crl, cih, cil);
    }
}

#pragma GCC pop_options

#endif  /* MBLAS_SIMD_DD */

void syr2k_diag_add(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                    const TC *a, std::ptrdiff_t lda,
                    const TC *b, std::ptrdiff_t ldb,
                    TC *c, std::ptrdiff_t ldc,
                    char UPLO, char TRANS)
{
    if (TRANS == 'N') {
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            const std::ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + static_cast<std::size_t>(j) * ldc;
            for (std::ptrdiff_t l = 0; l < k; ++l) {
                const TC t1 = cmul(alpha, A_(j, l));
                const TC t2 = cmul(alpha, B_(j, l));
                for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                    cj[i] = cadd(cj[i], cadd(cmul(B_(i, l), t1), cmul(A_(i, l), t2)));
                }
            }
        }
    } else {
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            const std::ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + static_cast<std::size_t>(j) * ldc;
            const TC *Aj = a + static_cast<std::size_t>(j) * lda;
            const TC *Bj = b + static_cast<std::size_t>(j) * ldb;
            for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                const TC *Ai = a + static_cast<std::size_t>(i) * lda;
                const TC *Bi = b + static_cast<std::size_t>(i) * ldb;
                TC s = zero_cdd;
                for (std::ptrdiff_t l = 0; l < k; ++l) {
                    s = cadd(s, cadd(cmul(Ai[l], Bj[l]), cmul(Bi[l], Aj[l])));
                }
                cj[i] = cadd(cj[i], cmul(alpha, s));
            }
        }
    }
}

inline void diag_dispatch(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                          const TC *a, std::ptrdiff_t lda, const TC *b, std::ptrdiff_t ldb,
                          TC *c, std::ptrdiff_t ldc, char UPLO, char TRANS)
{
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma() && jb <= kMaxBlockM) {
        simd_syr2k_diag_panels(jc, jb, k, alpha, a, lda, b, ldb, c, ldc, UPLO, TRANS);
        return;
    }
#endif
    syr2k_diag_add(jc, jb, k, alpha, a, lda, b, ldb, c, ldc, UPLO, TRANS);
}

} /* anonymous namespace */

std::ptrdiff_t wsyr2k_block_nb(void) {
    static std::ptrdiff_t nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void wsyr2k_scale_col(std::ptrdiff_t j, std::ptrdiff_t n, char UPLO, TC beta, TC *c, std::ptrdiff_t ldc) {
    const std::ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
    const std::ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
    TC *cj = c + static_cast<std::size_t>(j) * ldc;
    if (ceq0(beta)) for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = zero_cdd;
    else                  for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = cmul(cj[i], beta);
}

void wsyr2k_block(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t n, std::ptrdiff_t k, char UPLO, char TRANS,
                  TC alpha, TC beta, const TC *a, std::ptrdiff_t lda, const TC *b, std::ptrdiff_t ldb,
                  TC *c, std::ptrdiff_t ldc)
{
    /* Beta-scale this block's own triangle columns. */
    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        const std::ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const std::ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
        TC *cj = c + static_cast<std::size_t>(j) * ldc;
        if (ceq0(beta))      for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = zero_cdd;
        else if (!ceq1(beta)) for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = cmul(cj[i], beta);
    }

    diag_dispatch(jc, jb, k, alpha, a, lda, b, ldb, c, ldc, UPLO, TRANS);

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    if (UPLO == 'L') {
        const std::ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            const std::ptrdiff_t j0 = jc + jb;
            if (TRANS == 'N') {
                wgemm_serial(NN[0], TN[0], trailing, jb, k, &alpha, &A_(j0, 0), lda, &B_(jc, 0), ldb, &one_cdd, &C_(j0, jc), ldc);
                wgemm_serial(NN[0], TN[0], trailing, jb, k, &alpha, &B_(j0, 0), ldb, &A_(jc, 0), lda, &one_cdd, &C_(j0, jc), ldc);
            } else {
                wgemm_serial(TN[0], NN[0], trailing, jb, k, &alpha, &A_(0, j0), lda, &B_(0, jc), ldb, &one_cdd, &C_(j0, jc), ldc);
                wgemm_serial(TN[0], NN[0], trailing, jb, k, &alpha, &B_(0, j0), ldb, &A_(0, jc), lda, &one_cdd, &C_(j0, jc), ldc);
            }
        }
    } else {
        if (jc > 0) {
            if (TRANS == 'N') {
                wgemm_serial(NN[0], TN[0], jc, jb, k, &alpha, &A_(0, 0), lda, &B_(jc, 0), ldb, &one_cdd, &C_(0, jc), ldc);
                wgemm_serial(NN[0], TN[0], jc, jb, k, &alpha, &B_(0, 0), ldb, &A_(jc, 0), lda, &one_cdd, &C_(0, jc), ldc);
            } else {
                wgemm_serial(TN[0], NN[0], jc, jb, k, &alpha, &A_(0, 0), lda, &B_(0, jc), ldb, &one_cdd, &C_(0, jc), ldc);
                wgemm_serial(TN[0], NN[0], jc, jb, k, &alpha, &B_(0, 0), ldb, &A_(0, jc), lda, &one_cdd, &C_(0, jc), ldc);
            }
        }
    }
}

extern "C" void wsyr2k_serial(
    char uplo, char trans,
    std::ptrdiff_t n, std::ptrdiff_t k,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    const TC *b, std::ptrdiff_t ldb,
    const TC *beta_,
    TC *c, std::ptrdiff_t ldc)
{
    const TC alpha = *alpha_, beta = *beta_;
    const char UPLO = up(&uplo);
    char TRANS = up(&trans);
    if (TRANS == 'C') TRANS = 'T';

    if (n == 0) return;

    if (ceq0(alpha) || k == 0) {
        if (ceq1(beta)) return;
        for (std::ptrdiff_t j = 0; j < n; ++j) wsyr2k_scale_col(j, n, UPLO, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = wsyr2k_block_nb();
    for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
        const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        wsyr2k_block(jc, jb, n, k, UPLO, TRANS, alpha, beta, a, lda, b, ldb, c, ldc);
    }
}

#undef A_
#undef B_
#undef C_
