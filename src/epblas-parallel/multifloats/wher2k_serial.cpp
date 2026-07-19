/*
 * wher2k_serial — multifloats complex (DD) Hermitian rank-2k update, pure
 * single-thread worker. Owns ALL the numerics; no OpenMP on this path.
 *
 *   C := alpha · A · Bᴴ + conj(alpha) · B · Aᴴ + beta · C  (TRANS='N')
 *   C := alpha · Aᴴ · B + conj(alpha) · Bᴴ · A + beta · C  (TRANS='C')
 *   alpha complex, beta real. The diagonal of C stays real.
 *
 * Blocked: AVX2 SIMD (or scalar) rank-2 diagonal kernel + two conjugate-
 * transpose wgemm trailing calls per off-diagonal wing. The trailing gemms
 * route through wgemm_serial (no nested OpenMP) so wher2k_parallel.cpp can call
 * the block worker from inside its own omp region.
 */
#include "wher2k_kernel.h"
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
using TR = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h */
namespace {


const TR rzero{0.0, 0.0};
const TR rone {1.0, 0.0};
const TC czero{ rzero, rzero };
const TC cone { rone,  rzero };


using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;
using mf_kernels::rcmul;

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

/* her2k triangle unpack: diagonal cells preserve original C[i,i].im. */
inline void unpack_4col_her2k_triangle(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
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
            if (i != j_abs) {
                col[i].im.limbs[0] = ih[ir * kSimdLane + j];
                col[i].im.limbs[1] = il[ir * kSimdLane + j];
            }
        }
    }
}

using simd_exact::vbcast;

/* TRANS='N': t1 = α · conj(B(j_panel..+4, l)),
 *         t2 = conj(α) · conj(A(j_panel..+4, l));
 * C[i, panel] += A(i,l)·t1 + B(i,l)·t2 over i ∈ diag block. */
inline void simd_her2k_diag_tn(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                               const TC *a, std::ptrdiff_t lda, const TC *b, std::ptrdiff_t ldb,
                               std::ptrdiff_t j_panel, std::ptrdiff_t j_count,
                               double *crh, double *crl,
                               double *cih, double *cil)
{
    __m256d a_rh, a_rl, a_ih, a_il;
    vbcast(alpha, a_rh, a_rl, a_ih, a_il);
    const __m256d zero_v = _mm256_setzero_pd();
    __m256d ac_ih = _mm256_sub_pd(zero_v, a_ih);    /* conj(α).im = -α.im */
    __m256d ac_il = _mm256_sub_pd(zero_v, a_il);
    alignas(32) double aj_rh[kSimdLane], aj_rl[kSimdLane], aj_ih[kSimdLane], aj_il[kSimdLane];
    alignas(32) double bj_rh[kSimdLane], bj_rl[kSimdLane], bj_ih[kSimdLane], bj_il[kSimdLane];
    for (std::ptrdiff_t ll = 0; ll < k; ++ll) {
        for (std::ptrdiff_t j = 0; j < j_count; ++j) {
            const TC av = A_(j_panel + j, ll);
            const TC bv = B_(j_panel + j, ll);
            /* Pre-conjugate during pack: store -im so cmul receives conj(A), conj(B) directly. */
            aj_rh[j] = av.re.limbs[0]; aj_rl[j] = av.re.limbs[1];
            aj_ih[j] = -av.im.limbs[0]; aj_il[j] = -av.im.limbs[1];
            bj_rh[j] = bv.re.limbs[0]; bj_rl[j] = bv.re.limbs[1];
            bj_ih[j] = -bv.im.limbs[0]; bj_il[j] = -bv.im.limbs[1];
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
        /* t1 = α · conj(B) */
        simd_fast::cmul(a_rh, a_rl, a_ih, a_il, bjrh, bjrl, bjih, bjil,
                         t1rh, t1rl, t1ih, t1il);
        /* t2 = conj(α) · conj(A) */
        simd_fast::cmul(a_rh, a_rl, ac_ih, ac_il, ajrh, ajrl, ajih, ajil,
                         t2rh, t2rl, t2ih, t2il);
        for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
            const std::ptrdiff_t ir = i - jc;
            __m256d aih, ail_, aiih, aiil;
            __m256d bih, bil_, biih, biil;
            vbcast(A_(i, ll), aih, ail_, aiih, aiil);
            vbcast(B_(i, ll), bih, bil_, biih, biil);
            __m256d p1rh, p1rl, p1ih, p1il, p2rh, p2rl, p2ih, p2il;
            simd_fast::cmul(aih, ail_, aiih, aiil, t1rh, t1rl, t1ih, t1il,
                             p1rh, p1rl, p1ih, p1il);
            simd_fast::cmul(bih, bil_, biih, biil, t2rh, t2rl, t2ih, t2il,
                             p2rh, p2rl, p2ih, p2il);
            __m256d srh, srl, sih, sil;
            simd_fast::cadd(p1rh, p1rl, p1ih, p1il, p2rh, p2rl, p2ih, p2il,
                             srh, srl, sih, sil);
            __m256d ckrh = _mm256_load_pd(&crh[ir * kSimdLane]);
            __m256d ckrl = _mm256_load_pd(&crl[ir * kSimdLane]);
            __m256d ckih = _mm256_load_pd(&cih[ir * kSimdLane]);
            __m256d ckil = _mm256_load_pd(&cil[ir * kSimdLane]);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(ckrh, ckrl, ckih, ckil, srh, srl, sih, sil,
                             nrh, nrl, nih, nil_);
            _mm256_store_pd(&crh[ir * kSimdLane], nrh);
            _mm256_store_pd(&crl[ir * kSimdLane], nrl);
            _mm256_store_pd(&cih[ir * kSimdLane], nih);
            _mm256_store_pd(&cil[ir * kSimdLane], nil_);
        }
    }
}

/* TRANS='C' SIMD, KC-tiled: accumulate s1 = Σ conj(Ai[l])·Bj_4 and
 * s2 = Σ conj(Bi[l])·Aj_4 over l ∈ [l0, l0+kc) into per-row 4-wide
 * accumulators acc1/acc2. The aj_/bj_ scratch hold this chunk's 4 packed A/B
 * columns at chunk-local rows 0..kc-1. acc1/acc2 are loaded/stored each call, so
 * accumulation continues across chunks in the same order as a single
 * l=0..K-1 loop → bit-identical to the untiled path. The α·s1 + conj(α)·s2
 * combine is applied once after all chunks. */
inline void simd_her2k_diag_tc_chunk(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t kc,
                                     const TC *a, std::ptrdiff_t lda, const TC *b, std::ptrdiff_t ldb,
                                     std::ptrdiff_t l0,
                                     const double *ajrh, const double *ajrl,
                                     const double *ajih, const double *ajil,
                                     const double *bjrh, const double *bjrl,
                                     const double *bjih, const double *bjil,
                                     double *acc1_rh, double *acc1_rl,
                                     double *acc1_ih, double *acc1_il,
                                     double *acc2_rh, double *acc2_rl,
                                     double *acc2_ih, double *acc2_il)
{
    for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
        const std::ptrdiff_t ir = i - jc;
        const TC *Ai = a + static_cast<std::size_t>(i) * lda;
        const TC *Bi = b + static_cast<std::size_t>(i) * ldb;
        __m256d s1rh = _mm256_load_pd(&acc1_rh[ir * kSimdLane]);
        __m256d s1rl = _mm256_load_pd(&acc1_rl[ir * kSimdLane]);
        __m256d s1ih = _mm256_load_pd(&acc1_ih[ir * kSimdLane]);
        __m256d s1il = _mm256_load_pd(&acc1_il[ir * kSimdLane]);
        __m256d s2rh = _mm256_load_pd(&acc2_rh[ir * kSimdLane]);
        __m256d s2rl = _mm256_load_pd(&acc2_rl[ir * kSimdLane]);
        __m256d s2ih = _mm256_load_pd(&acc2_ih[ir * kSimdLane]);
        __m256d s2il = _mm256_load_pd(&acc2_il[ir * kSimdLane]);
        for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
            /* Broadcast conj(Ai[l0+ll]) and conj(Bi[l0+ll]) */
            __m256d aih = _mm256_set1_pd(Ai[l0 + ll].re.limbs[0]);
            __m256d ail_ = _mm256_set1_pd(Ai[l0 + ll].re.limbs[1]);
            __m256d aiih = _mm256_set1_pd(-Ai[l0 + ll].im.limbs[0]);
            __m256d aiil = _mm256_set1_pd(-Ai[l0 + ll].im.limbs[1]);
            __m256d bih = _mm256_set1_pd(Bi[l0 + ll].re.limbs[0]);
            __m256d bil_ = _mm256_set1_pd(Bi[l0 + ll].re.limbs[1]);
            __m256d biih = _mm256_set1_pd(-Bi[l0 + ll].im.limbs[0]);
            __m256d biil = _mm256_set1_pd(-Bi[l0 + ll].im.limbs[1]);
            __m256d ajrv = _mm256_load_pd(&ajrh[ll * kSimdLane]);
            __m256d ajrlv = _mm256_load_pd(&ajrl[ll * kSimdLane]);
            __m256d ajiv = _mm256_load_pd(&ajih[ll * kSimdLane]);
            __m256d ajilv = _mm256_load_pd(&ajil[ll * kSimdLane]);
            __m256d bjrv = _mm256_load_pd(&bjrh[ll * kSimdLane]);
            __m256d bjrlv = _mm256_load_pd(&bjrl[ll * kSimdLane]);
            __m256d bjiv = _mm256_load_pd(&bjih[ll * kSimdLane]);
            __m256d bjilv = _mm256_load_pd(&bjil[ll * kSimdLane]);
            /* p1 = conj(Ai[l]) · Bj_4 */
            __m256d p1rh, p1rl, p1ih, p1il;
            simd_fast::cmul(aih, ail_, aiih, aiil, bjrv, bjrlv, bjiv, bjilv,
                             p1rh, p1rl, p1ih, p1il);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(s1rh, s1rl, s1ih, s1il, p1rh, p1rl, p1ih, p1il,
                             nrh, nrl, nih, nil_);
            s1rh = nrh; s1rl = nrl; s1ih = nih; s1il = nil_;
            /* p2 = conj(Bi[l]) · Aj_4 */
            __m256d p2rh, p2rl, p2ih, p2il;
            simd_fast::cmul(bih, bil_, biih, biil, ajrv, ajrlv, ajiv, ajilv,
                             p2rh, p2rl, p2ih, p2il);
            simd_fast::cadd(s2rh, s2rl, s2ih, s2il, p2rh, p2rl, p2ih, p2il,
                             nrh, nrl, nih, nil_);
            s2rh = nrh; s2rl = nrl; s2ih = nih; s2il = nil_;
        }
        _mm256_store_pd(&acc1_rh[ir * kSimdLane], s1rh);
        _mm256_store_pd(&acc1_rl[ir * kSimdLane], s1rl);
        _mm256_store_pd(&acc1_ih[ir * kSimdLane], s1ih);
        _mm256_store_pd(&acc1_il[ir * kSimdLane], s1il);
        _mm256_store_pd(&acc2_rh[ir * kSimdLane], s2rh);
        _mm256_store_pd(&acc2_rl[ir * kSimdLane], s2rl);
        _mm256_store_pd(&acc2_ih[ir * kSimdLane], s2ih);
        _mm256_store_pd(&acc2_il[ir * kSimdLane], s2il);
    }
}

inline void simd_her2k_diag_panels(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                                   const TC *a, std::ptrdiff_t lda, const TC *b, std::ptrdiff_t ldb,
                                   TC *c, std::ptrdiff_t ldc, char UPLO, char TRANS)
{
    alignas(32) double crh[kMaxBlockM * kSimdLane], crl[kMaxBlockM * kSimdLane];
    alignas(32) double cih[kMaxBlockM * kSimdLane], cil[kMaxBlockM * kSimdLane];
    /* TRANS='C' scratch: one K-chunk of 4 packed A/B columns (bounded by kMaxK)
     * plus two per-row complex-DD accumulators (s1, s2) carried across chunks. */
    alignas(32) static thread_local double ajrh[kMaxK * kSimdLane], ajrl[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajih[kMaxK * kSimdLane], ajil[kMaxK * kSimdLane];
    alignas(32) static thread_local double bjrh[kMaxK * kSimdLane], bjrl[kMaxK * kSimdLane];
    alignas(32) static thread_local double bjih[kMaxK * kSimdLane], bjil[kMaxK * kSimdLane];
    alignas(32) double acc1_rh[kMaxBlockM * kSimdLane], acc1_rl[kMaxBlockM * kSimdLane];
    alignas(32) double acc1_ih[kMaxBlockM * kSimdLane], acc1_il[kMaxBlockM * kSimdLane];
    alignas(32) double acc2_rh[kMaxBlockM * kSimdLane], acc2_rl[kMaxBlockM * kSimdLane];
    alignas(32) double acc2_ih[kMaxBlockM * kSimdLane], acc2_il[kMaxBlockM * kSimdLane];

    for (std::ptrdiff_t j = jc; j < jc + jb; j += kSimdLane) {
        const std::ptrdiff_t jcount = (jc + jb - j < kSimdLane) ? (jc + jb - j) : kSimdLane;
        pack_4col_cdd(jb, jc, c, ldc, j, jcount, crh, crl, cih, cil);
        if (TRANS == 'N') {
            /* TRANS='N' reads A/B directly per l — K-independent, no scratch cap. */
            simd_her2k_diag_tn(jc, jb, k, alpha, a, lda, b, ldb, j, jcount,
                               crh, crl, cih, cil);
        } else {
            const __m256d zv = _mm256_setzero_pd();
            for (std::ptrdiff_t ir = 0; ir < jb; ++ir) {
                _mm256_store_pd(&acc1_rh[ir * kSimdLane], zv);
                _mm256_store_pd(&acc1_rl[ir * kSimdLane], zv);
                _mm256_store_pd(&acc1_ih[ir * kSimdLane], zv);
                _mm256_store_pd(&acc1_il[ir * kSimdLane], zv);
                _mm256_store_pd(&acc2_rh[ir * kSimdLane], zv);
                _mm256_store_pd(&acc2_rl[ir * kSimdLane], zv);
                _mm256_store_pd(&acc2_ih[ir * kSimdLane], zv);
                _mm256_store_pd(&acc2_il[ir * kSimdLane], zv);
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
                simd_her2k_diag_tc_chunk(jc, jb, kc, a, lda, b, ldb, l0,
                                         ajrh, ajrl, ajih, ajil,
                                         bjrh, bjrl, bjih, bjil,
                                         acc1_rh, acc1_rl, acc1_ih, acc1_il,
                                         acc2_rh, acc2_rl, acc2_ih, acc2_il);
            }
            /* Finalize: C[panel] += α·s1 + conj(α)·s2 (single combine, as untiled). */
            __m256d a_rh, a_rl, a_ih, a_il;
            vbcast(alpha, a_rh, a_rl, a_ih, a_il);
            const __m256d zero_v = _mm256_setzero_pd();
            __m256d ac_ih = _mm256_sub_pd(zero_v, a_ih);
            __m256d ac_il = _mm256_sub_pd(zero_v, a_il);
            for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
                const std::ptrdiff_t ir = i - jc;
                __m256d s1rh = _mm256_load_pd(&acc1_rh[ir * kSimdLane]);
                __m256d s1rl = _mm256_load_pd(&acc1_rl[ir * kSimdLane]);
                __m256d s1ih = _mm256_load_pd(&acc1_ih[ir * kSimdLane]);
                __m256d s1il = _mm256_load_pd(&acc1_il[ir * kSimdLane]);
                __m256d s2rh = _mm256_load_pd(&acc2_rh[ir * kSimdLane]);
                __m256d s2rl = _mm256_load_pd(&acc2_rl[ir * kSimdLane]);
                __m256d s2ih = _mm256_load_pd(&acc2_ih[ir * kSimdLane]);
                __m256d s2il = _mm256_load_pd(&acc2_il[ir * kSimdLane]);
                __m256d a1rh, a1rl, a1ih, a1il, a2rh, a2rl, a2ih, a2il;
                simd_fast::cmul(a_rh, a_rl, a_ih, a_il, s1rh, s1rl, s1ih, s1il,
                                 a1rh, a1rl, a1ih, a1il);
                simd_fast::cmul(a_rh, a_rl, ac_ih, ac_il, s2rh, s2rl, s2ih, s2il,
                                 a2rh, a2rl, a2ih, a2il);
                __m256d sumrh, sumrl, sumih, sumil;
                simd_fast::cadd(a1rh, a1rl, a1ih, a1il, a2rh, a2rl, a2ih, a2il,
                                 sumrh, sumrl, sumih, sumil);
                __m256d ckrh = _mm256_load_pd(&crh[ir * kSimdLane]);
                __m256d ckrl = _mm256_load_pd(&crl[ir * kSimdLane]);
                __m256d ckih = _mm256_load_pd(&cih[ir * kSimdLane]);
                __m256d ckil = _mm256_load_pd(&cil[ir * kSimdLane]);
                __m256d nrh, nrl, nih, nil_;
                simd_fast::cadd(ckrh, ckrl, ckih, ckil, sumrh, sumrl, sumih, sumil,
                                 nrh, nrl, nih, nil_);
                _mm256_store_pd(&crh[ir * kSimdLane], nrh);
                _mm256_store_pd(&crl[ir * kSimdLane], nrl);
                _mm256_store_pd(&cih[ir * kSimdLane], nih);
                _mm256_store_pd(&cil[ir * kSimdLane], nil_);
            }
        }
        unpack_4col_her2k_triangle(jc, jb, j, jcount, UPLO, c, ldc,
                                   crh, crl, cih, cil);
    }
}

#pragma GCC pop_options

#endif  /* MBLAS_SIMD_DD */

void her2k_diag_add(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t k, TC alpha,
                    const TC *a, std::ptrdiff_t lda,
                    const TC *b, std::ptrdiff_t ldb,
                    TC *c, std::ptrdiff_t ldc,
                    char UPLO, char TRANS)
{
    const TC alpha_conj = cconj(alpha);
    if (TRANS == 'N') {
        /* C(I,J) += α A(I,l) conj(B(J,l)) + conj(α) B(I,l) conj(A(J,l)) */
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            const std::ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            TC *cj = c + static_cast<std::size_t>(j) * ldc;
            for (std::ptrdiff_t l = 0; l < k; ++l) {
                const TC t1 = cmul(alpha,       cconj(B_(j, l)));
                const TC t2 = cmul(alpha_conj,  cconj(A_(j, l)));
                for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                    const TC prod = cadd(cmul(A_(i, l), t1), cmul(B_(i, l), t2));
                    if (i == j) cj[i] = TC{ cj[i].re + prod.re, cj[i].im };
                    else        cj[i] = cadd(cj[i], prod);
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
                TC s1 = czero, s2 = czero;
                for (std::ptrdiff_t l = 0; l < k; ++l) {
                    s1 = cadd(s1, cmul(cconj(Ai[l]), Bj[l]));
                    s2 = cadd(s2, cmul(cconj(Bi[l]), Aj[l]));
                }
                const TC as = cadd(cmul(alpha, s1), cmul(alpha_conj, s2));
                if (i == j) cj[i] = TC{ cj[i].re + as.re, cj[i].im };
                else        cj[i] = cadd(cj[i], as);
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
        simd_her2k_diag_panels(jc, jb, k, alpha, a, lda, b, ldb, c, ldc, UPLO, TRANS);
        return;
    }
#endif
    her2k_diag_add(jc, jb, k, alpha, a, lda, b, ldb, c, ldc, UPLO, TRANS);
}

} /* anonymous namespace */

std::ptrdiff_t wher2k_block_nb(void) {
    static std::ptrdiff_t nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void wher2k_zero_diag_im(std::ptrdiff_t j, TC *c, std::ptrdiff_t ldc) {
    c[static_cast<std::size_t>(j) * ldc + j].im = rzero;
}

void wher2k_scale_col(std::ptrdiff_t j, std::ptrdiff_t n, char UPLO, TR beta, TC *c, std::ptrdiff_t ldc) {
    const std::ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
    const std::ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
    TC *cj = c + static_cast<std::size_t>(j) * ldc;
    if (eq0(beta)) for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = czero;
    else {
        for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
            if (i == j) cj[i] = TC{ beta * cj[i].re, rzero };
            else        cj[i] = rcmul(beta, cj[i]);
        }
    }
}

void wher2k_block(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t n, std::ptrdiff_t k, char UPLO, char TRANS,
                  TC alpha, TR beta, const TC *a, std::ptrdiff_t lda, const TC *b, std::ptrdiff_t ldb,
                  TC *c, std::ptrdiff_t ldc)
{
    /* Beta-scale this block's own triangle columns (real-diag preservation). */
    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        const std::ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const std::ptrdiff_t i_hi = (UPLO == 'L') ? n : j + 1;
        TC *cj = c + static_cast<std::size_t>(j) * ldc;
        if (eq0(beta)) {
            for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = czero;
        } else if (!eq1(beta)) {
            for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                if (i == j) cj[i] = TC{ beta * cj[i].re, rzero };
                else        cj[i] = rcmul(beta, cj[i]);
            }
        } else {
            cj[j].im = rzero;
        }
    }

    diag_dispatch(jc, jb, k, alpha, a, lda, b, ldb, c, ldc, UPLO, TRANS);

    const char NN[1] = {'N'};
    const char CN[1] = {'C'};
    const TC alpha_conj = cconj(alpha);

    if (UPLO == 'L') {
        const std::ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            const std::ptrdiff_t j0 = jc + jb;
            if (TRANS == 'N') {
                wgemm_serial(NN[0], CN[0], trailing, jb, k, &alpha, &A_(j0, 0), lda, &B_(jc, 0), ldb, &cone, &C_(j0, jc), ldc);
                wgemm_serial(NN[0], CN[0], trailing, jb, k, &alpha_conj, &B_(j0, 0), ldb, &A_(jc, 0), lda, &cone, &C_(j0, jc), ldc);
            } else {
                wgemm_serial(CN[0], NN[0], trailing, jb, k, &alpha, &A_(0, j0), lda, &B_(0, jc), ldb, &cone, &C_(j0, jc), ldc);
                wgemm_serial(CN[0], NN[0], trailing, jb, k, &alpha_conj, &B_(0, j0), ldb, &A_(0, jc), lda, &cone, &C_(j0, jc), ldc);
            }
        }
    } else {
        if (jc > 0) {
            if (TRANS == 'N') {
                wgemm_serial(NN[0], CN[0], jc, jb, k, &alpha, &A_(0, 0), lda, &B_(jc, 0), ldb, &cone, &C_(0, jc), ldc);
                wgemm_serial(NN[0], CN[0], jc, jb, k, &alpha_conj, &B_(0, 0), ldb, &A_(jc, 0), lda, &cone, &C_(0, jc), ldc);
            } else {
                wgemm_serial(CN[0], NN[0], jc, jb, k, &alpha, &A_(0, 0), lda, &B_(0, jc), ldb, &cone, &C_(0, jc), ldc);
                wgemm_serial(CN[0], NN[0], jc, jb, k, &alpha_conj, &B_(0, 0), ldb, &A_(0, jc), lda, &cone, &C_(0, jc), ldc);
            }
        }
    }
}

extern "C" void wher2k_serial(
    char uplo, char trans,
    std::ptrdiff_t n, std::ptrdiff_t k,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    const TC *b, std::ptrdiff_t ldb,
    const TR *beta_,
    TC *c, std::ptrdiff_t ldc)
{
    const TC alpha = *alpha_;
    const TR beta  = *beta_;
    const char UPLO = up(&uplo);
    const char TRANS = up(&trans);

    if (n == 0) return;

    if ((eq0(alpha.re) && eq0(alpha.im)) || k == 0) {
        if (eq1(beta)) {
            for (std::ptrdiff_t j = 0; j < n; ++j) wher2k_zero_diag_im(j, c, ldc);
            return;
        }
        for (std::ptrdiff_t j = 0; j < n; ++j) wher2k_scale_col(j, n, UPLO, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = wher2k_block_nb();
    for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
        const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
        wher2k_block(jc, jb, n, k, UPLO, TRANS, alpha, beta, a, lda, b, ldb, c, ldc);
    }
}

#undef A_
#undef B_
#undef C_
