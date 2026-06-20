/*
 * wher2k_serial — multifloats complex (DD) Hermitian rank-2k update, pure
 * single-thread worker. Owns ALL the numerics; no OpenMP on this path.
 *
 *   C := alpha · A · Bᴴ + conj(alpha) · B · Aᴴ + beta · C  (TR='N')
 *   C := alpha · Aᴴ · B + conj(alpha) · Bᴴ · A + beta · C  (TR='C')
 *   alpha complex, beta real. The diagonal of C stays real.
 *
 * Blocked: AVX2 SIMD (or scalar) rank-2 diagonal kernel + two conjugate-
 * transpose wgemm trailing calls per off-diagonal wing. The trailing gemms
 * route through wgemm_serial (no nested OpenMP) so wher2k_parallel.cpp can call
 * the block worker from inside its own omp region.
 */
#include "wher2k_kernel.h"
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
const R rone {1.0, 0.0};
const T czero{ rzero, rzero };
const T cone { rone,  rzero };

inline bool dd_iszero(R x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (R x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }

inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }
inline T cconj(T const &a) { return T{ a.re, R{-a.im.limbs[0], -a.im.limbs[1]} }; }
inline T rcmul(R const &r, T const &z) { return T{ r * z.re, r * z.im }; }

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]
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

/* her2k triangle unpack: diagonal cells preserve original C[i,i].im. */
inline void unpack_4col_her2k_triangle(int jc, int jb, int j_start, int j_count,
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
            if (i != j_abs) {
                col[i].im.limbs[0] = ih[ir * kSimdLane + j];
                col[i].im.limbs[1] = il[ir * kSimdLane + j];
            }
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

/* TR='N': t1 = α · conj(B(j_panel..+4, l)),
 *         t2 = conj(α) · conj(A(j_panel..+4, l));
 * C[i, panel] += A(i,l)·t1 + B(i,l)·t2 over i ∈ diag block. */
inline void simd_her2k_diag_tn(int jc, int jb, int K, T alpha,
                               const T *a, int lda, const T *b, int ldb,
                               int j_panel, int j_count,
                               double *crh, double *crl,
                               double *cih, double *cil)
{
    __m256d a_rh, a_rl, a_ih, a_il;
    broadcast_cdd(alpha, a_rh, a_rl, a_ih, a_il);
    const __m256d zero_v = _mm256_setzero_pd();
    __m256d ac_ih = _mm256_sub_pd(zero_v, a_ih);    /* conj(α).im = -α.im */
    __m256d ac_il = _mm256_sub_pd(zero_v, a_il);
    alignas(32) double aj_rh[kSimdLane], aj_rl[kSimdLane], aj_ih[kSimdLane], aj_il[kSimdLane];
    alignas(32) double bj_rh[kSimdLane], bj_rl[kSimdLane], bj_ih[kSimdLane], bj_il[kSimdLane];
    for (int ll = 0; ll < K; ++ll) {
        for (int j = 0; j < j_count; ++j) {
            const T av = A_(j_panel + j, ll);
            const T bv = B_(j_panel + j, ll);
            /* Pre-conjugate during pack: store -im so cdd_mul receives conj(A), conj(B) directly. */
            aj_rh[j] = av.re.limbs[0]; aj_rl[j] = av.re.limbs[1];
            aj_ih[j] = -av.im.limbs[0]; aj_il[j] = -av.im.limbs[1];
            bj_rh[j] = bv.re.limbs[0]; bj_rl[j] = bv.re.limbs[1];
            bj_ih[j] = -bv.im.limbs[0]; bj_il[j] = -bv.im.limbs[1];
        }
        for (int j = j_count; j < kSimdLane; ++j) {
            aj_rh[j] = 0.0; aj_rl[j] = 0.0; aj_ih[j] = 0.0; aj_il[j] = 0.0;
            bj_rh[j] = 0.0; bj_rl[j] = 0.0; bj_ih[j] = 0.0; bj_il[j] = 0.0;
        }
        __m256d ajrh = _mm256_load_pd(aj_rh), ajrl = _mm256_load_pd(aj_rl);
        __m256d ajih = _mm256_load_pd(aj_ih), ajil = _mm256_load_pd(aj_il);
        __m256d bjrh = _mm256_load_pd(bj_rh), bjrl = _mm256_load_pd(bj_rl);
        __m256d bjih = _mm256_load_pd(bj_ih), bjil = _mm256_load_pd(bj_il);
        __m256d t1rh, t1rl, t1ih, t1il, t2rh, t2rl, t2ih, t2il;
        /* t1 = α · conj(B) */
        simd_dd::cdd_mul(a_rh, a_rl, a_ih, a_il, bjrh, bjrl, bjih, bjil,
                         t1rh, t1rl, t1ih, t1il);
        /* t2 = conj(α) · conj(A) */
        simd_dd::cdd_mul(a_rh, a_rl, ac_ih, ac_il, ajrh, ajrl, ajih, ajil,
                         t2rh, t2rl, t2ih, t2il);
        for (int i = jc; i < jc + jb; ++i) {
            const int ir = i - jc;
            __m256d aih, ail_, aiih, aiil;
            __m256d bih, bil_, biih, biil;
            broadcast_cdd(A_(i, ll), aih, ail_, aiih, aiil);
            broadcast_cdd(B_(i, ll), bih, bil_, biih, biil);
            __m256d p1rh, p1rl, p1ih, p1il, p2rh, p2rl, p2ih, p2il;
            simd_dd::cdd_mul(aih, ail_, aiih, aiil, t1rh, t1rl, t1ih, t1il,
                             p1rh, p1rl, p1ih, p1il);
            simd_dd::cdd_mul(bih, bil_, biih, biil, t2rh, t2rl, t2ih, t2il,
                             p2rh, p2rl, p2ih, p2il);
            __m256d srh, srl, sih, sil;
            simd_dd::cdd_add(p1rh, p1rl, p1ih, p1il, p2rh, p2rl, p2ih, p2il,
                             srh, srl, sih, sil);
            __m256d ckrh = _mm256_load_pd(&crh[ir * kSimdLane]);
            __m256d ckrl = _mm256_load_pd(&crl[ir * kSimdLane]);
            __m256d ckih = _mm256_load_pd(&cih[ir * kSimdLane]);
            __m256d ckil = _mm256_load_pd(&cil[ir * kSimdLane]);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(ckrh, ckrl, ckih, ckil, srh, srl, sih, sil,
                             nrh, nrl, nih, nil_);
            _mm256_store_pd(&crh[ir * kSimdLane], nrh);
            _mm256_store_pd(&crl[ir * kSimdLane], nrl);
            _mm256_store_pd(&cih[ir * kSimdLane], nih);
            _mm256_store_pd(&cil[ir * kSimdLane], nil_);
        }
    }
}

/* TR='C' SIMD, KC-tiled: accumulate s1 = Σ conj(Ai[l])·Bj_4 and
 * s2 = Σ conj(Bi[l])·Aj_4 over l ∈ [l0, l0+kc) into per-row 4-wide
 * accumulators acc1/acc2. The aj_/bj_ scratch hold this chunk's 4 packed A/B
 * columns at chunk-local rows 0..kc-1. acc1/acc2 are loaded/stored each call, so
 * accumulation continues across chunks in the same order as a single
 * l=0..K-1 loop → bit-identical to the untiled path. The α·s1 + conj(α)·s2
 * combine is applied once after all chunks. */
inline void simd_her2k_diag_tc_chunk(int jc, int jb, int kc,
                                     const T *a, int lda, const T *b, int ldb,
                                     int l0,
                                     const double *ajrh, const double *ajrl,
                                     const double *ajih, const double *ajil,
                                     const double *bjrh, const double *bjrl,
                                     const double *bjih, const double *bjil,
                                     double *acc1_rh, double *acc1_rl,
                                     double *acc1_ih, double *acc1_il,
                                     double *acc2_rh, double *acc2_rl,
                                     double *acc2_ih, double *acc2_il)
{
    for (int i = jc; i < jc + jb; ++i) {
        const int ir = i - jc;
        const T *Ai = a + static_cast<std::size_t>(i) * lda;
        const T *Bi = b + static_cast<std::size_t>(i) * ldb;
        __m256d s1rh = _mm256_load_pd(&acc1_rh[ir * kSimdLane]);
        __m256d s1rl = _mm256_load_pd(&acc1_rl[ir * kSimdLane]);
        __m256d s1ih = _mm256_load_pd(&acc1_ih[ir * kSimdLane]);
        __m256d s1il = _mm256_load_pd(&acc1_il[ir * kSimdLane]);
        __m256d s2rh = _mm256_load_pd(&acc2_rh[ir * kSimdLane]);
        __m256d s2rl = _mm256_load_pd(&acc2_rl[ir * kSimdLane]);
        __m256d s2ih = _mm256_load_pd(&acc2_ih[ir * kSimdLane]);
        __m256d s2il = _mm256_load_pd(&acc2_il[ir * kSimdLane]);
        for (int ll = 0; ll < kc; ++ll) {
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
            simd_dd::cdd_mul(aih, ail_, aiih, aiil, bjrv, bjrlv, bjiv, bjilv,
                             p1rh, p1rl, p1ih, p1il);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(s1rh, s1rl, s1ih, s1il, p1rh, p1rl, p1ih, p1il,
                             nrh, nrl, nih, nil_);
            s1rh = nrh; s1rl = nrl; s1ih = nih; s1il = nil_;
            /* p2 = conj(Bi[l]) · Aj_4 */
            __m256d p2rh, p2rl, p2ih, p2il;
            simd_dd::cdd_mul(bih, bil_, biih, biil, ajrv, ajrlv, ajiv, ajilv,
                             p2rh, p2rl, p2ih, p2il);
            simd_dd::cdd_add(s2rh, s2rl, s2ih, s2il, p2rh, p2rl, p2ih, p2il,
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

inline void simd_her2k_diag_panels(int jc, int jb, int K, T alpha,
                                   const T *a, int lda, const T *b, int ldb,
                                   T *c, int ldc, char UPLO, char TR_c)
{
    alignas(32) double crh[kMaxBlockM * kSimdLane], crl[kMaxBlockM * kSimdLane];
    alignas(32) double cih[kMaxBlockM * kSimdLane], cil[kMaxBlockM * kSimdLane];
    /* TR='C' scratch: one K-chunk of 4 packed A/B columns (bounded by kMaxK)
     * plus two per-row complex-DD accumulators (s1, s2) carried across chunks. */
    alignas(32) static thread_local double ajrh[kMaxK * kSimdLane], ajrl[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajih[kMaxK * kSimdLane], ajil[kMaxK * kSimdLane];
    alignas(32) static thread_local double bjrh[kMaxK * kSimdLane], bjrl[kMaxK * kSimdLane];
    alignas(32) static thread_local double bjih[kMaxK * kSimdLane], bjil[kMaxK * kSimdLane];
    alignas(32) double acc1_rh[kMaxBlockM * kSimdLane], acc1_rl[kMaxBlockM * kSimdLane];
    alignas(32) double acc1_ih[kMaxBlockM * kSimdLane], acc1_il[kMaxBlockM * kSimdLane];
    alignas(32) double acc2_rh[kMaxBlockM * kSimdLane], acc2_rl[kMaxBlockM * kSimdLane];
    alignas(32) double acc2_ih[kMaxBlockM * kSimdLane], acc2_il[kMaxBlockM * kSimdLane];

    for (int j = jc; j < jc + jb; j += kSimdLane) {
        const int jcount = (jc + jb - j < kSimdLane) ? (jc + jb - j) : kSimdLane;
        pack_4col_cdd(jb, jc, c, ldc, j, jcount, crh, crl, cih, cil);
        if (TR_c == 'N') {
            /* TR='N' reads A/B directly per l — K-independent, no scratch cap. */
            simd_her2k_diag_tn(jc, jb, K, alpha, a, lda, b, ldb, j, jcount,
                               crh, crl, cih, cil);
        } else {
            const __m256d zv = _mm256_setzero_pd();
            for (int ir = 0; ir < jb; ++ir) {
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
            for (int l0 = 0; l0 < K; l0 += kMaxK) {
                const int kc = (K - l0 < kMaxK) ? (K - l0) : kMaxK;
                for (int jj = 0; jj < jcount; ++jj) {
                    const T *acol = a + static_cast<std::size_t>(j + jj) * lda;
                    const T *bcol = b + static_cast<std::size_t>(j + jj) * ldb;
                    for (int ll = 0; ll < kc; ++ll) {
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
                for (int jj = jcount; jj < kSimdLane; ++jj)
                    for (int ll = 0; ll < kc; ++ll) {
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
            broadcast_cdd(alpha, a_rh, a_rl, a_ih, a_il);
            const __m256d zero_v = _mm256_setzero_pd();
            __m256d ac_ih = _mm256_sub_pd(zero_v, a_ih);
            __m256d ac_il = _mm256_sub_pd(zero_v, a_il);
            for (int i = jc; i < jc + jb; ++i) {
                const int ir = i - jc;
                __m256d s1rh = _mm256_load_pd(&acc1_rh[ir * kSimdLane]);
                __m256d s1rl = _mm256_load_pd(&acc1_rl[ir * kSimdLane]);
                __m256d s1ih = _mm256_load_pd(&acc1_ih[ir * kSimdLane]);
                __m256d s1il = _mm256_load_pd(&acc1_il[ir * kSimdLane]);
                __m256d s2rh = _mm256_load_pd(&acc2_rh[ir * kSimdLane]);
                __m256d s2rl = _mm256_load_pd(&acc2_rl[ir * kSimdLane]);
                __m256d s2ih = _mm256_load_pd(&acc2_ih[ir * kSimdLane]);
                __m256d s2il = _mm256_load_pd(&acc2_il[ir * kSimdLane]);
                __m256d a1rh, a1rl, a1ih, a1il, a2rh, a2rl, a2ih, a2il;
                simd_dd::cdd_mul(a_rh, a_rl, a_ih, a_il, s1rh, s1rl, s1ih, s1il,
                                 a1rh, a1rl, a1ih, a1il);
                simd_dd::cdd_mul(a_rh, a_rl, ac_ih, ac_il, s2rh, s2rl, s2ih, s2il,
                                 a2rh, a2rl, a2ih, a2il);
                __m256d sumrh, sumrl, sumih, sumil;
                simd_dd::cdd_add(a1rh, a1rl, a1ih, a1il, a2rh, a2rl, a2ih, a2il,
                                 sumrh, sumrl, sumih, sumil);
                __m256d ckrh = _mm256_load_pd(&crh[ir * kSimdLane]);
                __m256d ckrl = _mm256_load_pd(&crl[ir * kSimdLane]);
                __m256d ckih = _mm256_load_pd(&cih[ir * kSimdLane]);
                __m256d ckil = _mm256_load_pd(&cil[ir * kSimdLane]);
                __m256d nrh, nrl, nih, nil_;
                simd_dd::cdd_add(ckrh, ckrl, ckih, ckil, sumrh, sumrl, sumih, sumil,
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

#endif  /* MBLAS_SIMD_DD */

void her2k_diag_add(int jc, int jb, int K, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T *c, int ldc,
                    char UPLO, char TR_c)
{
    const T alpha_conj = cconj(alpha);
    if (TR_c == 'N') {
        /* C(I,J) += α A(I,l) conj(B(J,l)) + conj(α) B(I,l) conj(A(J,l)) */
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            for (int l = 0; l < K; ++l) {
                const T t1 = cmul(alpha,       cconj(B_(j, l)));
                const T t2 = cmul(alpha_conj,  cconj(A_(j, l)));
                for (int i = i_lo; i < i_hi; ++i) {
                    const T prod = cadd(cmul(A_(i, l), t1), cmul(B_(i, l), t2));
                    if (i == j) cj[i] = T{ cj[i].re + prod.re, cj[i].im };
                    else        cj[i] = cadd(cj[i], prod);
                }
            }
        }
    } else {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            const T *Aj = a + static_cast<std::size_t>(j) * lda;
            const T *Bj = b + static_cast<std::size_t>(j) * ldb;
            for (int i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + static_cast<std::size_t>(i) * lda;
                const T *Bi = b + static_cast<std::size_t>(i) * ldb;
                T s1 = czero, s2 = czero;
                for (int l = 0; l < K; ++l) {
                    s1 = cadd(s1, cmul(cconj(Ai[l]), Bj[l]));
                    s2 = cadd(s2, cmul(cconj(Bi[l]), Aj[l]));
                }
                const T as = cadd(cmul(alpha, s1), cmul(alpha_conj, s2));
                if (i == j) cj[i] = T{ cj[i].re + as.re, cj[i].im };
                else        cj[i] = cadd(cj[i], as);
            }
        }
    }
}

inline void diag_dispatch(int jc, int jb, int K, T alpha,
                          const T *a, int lda, const T *b, int ldb,
                          T *c, int ldc, char UPLO, char TR_c)
{
#ifdef MBLAS_SIMD_DD
    if (jb <= kMaxBlockM) {
        simd_her2k_diag_panels(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR_c);
        return;
    }
#endif
    her2k_diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR_c);
}

} /* anonymous namespace */

int wher2k_block_nb(void) {
    static int nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void wher2k_zero_diag_im(int j, T *c, int ldc) {
    c[static_cast<std::size_t>(j) * ldc + j].im = rzero;
}

void wher2k_scale_col(int j, int N, char UPLO, R beta, T *c, int ldc) {
    const int i_lo = (UPLO == 'L') ? j : 0;
    const int i_hi = (UPLO == 'L') ? N : j + 1;
    T *cj = c + static_cast<std::size_t>(j) * ldc;
    if (dd_iszero(beta)) for (int i = i_lo; i < i_hi; ++i) cj[i] = czero;
    else {
        for (int i = i_lo; i < i_hi; ++i) {
            if (i == j) cj[i] = T{ beta * cj[i].re, rzero };
            else        cj[i] = rcmul(beta, cj[i]);
        }
    }
}

void wher2k_block(int jc, int jb, int N, int K, char UPLO, char TR,
                  T alpha, R beta, const T *a, int lda, const T *b, int ldb,
                  T *c, int ldc)
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

    diag_dispatch(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR);

    const char NN[1] = {'N'};
    const char CN[1] = {'C'};
    const T alpha_conj = cconj(alpha);

    if (UPLO == 'L') {
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            const int j0 = jc + jb;
            if (TR == 'N') {
                wgemm_serial(NN, CN, &trailing, &jb, &K, &alpha,
                             &A_(j0, 0), &lda, &B_(jc, 0), &ldb,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
                wgemm_serial(NN, CN, &trailing, &jb, &K, &alpha_conj,
                             &B_(j0, 0), &ldb, &A_(jc, 0), &lda,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
            } else {
                wgemm_serial(CN, NN, &trailing, &jb, &K, &alpha,
                             &A_(0, j0), &lda, &B_(0, jc), &ldb,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
                wgemm_serial(CN, NN, &trailing, &jb, &K, &alpha_conj,
                             &B_(0, j0), &ldb, &A_(0, jc), &lda,
                             &cone, &C_(j0, jc), &ldc, 1, 1);
            }
        }
    } else {
        if (jc > 0) {
            if (TR == 'N') {
                wgemm_serial(NN, CN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &B_(jc, 0), &ldb,
                             &cone, &C_(0, jc), &ldc, 1, 1);
                wgemm_serial(NN, CN, &jc, &jb, &K, &alpha_conj,
                             &B_(0, 0), &ldb, &A_(jc, 0), &lda,
                             &cone, &C_(0, jc), &ldc, 1, 1);
            } else {
                wgemm_serial(CN, NN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &B_(0, jc), &ldb,
                             &cone, &C_(0, jc), &ldc, 1, 1);
                wgemm_serial(CN, NN, &jc, &jb, &K, &alpha_conj,
                             &B_(0, 0), &ldb, &A_(0, jc), &lda,
                             &cone, &C_(0, jc), &ldc, 1, 1);
            }
        }
    }
}

extern "C" void wher2k_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const R *beta_,
    T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_;
    const R beta  = *beta_;
    const char UPLO = up(uplo);
    const char TR_c = up(trans);

    if (N == 0) return;

    if ((dd_iszero(alpha.re) && dd_iszero(alpha.im)) || K == 0) {
        if (dd_isone(beta)) {
            for (int j = 0; j < N; ++j) wher2k_zero_diag_im(j, c, ldc);
            return;
        }
        for (int j = 0; j < N; ++j) wher2k_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const int nb = wher2k_block_nb();
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        wher2k_block(jc, jb, N, K, UPLO, TR_c, alpha, beta, a, lda, b, ldb, c, ldc);
    }
}

#undef A_
#undef B_
#undef C_
