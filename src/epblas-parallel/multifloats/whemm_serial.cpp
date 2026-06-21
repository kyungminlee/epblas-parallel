/*
 * whemm_serial — multifloats complex (DD) Hermitian matrix multiply, pure
 * single-thread worker. NOT symmetric (see wsymm). Owns ALL the numerics; no
 * OpenMP on this path.
 *
 * Same blocked SIMD strategy as wsymm: AVX2 4-wide pack of 4 columns of B and
 * C into SoA scratch (one ymm-pair per limb × {re, im}), run the "read A_IK
 * once, use twice" rank-1 kernel using simd_fast::cmul / cadd, unpack C
 * back. SIDE='R' holds 4 rows of C in registers across the k loop. The
 * Hermitian differences vs wsymm are: (a) A(i,k) is conjugated when used to
 * update the off-diagonal C cell, since A is stored Hermitian; (b) the
 * diagonal element A(i,i) is forced real; (c) the leading/trailing gemm wings
 * use conjugate-transpose 'C' (not 'T').
 *
 * The leading/trailing wings route through wgemm_serial (no nested OpenMP) so
 * whemm_parallel.cpp can call the block workers from inside its own omp region.
 */
#include "whemm_kernel.h"
#include "mf_util.h"
#include "mf_pred.h"
#include "mf_kernels.h"
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
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {


const R rzero{0.0, 0.0};
const T zero_cdd{ rzero, rzero };
const T one_cdd { R{1.0, 0.0}, rzero };


using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]
#define C_(i, j)  c[static_cast<std::size_t>(j) * ldc + (i)]

#ifdef MBLAS_SIMD_DD

constexpr int kSimdLane = simd_fast::NR;   /* 4 */
constexpr int kMaxBlockM = 128;          /* 4 cdd scratch × 128 × 4 = 16KB */

/* Pack `count` cells from cm[ic..ic+count, j_start..j_start+j_count) into
 * 4 SoA arrays {re_h, re_l, im_h, im_l}, indexed [0..count-1, 0..3]. */
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
            rh[i * kSimdLane + j] = 0.0;
            rl[i * kSimdLane + j] = 0.0;
            ih[i * kSimdLane + j] = 0.0;
            il[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_4col_cdd(int count, int row_start,
                            T *m, int ldm, int j_start, int j_count,
                            const double *rh, const double *rl,
                            const double *ih, const double *il)
{
    for (int j = 0; j < j_count; ++j) {
        T *col = m + static_cast<std::size_t>(j_start + j) * ldm;
        for (int i = 0; i < count; ++i) {
            col[row_start + i].re.limbs[0] = rh[i * kSimdLane + j];
            col[row_start + i].re.limbs[1] = rl[i * kSimdLane + j];
            col[row_start + i].im.limbs[0] = ih[i * kSimdLane + j];
            col[row_start + i].im.limbs[1] = il[i * kSimdLane + j];
        }
    }
}

/* Broadcast a complex DD scalar into 4 lane-wise ymm registers. */
using simd_exact::vbcast;

/* SIDE='L' Hermitian diag-block kernel, 4 column lanes.
 *
 * Differs from wsymm's simd_symm_diag_L by:
 *  - cj[k] += temp1 · conj(A(i,k))   — A's im negated for this product
 *  - diagonal A(i,i) is real (im=0)  — only A.re used
 */
inline void simd_hemm_diag_L(int ic, int ib, T alpha,
                             const T *a, int lda,
                             const double *brh, const double *brl,
                             const double *bih, const double *bil,
                             double *crh, double *crl,
                             double *cih, double *cil,
                             char UPLO)
{
    __m256d a_rh, a_rl, a_ih, a_il;
    vbcast(alpha, a_rh, a_rl, a_ih, a_il);
    const __m256d zero_v = _mm256_setzero_pd();

    auto body = [&](int i) {
        const int ir = i - ic;
        __m256d bi_rh = _mm256_load_pd(&brh[ir * kSimdLane]);
        __m256d bi_rl = _mm256_load_pd(&brl[ir * kSimdLane]);
        __m256d bi_ih = _mm256_load_pd(&bih[ir * kSimdLane]);
        __m256d bi_il = _mm256_load_pd(&bil[ir * kSimdLane]);
        __m256d t1rh, t1rl, t1ih, t1il;
        simd_fast::cmul(a_rh, a_rl, a_ih, a_il,
                         bi_rh, bi_rl, bi_ih, bi_il,
                         t1rh, t1rl, t1ih, t1il);
        __m256d t2rh = _mm256_setzero_pd();
        __m256d t2rl = _mm256_setzero_pd();
        __m256d t2ih = _mm256_setzero_pd();
        __m256d t2il = _mm256_setzero_pd();

        const int k_lo = (UPLO == 'L') ? ic       : i + 1;
        const int k_hi = (UPLO == 'L') ? i        : ic + ib;
        for (int k = k_lo; k < k_hi; ++k) {
            const int kr = k - ic;
            __m256d ak_rh, ak_rl, ak_ih, ak_il;
            vbcast(A_(i, k), ak_rh, ak_rl, ak_ih, ak_il);
            /* Conjugated copy of A(i,k) for cj[k] update */
            __m256d akc_ih = _mm256_sub_pd(zero_v, ak_ih);
            __m256d akc_il = _mm256_sub_pd(zero_v, ak_il);

            /* C[k,j] += temp1 · conj(A(i,k)) */
            __m256d ck_rh = _mm256_load_pd(&crh[kr * kSimdLane]);
            __m256d ck_rl = _mm256_load_pd(&crl[kr * kSimdLane]);
            __m256d ck_ih = _mm256_load_pd(&cih[kr * kSimdLane]);
            __m256d ck_il = _mm256_load_pd(&cil[kr * kSimdLane]);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(t1rh, t1rl, t1ih, t1il,
                             ak_rh, ak_rl, akc_ih, akc_il,
                             prh, prl, pih, pil);
            __m256d ncrh, ncrl, ncih, ncil;
            simd_fast::cadd(ck_rh, ck_rl, ck_ih, ck_il,
                             prh, prl, pih, pil,
                             ncrh, ncrl, ncih, ncil);
            _mm256_store_pd(&crh[kr * kSimdLane], ncrh);
            _mm256_store_pd(&crl[kr * kSimdLane], ncrl);
            _mm256_store_pd(&cih[kr * kSimdLane], ncih);
            _mm256_store_pd(&cil[kr * kSimdLane], ncil);

            /* temp2 += B[k,j] · A(i,k)  (no conjugate) */
            __m256d bk_rh = _mm256_load_pd(&brh[kr * kSimdLane]);
            __m256d bk_rl = _mm256_load_pd(&brl[kr * kSimdLane]);
            __m256d bk_ih = _mm256_load_pd(&bih[kr * kSimdLane]);
            __m256d bk_il = _mm256_load_pd(&bil[kr * kSimdLane]);
            __m256d qrh, qrl, qih, qil;
            simd_fast::cmul(bk_rh, bk_rl, bk_ih, bk_il,
                             ak_rh, ak_rl, ak_ih, ak_il,
                             qrh, qrl, qih, qil);
            __m256d nt2rh, nt2rl, nt2ih, nt2il;
            simd_fast::cadd(t2rh, t2rl, t2ih, t2il,
                             qrh, qrl, qih, qil,
                             nt2rh, nt2rl, nt2ih, nt2il);
            t2rh = nt2rh; t2rl = nt2rl; t2ih = nt2ih; t2il = nt2il;
        }

        /* C[i,j] += temp1 · A(i,i).re + alpha · temp2  (A diag is real) */
        const T aii = A_(i, i);
        __m256d aii_rh = _mm256_set1_pd(aii.re.limbs[0]);
        __m256d aii_rl = _mm256_set1_pd(aii.re.limbs[1]);
        __m256d d_rh, d_rl, d_ih, d_il;
        simd_fast::cmul(t1rh, t1rl, t1ih, t1il,
                         aii_rh, aii_rl, zero_v, zero_v,
                         d_rh, d_rl, d_ih, d_il);
        __m256d at_rh, at_rl, at_ih, at_il;
        simd_fast::cmul(a_rh, a_rl, a_ih, a_il,
                         t2rh, t2rl, t2ih, t2il,
                         at_rh, at_rl, at_ih, at_il);
        __m256d sum_rh, sum_rl, sum_ih, sum_il;
        simd_fast::cadd(d_rh, d_rl, d_ih, d_il,
                         at_rh, at_rl, at_ih, at_il,
                         sum_rh, sum_rl, sum_ih, sum_il);
        __m256d ci_rh = _mm256_load_pd(&crh[ir * kSimdLane]);
        __m256d ci_rl = _mm256_load_pd(&crl[ir * kSimdLane]);
        __m256d ci_ih = _mm256_load_pd(&cih[ir * kSimdLane]);
        __m256d ci_il = _mm256_load_pd(&cil[ir * kSimdLane]);
        __m256d ncirh, ncirl, nciih, nciil;
        simd_fast::cadd(ci_rh, ci_rl, ci_ih, ci_il,
                         sum_rh, sum_rl, sum_ih, sum_il,
                         ncirh, ncirl, nciih, nciil);
        _mm256_store_pd(&crh[ir * kSimdLane], ncirh);
        _mm256_store_pd(&crl[ir * kSimdLane], ncirl);
        _mm256_store_pd(&cih[ir * kSimdLane], nciih);
        _mm256_store_pd(&cil[ir * kSimdLane], nciil);
    };

    if (UPLO == 'L') for (int i = ic;          i < ic + ib;  ++i) body(i);
    else             for (int i = ic + ib - 1; i >= ic;      --i) body(i);
}

inline void simd_hemm_diag_L_panels(int ic, int ib, int N, T alpha,
                                    const T *a, int lda,
                                    const T *b, int ldb,
                                    T *c, int ldc, char UPLO)
{
    alignas(32) double brh[kMaxBlockM * kSimdLane];
    alignas(32) double brl[kMaxBlockM * kSimdLane];
    alignas(32) double bih[kMaxBlockM * kSimdLane];
    alignas(32) double bil[kMaxBlockM * kSimdLane];
    alignas(32) double crh[kMaxBlockM * kSimdLane];
    alignas(32) double crl[kMaxBlockM * kSimdLane];
    alignas(32) double cih[kMaxBlockM * kSimdLane];
    alignas(32) double cil[kMaxBlockM * kSimdLane];
    for (int j = 0; j < N; j += kSimdLane) {
        const int jc = (N - j < kSimdLane) ? (N - j) : kSimdLane;
        pack_4col_cdd(ib, ic, b, ldb, j, jc, brh, brl, bih, bil);
        pack_4col_cdd(ib, ic, c, ldc, j, jc, crh, crl, cih, cil);
        simd_hemm_diag_L(ic, ib, alpha, a, lda,
                         brh, brl, bih, bil,
                         crh, crl, cih, cil, UPLO);
        unpack_4col_cdd(ib, ic, c, ldc, j, jc, crh, crl, cih, cil);
    }
}

#endif  /* MBLAS_SIMD_DD */

void hemm_diag_add_L(int ic, int ib, int N, T alpha,
                     const T *a, int lda, const T *b, int ldb,
                     T *c, int ldc, char UPLO)
{
    for (int j = 0; j < N; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        const T *bj = b + static_cast<std::size_t>(j) * ldb;
        if (UPLO == 'L') {
            for (int i = ic; i < ic + ib; ++i) {
                const T temp1 = cmul(alpha, bj[i]);
                T temp2 = zero_cdd;
                for (int k = ic; k < i; ++k) {
                    cj[k]  = cadd(cj[k], cmul(temp1, cconj(A_(i, k))));
                    temp2  = cadd(temp2, cmul(bj[k], A_(i, k)));
                }
                /* Diagonal of A is real: take real part. */
                const T diag = T{ A_(i, i).re, rzero };
                cj[i] = cadd(cj[i], cadd(cmul(temp1, diag), cmul(alpha, temp2)));
            }
        } else {
            for (int i = ic + ib - 1; i >= ic; --i) {
                const T temp1 = cmul(alpha, bj[i]);
                T temp2 = zero_cdd;
                for (int k = i + 1; k < ic + ib; ++k) {
                    cj[k]  = cadd(cj[k], cmul(temp1, cconj(A_(i, k))));
                    temp2  = cadd(temp2, cmul(bj[k], A_(i, k)));
                }
                const T diag = T{ A_(i, i).re, rzero };
                cj[i] = cadd(cj[i], cadd(cmul(temp1, diag), cmul(alpha, temp2)));
            }
        }
    }
}

#ifdef MBLAS_SIMD_DD

using simd_exact::cload4;
using simd_exact::cstore4;

/* SIDE='R' Hermitian diag, 4-row SIMD. Differences from wsymm-R:
 *  - A(j,j) is real (im=0)
 *  - Off-diag uses conj on the stored half (mirroring scalar code) */
inline void simd_hemm_diag_R(int jc, int jb, int M, T alpha,
                             const T *a, int lda, const T *b, int ldb,
                             T *c, int ldc, char UPLO)
{
    const int M4 = M & ~3;

    for (int ib = 0; ib < M4; ib += 4) {
        for (int j = jc; j < jc + jb; ++j) {
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            __m256d crh, crl, cih, cil;
            cload4(cj + ib, crh, crl, cih, cil);

            for (int k = jc; k < jc + jb; ++k) {
                T tval;
                if (k == j) {
                    const T diag = T{ A_(j, j).re, rzero };  /* real */
                    tval = cmul(alpha, diag);
                } else if (UPLO == 'L') {
                    /* k < j: A_eff(k,j) = conj(A(j,k)) since (j,k) is in stored half;
                     * k > j: A_eff(k,j) = A(k,j) directly (stored). */
                    tval = (k < j) ? cmul(alpha, cconj(A_(j, k)))
                                   : cmul(alpha, A_(k, j));
                } else { /* UPLO=U */
                    tval = (k < j) ? cmul(alpha, A_(k, j))
                                   : cmul(alpha, cconj(A_(j, k)));
                }
                if (ceq0(tval)) continue;
                __m256d trh = _mm256_set1_pd(tval.re.limbs[0]);
                __m256d trl = _mm256_set1_pd(tval.re.limbs[1]);
                __m256d tih = _mm256_set1_pd(tval.im.limbs[0]);
                __m256d til = _mm256_set1_pd(tval.im.limbs[1]);
                const T *bk = b + static_cast<std::size_t>(k) * ldb;
                __m256d brh, brl, bih, bil;
                cload4(bk + ib, brh, brl, bih, bil);
                __m256d prh, prl, pih, pil;
                simd_fast::cmul(trh, trl, tih, til,
                                 brh, brl, bih, bil,
                                 prh, prl, pih, pil);
                __m256d nrh, nrl, nih, nil_;
                simd_fast::cadd(crh, crl, cih, cil,
                                 prh, prl, pih, pil,
                                 nrh, nrl, nih, nil_);
                crh = nrh; crl = nrl; cih = nih; cil = nil_;
            }

            cstore4(cj + ib, crh, crl, cih, cil);
        }
    }

    /* Scalar tail */
    if (M4 < M) {
        for (int j = jc; j < jc + jb; ++j) {
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            {
                const T diag = T{ A_(j, j).re, rzero };
                const T t = cmul(alpha, diag);
                for (int i = M4; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, j)));
            }
            if (UPLO == 'L') {
                for (int k = jc; k < j; ++k) {
                    const T t = cmul(alpha, cconj(A_(j, k)));
                    if (!ceq0(t)) for (int i = M4; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
                }
                for (int k = j + 1; k < jc + jb; ++k) {
                    const T t = cmul(alpha, A_(k, j));
                    if (!ceq0(t)) for (int i = M4; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
                }
            } else {
                for (int k = jc; k < j; ++k) {
                    const T t = cmul(alpha, A_(k, j));
                    if (!ceq0(t)) for (int i = M4; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
                }
                for (int k = j + 1; k < jc + jb; ++k) {
                    const T t = cmul(alpha, cconj(A_(j, k)));
                    if (!ceq0(t)) for (int i = M4; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
                }
            }
        }
    }
}

#endif  /* MBLAS_SIMD_DD */

void hemm_diag_add_R(int jc, int jb, int M, T alpha,
                     const T *a, int lda, const T *b, int ldb,
                     T *c, int ldc, char UPLO)
{
    for (int j = jc; j < jc + jb; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        {
            const T diag = T{ A_(j, j).re, rzero };
            const T t = cmul(alpha, diag);
            for (int i = 0; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, j)));
        }
        if (UPLO == 'L') {
            for (int k = jc; k < j; ++k) {
                const T t = cmul(alpha, cconj(A_(j, k)));
                if (!ceq0(t)) for (int i = 0; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
            }
            for (int k = j + 1; k < jc + jb; ++k) {
                const T t = cmul(alpha, A_(k, j));
                if (!ceq0(t)) for (int i = 0; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
            }
        } else {
            for (int k = jc; k < j; ++k) {
                const T t = cmul(alpha, A_(k, j));
                if (!ceq0(t)) for (int i = 0; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
            }
            for (int k = j + 1; k < jc + jb; ++k) {
                const T t = cmul(alpha, cconj(A_(j, k)));
                if (!ceq0(t)) for (int i = 0; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
            }
        }
    }
}

inline void diag_R_dispatch(int jc, int jb, int M, T alpha,
                            const T *a, int lda, const T *b, int ldb,
                            T *c, int ldc, char UPLO)
{
#ifdef MBLAS_SIMD_DD
    simd_hemm_diag_R(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
    return;
#else
    hemm_diag_add_R(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
#endif
}

inline void diag_L_dispatch(int ic, int ib, int N, T alpha,
                            const T *a, int lda, const T *b, int ldb,
                            T *c, int ldc, char UPLO)
{
#ifdef MBLAS_SIMD_DD
    if (ib <= kMaxBlockM) {
        simd_hemm_diag_L_panels(ic, ib, N, alpha, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }
#endif
    hemm_diag_add_L(ic, ib, N, alpha, a, lda, b, ldb, c, ldc, UPLO);
}

} /* anonymous namespace */

int whemm_block_nb(void) {
    static int nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void whemm_scale_col(int j, int M, T beta, T *c, int ldc) {
    T *cj = c + static_cast<std::size_t>(j) * ldc;
    if (ceq0(beta)) for (int i = 0; i < M; ++i) cj[i] = zero_cdd;
    else                  for (int i = 0; i < M; ++i) cj[i] = cmul(cj[i], beta);
}

void whemm_block_L(int ic, int ib, int M, int N, char UPLO,
                   T alpha, T beta, const T *a, int lda, const T *b, int ldb,
                   T *c, int ldc)
{
    const char NN[1] = {'N'};
    const char CN[1] = {'C'};

    /* beta-scale this block's rows across all columns */
    for (int j = 0; j < N; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (ceq0(beta))      for (int i = ic; i < ic + ib; ++i) cj[i] = zero_cdd;
        else if (!ceq1(beta)) for (int i = ic; i < ic + ib; ++i) cj[i] = cmul(cj[i], beta);
    }
    if (UPLO == 'L') {
        if (ic > 0) {
            wgemm_serial(NN, NN, &ib, &N, &ic, &alpha,
                         &A_(ic, 0), &lda, &B_(0, 0), &ldb,
                         &one_cdd, &C_(ic, 0), &ldc, 1, 1);
        }
        diag_L_dispatch(ic, ib, N, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const int trailing = M - ic - ib;
        if (trailing > 0) {
            wgemm_serial(CN, NN, &ib, &N, &trailing, &alpha,
                         &A_(ic + ib, ic), &lda, &B_(ic + ib, 0), &ldb,
                         &one_cdd, &C_(ic, 0), &ldc, 1, 1);
        }
    } else {
        if (ic > 0) {
            wgemm_serial(CN, NN, &ib, &N, &ic, &alpha,
                         &A_(0, ic), &lda, &B_(0, 0), &ldb,
                         &one_cdd, &C_(ic, 0), &ldc, 1, 1);
        }
        diag_L_dispatch(ic, ib, N, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const int trailing = M - ic - ib;
        if (trailing > 0) {
            wgemm_serial(NN, NN, &ib, &N, &trailing, &alpha,
                         &A_(ic, ic + ib), &lda, &B_(ic + ib, 0), &ldb,
                         &one_cdd, &C_(ic, 0), &ldc, 1, 1);
        }
    }
}

void whemm_block_R(int jc, int jb, int M, int N, char UPLO,
                   T alpha, T beta, const T *a, int lda, const T *b, int ldb,
                   T *c, int ldc)
{
    const char NN[1] = {'N'};
    const char CN[1] = {'C'};

    /* beta-scale this block's columns over all rows */
    for (int j = jc; j < jc + jb; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (ceq0(beta))      for (int i = 0; i < M; ++i) cj[i] = zero_cdd;
        else if (!ceq1(beta)) for (int i = 0; i < M; ++i) cj[i] = cmul(cj[i], beta);
    }
    if (UPLO == 'L') {
        if (jc > 0) {
            wgemm_serial(NN, CN, &M, &jb, &jc, &alpha,
                         &B_(0, 0), &ldb, &A_(jc, 0), &lda,
                         &one_cdd, &C_(0, jc), &ldc, 1, 1);
        }
        diag_R_dispatch(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            wgemm_serial(NN, NN, &M, &jb, &trailing, &alpha,
                         &B_(0, jc + jb), &ldb, &A_(jc + jb, jc), &lda,
                         &one_cdd, &C_(0, jc), &ldc, 1, 1);
        }
    } else {
        if (jc > 0) {
            wgemm_serial(NN, NN, &M, &jb, &jc, &alpha,
                         &B_(0, 0), &ldb, &A_(0, jc), &lda,
                         &one_cdd, &C_(0, jc), &ldc, 1, 1);
        }
        diag_R_dispatch(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            wgemm_serial(NN, CN, &M, &jb, &trailing, &alpha,
                         &B_(0, jc + jb), &ldb, &A_(jc, jc + jb), &lda,
                         &one_cdd, &C_(0, jc), &ldc, 1, 1);
        }
    }
}

extern "C" void whemm_serial(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    std::size_t side_len, std::size_t uplo_len)
{
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = up(side);
    const char UPLO = up(uplo);

    if (M == 0 || N == 0) return;

    if (ceq0(alpha)) {
        if (ceq1(beta)) return;
        for (int j = 0; j < N; ++j) whemm_scale_col(j, M, beta, c, ldc);
        return;
    }

    const int nb = whemm_block_nb();
    if (SIDE == 'L') {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            whemm_block_L(ic, ib, M, N, UPLO, alpha, beta, a, lda, b, ldb, c, ldc);
        }
    } else {
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            whemm_block_R(jc, jb, M, N, UPLO, alpha, beta, a, lda, b, ldb, c, ldc);
        }
    }
}

#undef A_
#undef B_
#undef C_
