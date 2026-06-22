/*
 * msymm_serial — multifloats real (DD) symmetric matrix multiply, pure
 * single-thread worker. Owns ALL the numerics; no OpenMP on this path.
 *
 * Blocked: AVX2 4-wide SIMD diagonal kernel + mgemm_serial trailing update.
 *
 * SIMD design (mirrors mtrsm's diagonal-block strategy):
 *   - SIDE='L': pack 4 columns of B and C into SoA stack scratch, run the
 *     rank-1 i,k inner kernel with A as scalar broadcasts and B,C as 4-wide
 *     SIMD lanes (4 columns of C updated in parallel), unpack C back.
 *   - SIDE='R': hold 4 rows of C in 2 ymm regs across the k loop.
 *
 * The leading/trailing rank-k wings route through mgemm_serial (no nested
 * OpenMP) so msymm_parallel.cpp can call the block workers from inside its
 * own omp region.
 */
#include "msymm_kernel.h"
#include "mf_util.h"
#include "mf_pred.h"
#include "mgemm_kernel.h"
#include <cstddef>
#include <cstdlib>
#include <cctype>

#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;

using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
namespace {


const T zero_dd{0.0, 0.0};
const T one_dd {1.0, 0.0};

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]
#define C_(i, j)  c[static_cast<std::size_t>(j) * ldc + (i)]

#ifdef MBLAS_SIMD_DD

constexpr std::ptrdiff_t kSimdLane = simd_fast::NR;   /* 4 */
constexpr std::ptrdiff_t kMaxBlockM = 256;

/* Pack `count` cells from B[ic..ic+count, j_start..j_start+j_count)
 * into SoA scratch [bh,bl][0..count-1, 0..3]. Zero-pad lanes ≥ j_count. */
inline void pack_4col(std::ptrdiff_t count, std::ptrdiff_t row_start,
                      const T *mat, std::ptrdiff_t ldm, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                      double *h, double *l)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        const T *col = mat + static_cast<std::size_t>(j_start + j) * ldm;
        for (std::ptrdiff_t i = 0; i < count; ++i) {
            h[i * kSimdLane + j] = col[row_start + i].limbs[0];
            l[i * kSimdLane + j] = col[row_start + i].limbs[1];
        }
    }
    for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j)
        for (std::ptrdiff_t i = 0; i < count; ++i) {
            h[i * kSimdLane + j] = 0.0;
            l[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_4col(std::ptrdiff_t count, std::ptrdiff_t row_start,
                        T *mat, std::ptrdiff_t ldm, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                        const double *h, const double *l)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        T *col = mat + static_cast<std::size_t>(j_start + j) * ldm;
        for (std::ptrdiff_t i = 0; i < count; ++i) {
            col[row_start + i].limbs[0] = h[i * kSimdLane + j];
            col[row_start + i].limbs[1] = l[i * kSimdLane + j];
        }
    }
}

/* SIDE='L' symmetric-multiply diag-block kernel, 4 column lanes.
 *
 * For each i in the diag block (rows ic..ic+ib of A), apply the
 * symmetric "read A(i,k) once, use twice" pattern:
 *   temp1 = alpha · B[i,j]
 *   For k in same-half of triangle (k<i for L; k>i for U):
 *     C[k,j] += temp1 · A(i,k)            (off-diag scatter)
 *     temp2  += B[k,j] · A(i,k)           (per-lane accumulator)
 *   C[i,j] += temp1 · A(i,i) + alpha · temp2
 *
 * Packed scratch is block-relative: row index 0..ib-1 maps to absolute
 * row ic+0..ic+ib-1. A is read via absolute (i,k) indices.
 */
inline void simd_symm_diag_L(std::ptrdiff_t ic, std::ptrdiff_t ib, T alpha,
                             const T *a, std::ptrdiff_t lda,
                             const double *bh, const double *bl,
                             double *ch, double *cl,
                             char UPLO)
{
    const __m256d alpha_h = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d alpha_l = _mm256_set1_pd(alpha.limbs[1]);

    auto body = [&](std::ptrdiff_t i) {
        const std::ptrdiff_t ir = i - ic;
        __m256d bi_h = _mm256_load_pd(&bh[ir * kSimdLane]);
        __m256d bi_l = _mm256_load_pd(&bl[ir * kSimdLane]);
        __m256d t1h, t1l;
        simd_fast::mul(alpha_h, alpha_l, bi_h, bi_l, t1h, t1l);
        __m256d t2h = _mm256_setzero_pd();
        __m256d t2l = _mm256_setzero_pd();

        const std::ptrdiff_t k_lo = (UPLO == 'L') ? ic       : i + 1;
        const std::ptrdiff_t k_hi = (UPLO == 'L') ? i        : ic + ib;
        for (std::ptrdiff_t k = k_lo; k < k_hi; ++k) {
            const std::ptrdiff_t kr = k - ic;
            const T aik = A_(i, k);
            __m256d aih = _mm256_set1_pd(aik.limbs[0]);
            __m256d ail = _mm256_set1_pd(aik.limbs[1]);
            /* C[k,j] += temp1 · A(i,k) */
            __m256d ck_h = _mm256_load_pd(&ch[kr * kSimdLane]);
            __m256d ck_l = _mm256_load_pd(&cl[kr * kSimdLane]);
            __m256d ph, pl;
            simd_fast::mul(t1h, t1l, aih, ail, ph, pl);
            __m256d new_ckh, new_ckl;
            simd_fast::add(ck_h, ck_l, ph, pl, new_ckh, new_ckl);
            _mm256_store_pd(&ch[kr * kSimdLane], new_ckh);
            _mm256_store_pd(&cl[kr * kSimdLane], new_ckl);
            /* temp2 += B[k,j] · A(i,k) */
            __m256d bk_h = _mm256_load_pd(&bh[kr * kSimdLane]);
            __m256d bk_l = _mm256_load_pd(&bl[kr * kSimdLane]);
            __m256d qh, ql;
            simd_fast::mul(bk_h, bk_l, aih, ail, qh, ql);
            __m256d new_t2h, new_t2l;
            simd_fast::add(t2h, t2l, qh, ql, new_t2h, new_t2l);
            t2h = new_t2h; t2l = new_t2l;
        }
        /* Diagonal cell: C[i,j] += temp1·A(i,i) + alpha·temp2 */
        const T aii = A_(i, i);
        __m256d aii_h = _mm256_set1_pd(aii.limbs[0]);
        __m256d aii_l = _mm256_set1_pd(aii.limbs[1]);
        __m256d diag_h, diag_l;
        simd_fast::mul(t1h, t1l, aii_h, aii_l, diag_h, diag_l);
        __m256d at2h, at2l;
        simd_fast::mul(alpha_h, alpha_l, t2h, t2l, at2h, at2l);
        __m256d sum_h, sum_l;
        simd_fast::add(diag_h, diag_l, at2h, at2l, sum_h, sum_l);
        __m256d ci_h = _mm256_load_pd(&ch[ir * kSimdLane]);
        __m256d ci_l = _mm256_load_pd(&cl[ir * kSimdLane]);
        __m256d new_cih, new_cil;
        simd_fast::add(ci_h, ci_l, sum_h, sum_l, new_cih, new_cil);
        _mm256_store_pd(&ch[ir * kSimdLane], new_cih);
        _mm256_store_pd(&cl[ir * kSimdLane], new_cil);
    };

    if (UPLO == 'L') for (std::ptrdiff_t i = ic;          i < ic + ib;  ++i) body(i);
    else             for (std::ptrdiff_t i = ic + ib - 1; i >= ic;      --i) body(i);
}

/* Drive the SIMD diag over a column range [0..N) in 4-column panels. */
inline void simd_symm_diag_L_panels(std::ptrdiff_t ic, std::ptrdiff_t ib, std::ptrdiff_t n, T alpha,
                                    const T *a, std::ptrdiff_t lda,
                                    const T *b, std::ptrdiff_t ldb,
                                    T *c, std::ptrdiff_t ldc, char UPLO)
{
    alignas(32) double bh[kMaxBlockM * kSimdLane];
    alignas(32) double bl[kMaxBlockM * kSimdLane];
    alignas(32) double ch[kMaxBlockM * kSimdLane];
    alignas(32) double cl[kMaxBlockM * kSimdLane];
    for (std::ptrdiff_t j = 0; j < n; j += kSimdLane) {
        const std::ptrdiff_t jc = (n - j < kSimdLane) ? (n - j) : kSimdLane;
        pack_4col(ib, ic, b, ldb, j, jc, bh, bl);
        pack_4col(ib, ic, c, ldc, j, jc, ch, cl);
        simd_symm_diag_L(ic, ib, alpha, a, lda, bh, bl, ch, cl, UPLO);
        unpack_4col(ib, ic, c, ldc, j, jc, ch, cl);
    }
}

#endif  /* MBLAS_SIMD_DD */

/* Scalar fallback diag for SIDE='L' (also used when SIMD off or
 * block too big for stack scratch). */
void symm_diag_add_L(std::ptrdiff_t ic, std::ptrdiff_t ib, std::ptrdiff_t n, T alpha,
                     const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                     T *c, std::ptrdiff_t ldc, char UPLO)
{
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        const T *bj = b + static_cast<std::size_t>(j) * ldb;
        if (UPLO == 'L') {
            for (std::ptrdiff_t i = ic; i < ic + ib; ++i) {
                const T temp1 = alpha * bj[i];
                T temp2 = zero_dd;
                for (std::ptrdiff_t k = ic; k < i; ++k) {
                    cj[k]  = cj[k]  + temp1 * A_(i, k);
                    temp2  = temp2  + bj[k] * A_(i, k);
                }
                cj[i] = cj[i] + temp1 * A_(i, i) + alpha * temp2;
            }
        } else {
            for (std::ptrdiff_t i = ic + ib - 1; i >= ic; --i) {
                const T temp1 = alpha * bj[i];
                T temp2 = zero_dd;
                for (std::ptrdiff_t k = i + 1; k < ic + ib; ++k) {
                    cj[k]  = cj[k]  + temp1 * A_(i, k);
                    temp2  = temp2  + bj[k] * A_(i, k);
                }
                cj[i] = cj[i] + temp1 * A_(i, i) + alpha * temp2;
            }
        }
    }
}

#ifdef MBLAS_SIMD_DD

/* AoS→SoA 4-cell transpose load: canonical simd_exact::load_dd4 (col + ofs). */
using simd_exact::load_dd4;
using simd_exact::store_dd4;

/* SIDE='R' symmetric diag-block kernel, 4-row SIMD.
 *
 * For each i-block of 4 rows: hold C[i..i+3, j] in 2 ymm regs across
 * the k loop, accumulate α·A_eff(j,k)·B[i..i+3, k] for k ∈ [jc, jc+jb),
 * store back. A_eff uses the symmetric mirror via UPLO.
 * Tail (M % 4 != 0) falls back to scalar. */
inline void simd_symm_diag_R(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t m, T alpha,
                             const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                             T *c, std::ptrdiff_t ldc, char UPLO)
{
    const std::ptrdiff_t M4 = m & ~3;

    for (std::ptrdiff_t ib = 0; ib < M4; ib += 4) {
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            __m256d ch, cl;
            load_dd4(cj + ib, ch, cl);

            for (std::ptrdiff_t k = jc; k < jc + jb; ++k) {
                T tval;
                if (k == j)                  tval = alpha * A_(j, j);
                else if (UPLO == 'L')        tval = (k < j) ? (alpha * A_(j, k))
                                                            : (alpha * A_(k, j));
                else /* UPLO == 'U' */       tval = (k < j) ? (alpha * A_(k, j))
                                                            : (alpha * A_(j, k));
                if (eq0(tval)) continue;
                const __m256d th = _mm256_set1_pd(tval.limbs[0]);
                const __m256d tl = _mm256_set1_pd(tval.limbs[1]);
                const T *bk = b + static_cast<std::size_t>(k) * ldb;
                __m256d bh, bl;
                load_dd4(bk + ib, bh, bl);
                __m256d ph, pl;
                simd_fast::mul(th, tl, bh, bl, ph, pl);
                __m256d nh, nl;
                simd_fast::add(ch, cl, ph, pl, nh, nl);
                ch = nh; cl = nl;
            }

            store_dd4(cj + ib, ch, cl);
        }
    }

    /* Scalar tail rows (at most 3) */
    if (M4 < m) {
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            {
                const T t = alpha * A_(j, j);
                for (std::ptrdiff_t i = M4; i < m; ++i) cj[i] = cj[i] + t * B_(i, j);
            }
            if (UPLO == 'L') {
                for (std::ptrdiff_t k = jc; k < j; ++k) {
                    const T t = alpha * A_(j, k);
                    if (!eq0(t)) for (std::ptrdiff_t i = M4; i < m; ++i) cj[i] = cj[i] + t * B_(i, k);
                }
                for (std::ptrdiff_t k = j + 1; k < jc + jb; ++k) {
                    const T t = alpha * A_(k, j);
                    if (!eq0(t)) for (std::ptrdiff_t i = M4; i < m; ++i) cj[i] = cj[i] + t * B_(i, k);
                }
            } else {
                for (std::ptrdiff_t k = jc; k < j; ++k) {
                    const T t = alpha * A_(k, j);
                    if (!eq0(t)) for (std::ptrdiff_t i = M4; i < m; ++i) cj[i] = cj[i] + t * B_(i, k);
                }
                for (std::ptrdiff_t k = j + 1; k < jc + jb; ++k) {
                    const T t = alpha * A_(j, k);
                    if (!eq0(t)) for (std::ptrdiff_t i = M4; i < m; ++i) cj[i] = cj[i] + t * B_(i, k);
                }
            }
        }
    }
}

#endif  /* MBLAS_SIMD_DD */

void symm_diag_add_R(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t m, T alpha,
                     const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                     T *c, std::ptrdiff_t ldc, char UPLO)
{
    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        {
            const T t = alpha * A_(j, j);
            for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cj[i] + t * B_(i, j);
        }
        if (UPLO == 'L') {
            for (std::ptrdiff_t k = jc; k < j; ++k) {
                const T t = alpha * A_(j, k);
                if (!eq0(t)) for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cj[i] + t * B_(i, k);
            }
            for (std::ptrdiff_t k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * A_(k, j);
                if (!eq0(t)) for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cj[i] + t * B_(i, k);
            }
        } else {
            for (std::ptrdiff_t k = jc; k < j; ++k) {
                const T t = alpha * A_(k, j);
                if (!eq0(t)) for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cj[i] + t * B_(i, k);
            }
            for (std::ptrdiff_t k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * A_(j, k);
                if (!eq0(t)) for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cj[i] + t * B_(i, k);
            }
        }
    }
}

inline void diag_R_dispatch(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t m, T alpha,
                            const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                            T *c, std::ptrdiff_t ldc, char UPLO)
{
#ifdef MBLAS_SIMD_DD
    simd_symm_diag_R(jc, jb, m, alpha, a, lda, b, ldb, c, ldc, UPLO);
    return;
#else
    symm_diag_add_R(jc, jb, m, alpha, a, lda, b, ldb, c, ldc, UPLO);
#endif
}

inline void diag_L_dispatch(std::ptrdiff_t ic, std::ptrdiff_t ib, std::ptrdiff_t n, T alpha,
                            const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                            T *c, std::ptrdiff_t ldc, char UPLO)
{
#ifdef MBLAS_SIMD_DD
    if (ib <= kMaxBlockM) {
        simd_symm_diag_L_panels(ic, ib, n, alpha, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }
#endif
    symm_diag_add_L(ic, ib, n, alpha, a, lda, b, ldb, c, ldc, UPLO);
}

} /* anonymous namespace */

std::ptrdiff_t msymm_block_nb(void) {
    static std::ptrdiff_t nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void msymm_scale_col(std::ptrdiff_t j, std::ptrdiff_t m, T beta, T *c, std::ptrdiff_t ldc) {
    T *cj = c + static_cast<std::size_t>(j) * ldc;
    if (eq0(beta)) for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = zero_dd;
    else                 for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cj[i] * beta;
}

void msymm_block_L(std::ptrdiff_t ic, std::ptrdiff_t ib, std::ptrdiff_t m, std::ptrdiff_t n, char UPLO,
                   T alpha, T beta, const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                   T *c, std::ptrdiff_t ldc)
{
    /* beta-scale this block's rows across all columns */
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (eq0(beta))      for (std::ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] = zero_dd;
        else if (!eq1(beta)) for (std::ptrdiff_t i = ic; i < ic + ib; ++i) cj[i] = cj[i] * beta;
    }
    if (UPLO == 'L') {
        if (ic > 0) {
            mgemm_serial('N', 'N', ib, n, ic, &alpha,
                         &A_(ic, 0), lda, &B_(0, 0), ldb,
                         &one_dd, &C_(ic, 0), ldc);
        }
        diag_L_dispatch(ic, ib, n, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const std::ptrdiff_t trailing = m - ic - ib;
        if (trailing > 0) {
            mgemm_serial('T', 'N', ib, n, trailing, &alpha,
                         &A_(ic + ib, ic), lda, &B_(ic + ib, 0), ldb,
                         &one_dd, &C_(ic, 0), ldc);
        }
    } else {
        if (ic > 0) {
            mgemm_serial('T', 'N', ib, n, ic, &alpha,
                         &A_(0, ic), lda, &B_(0, 0), ldb,
                         &one_dd, &C_(ic, 0), ldc);
        }
        diag_L_dispatch(ic, ib, n, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const std::ptrdiff_t trailing = m - ic - ib;
        if (trailing > 0) {
            mgemm_serial('N', 'N', ib, n, trailing, &alpha,
                         &A_(ic, ic + ib), lda, &B_(ic + ib, 0), ldb,
                         &one_dd, &C_(ic, 0), ldc);
        }
    }
}

void msymm_block_R(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t m, std::ptrdiff_t n, char UPLO,
                   T alpha, T beta, const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                   T *c, std::ptrdiff_t ldc)
{
    /* beta-scale this block's columns over all rows */
    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (eq0(beta))      for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = zero_dd;
        else if (!eq1(beta)) for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cj[i] * beta;
    }
    if (UPLO == 'L') {
        if (jc > 0) {
            mgemm_serial('N', 'T', m, jb, jc, &alpha,
                         &B_(0, 0), ldb, &A_(jc, 0), lda,
                         &one_dd, &C_(0, jc), ldc);
        }
        diag_R_dispatch(jc, jb, m, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const std::ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            mgemm_serial('N', 'N', m, jb, trailing, &alpha,
                         &B_(0, jc + jb), ldb, &A_(jc + jb, jc), lda,
                         &one_dd, &C_(0, jc), ldc);
        }
    } else {
        if (jc > 0) {
            mgemm_serial('N', 'N', m, jb, jc, &alpha,
                         &B_(0, 0), ldb, &A_(0, jc), lda,
                         &one_dd, &C_(0, jc), ldc);
        }
        diag_R_dispatch(jc, jb, m, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const std::ptrdiff_t trailing = n - jc - jb;
        if (trailing > 0) {
            mgemm_serial('N', 'T', m, jb, trailing, &alpha,
                         &B_(0, jc + jb), ldb, &A_(jc, jc + jb), lda,
                         &one_dd, &C_(0, jc), ldc);
        }
    }
}

extern "C" void msymm_serial(
    char side, char uplo,
    std::ptrdiff_t m, std::ptrdiff_t n,
    const T *alpha_,
    const T *a, std::ptrdiff_t lda,
    const T *b, std::ptrdiff_t ldb,
    const T *beta_,
    T *c, std::ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);

    if (m == 0 || n == 0) return;

    if (eq0(alpha)) {
        if (eq1(beta)) return;
        for (std::ptrdiff_t j = 0; j < n; ++j) msymm_scale_col(j, m, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = msymm_block_nb();
    if (SIDE == 'L') {
        for (std::ptrdiff_t ic = 0; ic < m; ic += nb) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            msymm_block_L(ic, ib, m, n, UPLO, alpha, beta, a, lda, b, ldb, c, ldc);
        }
    } else {
        for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
            const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
            msymm_block_R(jc, jb, m, n, UPLO, alpha, beta, a, lda, b, ldb, c, ldc);
        }
    }
}

#undef A_
#undef B_
#undef C_
