/*
 * msyr2k_serial — multifloats real (DD) symmetric rank-2k update, pure
 * single-thread worker. Owns ALL the numerics; no OpenMP on this path.
 *
 *   C := alpha · (A · Bᵀ + B · Aᵀ) + beta · C        (TRANS='N')
 *   C := alpha · (Aᵀ · B + Bᵀ · A) + beta · C        (TRANS='T'/'C')
 *
 * Blocked: AVX2 SIMD (or scalar) rank-2 diagonal kernel + two mgemm trailing
 * calls per off-diagonal wing. The trailing gemms route through mgemm_serial
 * (no nested OpenMP) so msyr2k_parallel.cpp can call the block worker from
 * inside its own omp region.
 */
#include "msyr2k_kernel.h"
#include "mf_util.h"
#include "mf_pred.h"
#include "mgemm_kernel.h"
#include <cstddef>
#include <cstdlib>
#include <cctype>

#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"
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

constexpr std::ptrdiff_t kSimdLane = simd_fast::NR;
constexpr std::ptrdiff_t kMaxBlockM = 128;
constexpr std::ptrdiff_t kMaxK      = 512;

inline void pack_4col(std::ptrdiff_t count, std::ptrdiff_t row_start,
                      const T *m, std::ptrdiff_t ldm, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                      double *h, double *l)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        const T *col = m + static_cast<std::size_t>(j_start + j) * ldm;
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

inline void unpack_4col_triangle(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                                 char UPLO, T *c, std::ptrdiff_t ldc,
                                 const double *h, const double *l)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        const std::ptrdiff_t j_abs = j_start + j;
        const std::ptrdiff_t i_lo = (UPLO == 'L') ? j_abs   : jc;
        const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc + jb : j_abs + 1;
        T *col = c + static_cast<std::size_t>(j_abs) * ldc;
        for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
            const std::ptrdiff_t ir = i - jc;
            col[i].limbs[0] = h[ir * kSimdLane + j];
            col[i].limbs[1] = l[ir * kSimdLane + j];
        }
    }
}

/* TR='N' rank-2 update: for each l, broadcast α·A(j_panel..+4, l) → t1,
 * α·B(j_panel..+4, l) → t2, then for each i in diag block update
 * C[i, panel] += B(i,l)·t1 + A(i,l)·t2. */
inline void simd_syr2k_diag_tn(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t K, T alpha,
                               const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                               std::ptrdiff_t j_panel, std::ptrdiff_t j_count,
                               double *ch, double *cl)
{
    const __m256d a_h = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d a_l = _mm256_set1_pd(alpha.limbs[1]);
    alignas(32) double aj_h[kSimdLane], aj_l[kSimdLane];
    alignas(32) double bj_h[kSimdLane], bj_l[kSimdLane];
    for (std::ptrdiff_t ll = 0; ll < K; ++ll) {
        for (std::ptrdiff_t j = 0; j < j_count; ++j) {
            aj_h[j] = A_(j_panel + j, ll).limbs[0];
            aj_l[j] = A_(j_panel + j, ll).limbs[1];
            bj_h[j] = B_(j_panel + j, ll).limbs[0];
            bj_l[j] = B_(j_panel + j, ll).limbs[1];
        }
        for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j) {
            aj_h[j] = 0.0; aj_l[j] = 0.0; bj_h[j] = 0.0; bj_l[j] = 0.0;
        }
        __m256d ajh = _mm256_load_pd(aj_h);
        __m256d ajl = _mm256_load_pd(aj_l);
        __m256d bjh = _mm256_load_pd(bj_h);
        __m256d bjl = _mm256_load_pd(bj_l);
        __m256d t1h, t1l, t2h, t2l;
        simd_fast::mul(a_h, a_l, ajh, ajl, t1h, t1l);  /* t1 = α·Aj */
        simd_fast::mul(a_h, a_l, bjh, bjl, t2h, t2l);  /* t2 = α·Bj */
        for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
            const std::ptrdiff_t ir = i - jc;
            const T ail = A_(i, ll);
            const T bil = B_(i, ll);
            __m256d aih = _mm256_set1_pd(ail.limbs[0]);
            __m256d aili = _mm256_set1_pd(ail.limbs[1]);
            __m256d bih = _mm256_set1_pd(bil.limbs[0]);
            __m256d bili = _mm256_set1_pd(bil.limbs[1]);
            __m256d p1h, p1l, p2h, p2l;
            simd_fast::mul(bih, bili, t1h, t1l, p1h, p1l);  /* B(i,l)·t1 */
            simd_fast::mul(aih, aili, t2h, t2l, p2h, p2l);  /* A(i,l)·t2 */
            __m256d sh, sl;
            simd_fast::add(p1h, p1l, p2h, p2l, sh, sl);
            __m256d ck_h = _mm256_load_pd(&ch[ir * kSimdLane]);
            __m256d ck_l = _mm256_load_pd(&cl[ir * kSimdLane]);
            __m256d nh, nl;
            simd_fast::add(ck_h, ck_l, sh, sl, nh, nl);
            _mm256_store_pd(&ch[ir * kSimdLane], nh);
            _mm256_store_pd(&cl[ir * kSimdLane], nl);
        }
    }
}

/* TR='T' dot product SIMD, KC-tiled: accumulate Ai(l)·Bj + Bi(l)·Aj over
 * l ∈ [l0, l0+kc) into the per-row 4-wide DD accumulator acc. ajh/.../bjl
 * hold this chunk's 4 packed A & B columns at chunk-local rows 0..kc-1.
 * acc is loaded/stored each call, so accumulation continues across chunks
 * in the same order as a single l=0..K-1 loop → bit-identical to untiled. */
inline void simd_syr2k_diag_tt_chunk(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t kc,
                                     const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                                     std::ptrdiff_t l0,
                                     const double *ajh, const double *ajl,
                                     const double *bjh, const double *bjl,
                                     double *acc_h, double *acc_l)
{
    for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
        const std::ptrdiff_t ir = i - jc;
        const T *Ai = a + static_cast<std::size_t>(i) * lda;
        const T *Bi = b + static_cast<std::size_t>(i) * ldb;
        __m256d sh = _mm256_load_pd(&acc_h[ir * kSimdLane]);
        __m256d sl = _mm256_load_pd(&acc_l[ir * kSimdLane]);
        for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
            const std::ptrdiff_t l = l0 + ll;
            __m256d aih = _mm256_set1_pd(Ai[l].limbs[0]);
            __m256d aili = _mm256_set1_pd(Ai[l].limbs[1]);
            __m256d bih = _mm256_set1_pd(Bi[l].limbs[0]);
            __m256d bili = _mm256_set1_pd(Bi[l].limbs[1]);
            __m256d ajv = _mm256_load_pd(&ajh[ll * kSimdLane]);
            __m256d ajvl = _mm256_load_pd(&ajl[ll * kSimdLane]);
            __m256d bjv = _mm256_load_pd(&bjh[ll * kSimdLane]);
            __m256d bjvl = _mm256_load_pd(&bjl[ll * kSimdLane]);
            __m256d p1h, p1l, p2h, p2l;
            simd_fast::mul(aih, aili, bjv, bjvl, p1h, p1l);   /* Ai(l) · Bj */
            simd_fast::mul(bih, bili, ajv, ajvl, p2h, p2l);   /* Bi(l) · Aj */
            __m256d ph, pl;
            simd_fast::add(p1h, p1l, p2h, p2l, ph, pl);
            __m256d nh, nl;
            simd_fast::add(sh, sl, ph, pl, nh, nl);
            sh = nh; sl = nl;
        }
        _mm256_store_pd(&acc_h[ir * kSimdLane], sh);
        _mm256_store_pd(&acc_l[ir * kSimdLane], sl);
    }
}

inline void simd_syr2k_diag_panels(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t K, T alpha,
                                   const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                                   T *c, std::ptrdiff_t ldc, char UPLO, char TR)
{
    alignas(32) double ch[kMaxBlockM * kSimdLane];
    alignas(32) double cl[kMaxBlockM * kSimdLane];
    /* TR='T' scratch: one K-chunk of 4 packed A & B columns (bounded by
     * kMaxK) plus a per-row DD accumulator carried across chunks. */
    alignas(32) static thread_local double ajh[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajl[kMaxK * kSimdLane];
    alignas(32) static thread_local double bjh[kMaxK * kSimdLane];
    alignas(32) static thread_local double bjl[kMaxK * kSimdLane];
    alignas(32) double acc_h[kMaxBlockM * kSimdLane];
    alignas(32) double acc_l[kMaxBlockM * kSimdLane];

    for (std::ptrdiff_t j = jc; j < jc + jb; j += kSimdLane) {
        const std::ptrdiff_t jcount = (jc + jb - j < kSimdLane) ? (jc + jb - j) : kSimdLane;
        pack_4col(jb, jc, c, ldc, j, jcount, ch, cl);
        if (TR == 'N') {
            /* TR='N' reads A/B directly per l — K-independent, no scratch cap. */
            simd_syr2k_diag_tn(jc, jb, K, alpha, a, lda, b, ldb, j, jcount, ch, cl);
        } else {
            const __m256d zv = _mm256_setzero_pd();
            for (std::ptrdiff_t ir = 0; ir < jb; ++ir) {
                _mm256_store_pd(&acc_h[ir * kSimdLane], zv);
                _mm256_store_pd(&acc_l[ir * kSimdLane], zv);
            }
            /* KC-tile over K so any K fits the bounded pre-pack scratch. */
            for (std::ptrdiff_t l0 = 0; l0 < K; l0 += kMaxK) {
                const std::ptrdiff_t kc = (K - l0 < kMaxK) ? (K - l0) : kMaxK;
                for (std::ptrdiff_t jj = 0; jj < jcount; ++jj) {
                    const T *acol = a + static_cast<std::size_t>(j + jj) * lda;
                    const T *bcol = b + static_cast<std::size_t>(j + jj) * ldb;
                    for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
                        ajh[ll * kSimdLane + jj] = acol[l0 + ll].limbs[0];
                        ajl[ll * kSimdLane + jj] = acol[l0 + ll].limbs[1];
                        bjh[ll * kSimdLane + jj] = bcol[l0 + ll].limbs[0];
                        bjl[ll * kSimdLane + jj] = bcol[l0 + ll].limbs[1];
                    }
                }
                for (std::ptrdiff_t jj = jcount; jj < kSimdLane; ++jj)
                    for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
                        ajh[ll * kSimdLane + jj] = 0.0; ajl[ll * kSimdLane + jj] = 0.0;
                        bjh[ll * kSimdLane + jj] = 0.0; bjl[ll * kSimdLane + jj] = 0.0;
                    }
                simd_syr2k_diag_tt_chunk(jc, jb, kc, a, lda, b, ldb, l0,
                                         ajh, ajl, bjh, bjl, acc_h, acc_l);
            }
            /* Finalize: C[panel] += alpha · acc (single alpha-mul, as untiled). */
            const __m256d a_h = _mm256_set1_pd(alpha.limbs[0]);
            const __m256d a_l = _mm256_set1_pd(alpha.limbs[1]);
            for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
                const std::ptrdiff_t ir = i - jc;
                __m256d sh = _mm256_load_pd(&acc_h[ir * kSimdLane]);
                __m256d sl = _mm256_load_pd(&acc_l[ir * kSimdLane]);
                __m256d ph, pl;
                simd_fast::mul(a_h, a_l, sh, sl, ph, pl);
                __m256d ck_h = _mm256_load_pd(&ch[ir * kSimdLane]);
                __m256d ck_l = _mm256_load_pd(&cl[ir * kSimdLane]);
                __m256d nh, nl;
                simd_fast::add(ck_h, ck_l, ph, pl, nh, nl);
                _mm256_store_pd(&ch[ir * kSimdLane], nh);
                _mm256_store_pd(&cl[ir * kSimdLane], nl);
            }
        }
        unpack_4col_triangle(jc, jb, j, jcount, UPLO, c, ldc, ch, cl);
    }
}

#endif  /* MBLAS_SIMD_DD */

void syr2k_diag_add(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t K, T alpha,
                    const T *a, std::ptrdiff_t lda,
                    const T *b, std::ptrdiff_t ldb,
                    T *c, std::ptrdiff_t ldc,
                    char UPLO, char TR)
{
    if (TR == 'N') {
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            const std::ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            for (std::ptrdiff_t l = 0; l < K; ++l) {
                const T t1 = alpha * A_(j, l);
                const T t2 = alpha * B_(j, l);
                for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                    cj[i] = cj[i] + B_(i, l) * t1 + A_(i, l) * t2;
                }
            }
        }
    } else {
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            const std::ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            const T *Aj = a + static_cast<std::size_t>(j) * lda;
            const T *Bj = b + static_cast<std::size_t>(j) * ldb;
            for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + static_cast<std::size_t>(i) * lda;
                const T *Bi = b + static_cast<std::size_t>(i) * ldb;
                T s = zero_dd;
                for (std::ptrdiff_t l = 0; l < K; ++l) s = s + Ai[l] * Bj[l] + Bi[l] * Aj[l];
                cj[i] = cj[i] + alpha * s;
            }
        }
    }
}

inline void diag_dispatch(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t K, T alpha,
                          const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                          T *c, std::ptrdiff_t ldc, char UPLO, char TR)
{
#ifdef MBLAS_SIMD_DD
    if (jb <= kMaxBlockM) {
        simd_syr2k_diag_panels(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR);
        return;
    }
#endif
    syr2k_diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR);
}

} /* anonymous namespace */

std::ptrdiff_t msyr2k_block_nb(void) {
    static std::ptrdiff_t nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void msyr2k_scale_col(std::ptrdiff_t j, std::ptrdiff_t N, char UPLO, T beta, T *c, std::ptrdiff_t ldc) {
    const std::ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
    const std::ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
    T *cj = c + static_cast<std::size_t>(j) * ldc;
    if (eq0(beta)) for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = zero_dd;
    else                 for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = cj[i] * beta;
}

void msyr2k_block(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t N, std::ptrdiff_t K, char UPLO, char TR,
                  T alpha, T beta, const T *a, std::ptrdiff_t lda, const T *b, std::ptrdiff_t ldb,
                  T *c, std::ptrdiff_t ldc)
{
    /* Beta-scale this block's own triangle columns. */
    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        const std::ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const std::ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (eq0(beta))      for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = zero_dd;
        else if (!eq1(beta)) for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = cj[i] * beta;
    }

    diag_dispatch(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR);

    if (UPLO == 'L') {
        const std::ptrdiff_t trailing = N - jc - jb;
        if (trailing > 0) {
            const std::ptrdiff_t j0 = jc + jb;
            if (TR == 'N') {
                mgemm_serial('N', 'T', trailing, jb, K, &alpha,
                             &A_(j0, 0), lda, &B_(jc, 0), ldb,
                             &one_dd, &C_(j0, jc), ldc);
                mgemm_serial('N', 'T', trailing, jb, K, &alpha,
                             &B_(j0, 0), ldb, &A_(jc, 0), lda,
                             &one_dd, &C_(j0, jc), ldc);
            } else {
                mgemm_serial('T', 'N', trailing, jb, K, &alpha,
                             &A_(0, j0), lda, &B_(0, jc), ldb,
                             &one_dd, &C_(j0, jc), ldc);
                mgemm_serial('T', 'N', trailing, jb, K, &alpha,
                             &B_(0, j0), ldb, &A_(0, jc), lda,
                             &one_dd, &C_(j0, jc), ldc);
            }
        }
    } else {
        if (jc > 0) {
            if (TR == 'N') {
                mgemm_serial('N', 'T', jc, jb, K, &alpha,
                             &A_(0, 0), lda, &B_(jc, 0), ldb,
                             &one_dd, &C_(0, jc), ldc);
                mgemm_serial('N', 'T', jc, jb, K, &alpha,
                             &B_(0, 0), ldb, &A_(jc, 0), lda,
                             &one_dd, &C_(0, jc), ldc);
            } else {
                mgemm_serial('T', 'N', jc, jb, K, &alpha,
                             &A_(0, 0), lda, &B_(0, jc), ldb,
                             &one_dd, &C_(0, jc), ldc);
                mgemm_serial('T', 'N', jc, jb, K, &alpha,
                             &B_(0, 0), ldb, &A_(0, jc), lda,
                             &one_dd, &C_(0, jc), ldc);
            }
        }
    }
}

extern "C" void msyr2k_serial(
    char uplo, char trans,
    std::ptrdiff_t N, std::ptrdiff_t K,
    const T *alpha_,
    const T *a, std::ptrdiff_t lda,
    const T *b, std::ptrdiff_t ldb,
    const T *beta_,
    T *c, std::ptrdiff_t ldc)
{
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(&uplo);
    char TR = up(&trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    if (eq0(alpha) || K == 0) {
        if (eq1(beta)) return;
        for (std::ptrdiff_t j = 0; j < N; ++j) msyr2k_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = msyr2k_block_nb();
    for (std::ptrdiff_t jc = 0; jc < N; jc += nb) {
        const std::ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        msyr2k_block(jc, jb, N, K, UPLO, TR, alpha, beta, a, lda, b, ldb, c, ldc);
    }
}

#undef A_
#undef B_
#undef C_
