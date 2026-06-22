/*
 * msyrk_serial — multifloats real (DD) symmetric rank-k update, pure
 * single-thread worker. Owns ALL the numerics; no OpenMP on this path.
 *
 * AVX2 4-wide SIMD diag kernel + mgemm_serial trailing.
 *
 * Strategy (matches the msymm pattern, adapted for rank-k shape):
 *  - For each 4-column panel of the diag block C[jc..jc+jb, j..j+4]:
 *    pack into SoA ch/cl scratch.
 *  - TR='N' rank-1 form (j outer, l middle, i inner):
 *        SIMD-load alpha · A(j..j+4, l) into a 4-wide vector,
 *        broadcast A(i, l), update C[i, j..j+4] across all i in
 *        the diag block. Computes the full square; unpack writes
 *        only the UPLO triangle.
 *  - TR='T' dot-product form:
 *        Pre-pack 4 columns of A's "Aj" panel (rows 0..K) into
 *        SoA scratch ajh/ajl so the inner l loop reads stride-1.
 *        For each i in the diag block, run a 4-wide dot product
 *        across l, store into 4-wide partial C row. Unpack only
 *        the UPLO triangle.
 *
 * Stack scratch is bounded by kMaxBlockM (rows of diag block ≤ 128).
 *
 * The trailing rank-k update routes through mgemm_serial (no nested OpenMP) so
 * msyrk_parallel.cpp can call msyrk_block from inside its own omp region.
 */
#include "msyrk_kernel.h"
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

/* Unpack only UPLO triangle of C[jc..jc+jb, j_start..j_start+j_count). */
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

/* TR='N' rank-1 SIMD: for each l, broadcast A(i,l) and update
 * 4-wide C[i, j_panel..+4] using α·A(j_panel..+4, l). */
inline void simd_syrk_diag_tn(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t K, T alpha,
                              const T *a, std::ptrdiff_t lda,
                              std::ptrdiff_t j_panel, std::ptrdiff_t j_count,
                              double *ch, double *cl)
{
    const __m256d a_h = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d a_l = _mm256_set1_pd(alpha.limbs[1]);
    /* Load A(j_panel..j_panel+4, l) for each l — 4-wide stride-1 read
     * from column l (column-major). We can load all-4 even if j_count<4
     * because pack_4col zero-pads the trailing columns of A is N/A here
     * (we read A directly); use a temp load with explicit zero for tail. */
    alignas(32) double aj_buf_h[kSimdLane];
    alignas(32) double aj_buf_l[kSimdLane];
    for (std::ptrdiff_t l = 0; l < K; ++l) {
        for (std::ptrdiff_t j = 0; j < j_count; ++j) {
            aj_buf_h[j] = A_(j_panel + j, l).limbs[0];
            aj_buf_l[j] = A_(j_panel + j, l).limbs[1];
        }
        for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j) {
            aj_buf_h[j] = 0.0; aj_buf_l[j] = 0.0;
        }
        __m256d aj_h = _mm256_load_pd(aj_buf_h);
        __m256d aj_l = _mm256_load_pd(aj_buf_l);
        /* t = alpha * Aj */
        __m256d th, tl;
        simd_fast::mul(a_h, a_l, aj_h, aj_l, th, tl);
        /* For each i in diag block, update C[i, panel] += t * A(i, l) */
        for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
            const std::ptrdiff_t ir = i - jc;
            const T ail = A_(i, l);
            __m256d aih = _mm256_set1_pd(ail.limbs[0]);
            __m256d aili = _mm256_set1_pd(ail.limbs[1]);
            __m256d ph, pl;
            simd_fast::mul(th, tl, aih, aili, ph, pl);
            __m256d ck_h = _mm256_load_pd(&ch[ir * kSimdLane]);
            __m256d ck_l = _mm256_load_pd(&cl[ir * kSimdLane]);
            __m256d nh, nl;
            simd_fast::add(ck_h, ck_l, ph, pl, nh, nl);
            _mm256_store_pd(&ch[ir * kSimdLane], nh);
            _mm256_store_pd(&cl[ir * kSimdLane], nl);
        }
    }
}

/* TR='T' dot product SIMD, KC-tiled: accumulate the dot over l ∈
 * [l0, l0+kc) into the per-row 4-wide DD accumulator acc. ajh/ajl hold
 * this chunk's 4 packed A columns at chunk-local rows 0..kc-1. acc is
 * loaded/stored each call, so accumulation continues across chunks in the
 * same order as a single l=0..K-1 loop → bit-identical to the untiled path. */
inline void simd_syrk_diag_tt_chunk(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t kc,
                                    const T *a, std::ptrdiff_t lda, std::ptrdiff_t l0,
                                    const double *ajh, const double *ajl,
                                    double *acc_h, double *acc_l)
{
    for (std::ptrdiff_t i = jc; i < jc + jb; ++i) {
        const std::ptrdiff_t ir = i - jc;
        /* Ai column — read stride-1 */
        const T *Ai = a + static_cast<std::size_t>(i) * lda;
        __m256d sh = _mm256_load_pd(&acc_h[ir * kSimdLane]);
        __m256d sl = _mm256_load_pd(&acc_l[ir * kSimdLane]);
        for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
            const T ail = Ai[l0 + ll];
            __m256d aih = _mm256_set1_pd(ail.limbs[0]);
            __m256d aili = _mm256_set1_pd(ail.limbs[1]);
            __m256d ajhv = _mm256_load_pd(&ajh[ll * kSimdLane]);
            __m256d ajlv = _mm256_load_pd(&ajl[ll * kSimdLane]);
            __m256d ph, pl;
            simd_fast::mul(aih, aili, ajhv, ajlv, ph, pl);
            __m256d nh, nl;
            simd_fast::add(sh, sl, ph, pl, nh, nl);
            sh = nh; sl = nl;
        }
        _mm256_store_pd(&acc_h[ir * kSimdLane], sh);
        _mm256_store_pd(&acc_l[ir * kSimdLane], sl);
    }
}

inline void simd_syrk_diag_panels(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t K, T alpha,
                                  const T *a, std::ptrdiff_t lda,
                                  T *c, std::ptrdiff_t ldc,
                                  char UPLO, char TR)
{
    alignas(32) double ch[kMaxBlockM * kSimdLane];
    alignas(32) double cl[kMaxBlockM * kSimdLane];
    /* TR='T' scratch: one K-chunk of 4 packed A columns (bounded by kMaxK)
     * plus a per-row DD accumulator carried across chunks. */
    alignas(32) static thread_local double ajh_scratch[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajl_scratch[kMaxK * kSimdLane];
    alignas(32) double acc_h[kMaxBlockM * kSimdLane];
    alignas(32) double acc_l[kMaxBlockM * kSimdLane];

    for (std::ptrdiff_t j = jc; j < jc + jb; j += kSimdLane) {
        const std::ptrdiff_t jcount = (jc + jb - j < kSimdLane) ? (jc + jb - j) : kSimdLane;
        pack_4col(jb, jc, c, ldc, j, jcount, ch, cl);
        if (TR == 'N') {
            /* TR='N' reads A directly per l — K-independent, no scratch cap. */
            simd_syrk_diag_tn(jc, jb, K, alpha, a, lda, j, jcount, ch, cl);
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
                    const T *col = a + static_cast<std::size_t>(j + jj) * lda;
                    for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
                        ajh_scratch[ll * kSimdLane + jj] = col[l0 + ll].limbs[0];
                        ajl_scratch[ll * kSimdLane + jj] = col[l0 + ll].limbs[1];
                    }
                }
                for (std::ptrdiff_t jj = jcount; jj < kSimdLane; ++jj)
                    for (std::ptrdiff_t ll = 0; ll < kc; ++ll) {
                        ajh_scratch[ll * kSimdLane + jj] = 0.0;
                        ajl_scratch[ll * kSimdLane + jj] = 0.0;
                    }
                simd_syrk_diag_tt_chunk(jc, jb, kc, a, lda, l0,
                                        ajh_scratch, ajl_scratch, acc_h, acc_l);
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

void syrk_diag_add(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t K, T alpha,
                   const T *a, std::ptrdiff_t lda, T *c, std::ptrdiff_t ldc,
                   char UPLO, char TR)
{
    if (TR == 'N') {
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            const std::ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            for (std::ptrdiff_t l = 0; l < K; ++l) {
                const T ajl = A_(j, l);
                if (!eq0(ajl)) {
                    const T t = alpha * ajl;
                    for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = cj[i] + t * A_(i, l);
                }
            }
        }
    } else {
        for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
            const std::ptrdiff_t i_lo = (UPLO == 'L') ? j     : jc;
            const std::ptrdiff_t i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            const T *Aj = a + static_cast<std::size_t>(j) * lda;
            for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) {
                const T *Ai = a + static_cast<std::size_t>(i) * lda;
                T s = zero_dd;
                for (std::ptrdiff_t l = 0; l < K; ++l) s = s + Ai[l] * Aj[l];
                cj[i] = cj[i] + alpha * s;
            }
        }
    }
}

inline void diag_dispatch(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t K, T alpha,
                          const T *a, std::ptrdiff_t lda, T *c, std::ptrdiff_t ldc,
                          char UPLO, char TR)
{
#ifdef MBLAS_SIMD_DD
    if (jb <= kMaxBlockM) {
        simd_syrk_diag_panels(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);
        return;
    }
#endif
    syrk_diag_add(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);
}

} /* anonymous namespace */

std::ptrdiff_t msyrk_block_nb(void) {
    static std::ptrdiff_t nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void msyrk_scale_col(std::ptrdiff_t j, std::ptrdiff_t N, char UPLO, T beta, T *c, std::ptrdiff_t ldc) {
    const std::ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
    const std::ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
    T *cj = c + static_cast<std::size_t>(j) * ldc;
    if (eq0(beta)) for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = zero_dd;
    else                 for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = cj[i] * beta;
}

void msyrk_block(std::ptrdiff_t jc, std::ptrdiff_t jb, std::ptrdiff_t N, std::ptrdiff_t K, char UPLO, char TR,
                 T alpha, T beta, const T *a, std::ptrdiff_t lda, T *c, std::ptrdiff_t ldc)
{
    /* Beta-scale this block's own triangle columns. */
    for (std::ptrdiff_t j = jc; j < jc + jb; ++j) {
        const std::ptrdiff_t i_lo = (UPLO == 'L') ? j : 0;
        const std::ptrdiff_t i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (eq0(beta))      for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = zero_dd;
        else if (!eq1(beta)) for (std::ptrdiff_t i = i_lo; i < i_hi; ++i) cj[i] = cj[i] * beta;
    }

    diag_dispatch(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);

    if (UPLO == 'L') {
        const std::ptrdiff_t trailing = N - jc - jb;
        if (trailing > 0) {
            const std::ptrdiff_t j0 = jc + jb;
            if (TR == 'N') {
                mgemm_serial('N', 'T', trailing, jb, K, &alpha,
                             &A_(j0, 0), lda, &A_(jc, 0), lda,
                             &one_dd, &C_(j0, jc), ldc);
            } else {
                mgemm_serial('T', 'N', trailing, jb, K, &alpha,
                             &A_(0, j0), lda, &A_(0, jc), lda,
                             &one_dd, &C_(j0, jc), ldc);
            }
        }
    } else {
        if (jc > 0) {
            if (TR == 'N') {
                mgemm_serial('N', 'T', jc, jb, K, &alpha,
                             &A_(0, 0), lda, &A_(jc, 0), lda,
                             &one_dd, &C_(0, jc), ldc);
            } else {
                mgemm_serial('T', 'N', jc, jb, K, &alpha,
                             &A_(0, 0), lda, &A_(0, jc), lda,
                             &one_dd, &C_(0, jc), ldc);
            }
        }
    }
}

extern "C" void msyrk_serial(
    char uplo, char trans,
    std::ptrdiff_t N, std::ptrdiff_t K,
    const T *alpha_,
    const T *a, std::ptrdiff_t lda,
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
        for (std::ptrdiff_t j = 0; j < N; ++j) msyrk_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const std::ptrdiff_t nb = msyrk_block_nb();
    for (std::ptrdiff_t jc = 0; jc < N; jc += nb) {
        const std::ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
        msyrk_block(jc, jb, N, K, UPLO, TR, alpha, beta, a, lda, c, ldc);
    }
}

#undef A_
#undef C_
