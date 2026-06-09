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
#include "mgemm_kernel.h"
#include <cstddef>
#include <cstdlib>
#include <cctype>

#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {

inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

const T zero_dd{0.0, 0.0};
const T one_dd {1.0, 0.0};
inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (T x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define C_(i, j)  c[static_cast<std::size_t>(j) * ldc + (i)]

#ifdef MBLAS_SIMD_DD

constexpr int kSimdLane = simd_dd::NR;
constexpr int kMaxBlockM = 128;
constexpr int kMaxK      = 512;

inline void pack_4col(int count, int row_start,
                      const T *m, int ldm, int j_start, int j_count,
                      double *h, double *l)
{
    for (int j = 0; j < j_count; ++j) {
        const T *col = m + static_cast<std::size_t>(j_start + j) * ldm;
        for (int i = 0; i < count; ++i) {
            h[i * kSimdLane + j] = col[row_start + i].limbs[0];
            l[i * kSimdLane + j] = col[row_start + i].limbs[1];
        }
    }
    for (int j = j_count; j < kSimdLane; ++j)
        for (int i = 0; i < count; ++i) {
            h[i * kSimdLane + j] = 0.0;
            l[i * kSimdLane + j] = 0.0;
        }
}

/* Unpack only UPLO triangle of C[jc..jc+jb, j_start..j_start+j_count). */
inline void unpack_4col_triangle(int jc, int jb, int j_start, int j_count,
                                 char UPLO, T *c, int ldc,
                                 const double *h, const double *l)
{
    for (int j = 0; j < j_count; ++j) {
        const int j_abs = j_start + j;
        const int i_lo = (UPLO == 'L') ? j_abs   : jc;
        const int i_hi = (UPLO == 'L') ? jc + jb : j_abs + 1;
        T *col = c + static_cast<std::size_t>(j_abs) * ldc;
        for (int i = i_lo; i < i_hi; ++i) {
            const int ir = i - jc;
            col[i].limbs[0] = h[ir * kSimdLane + j];
            col[i].limbs[1] = l[ir * kSimdLane + j];
        }
    }
}

/* TR='N' rank-1 SIMD: for each l, broadcast A(i,l) and update
 * 4-wide C[i, j_panel..+4] using α·A(j_panel..+4, l). */
inline void simd_syrk_diag_tn(int jc, int jb, int K, T alpha,
                              const T *a, int lda,
                              int j_panel, int j_count,
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
    for (int l = 0; l < K; ++l) {
        for (int j = 0; j < j_count; ++j) {
            aj_buf_h[j] = A_(j_panel + j, l).limbs[0];
            aj_buf_l[j] = A_(j_panel + j, l).limbs[1];
        }
        for (int j = j_count; j < kSimdLane; ++j) {
            aj_buf_h[j] = 0.0; aj_buf_l[j] = 0.0;
        }
        __m256d aj_h = _mm256_load_pd(aj_buf_h);
        __m256d aj_l = _mm256_load_pd(aj_buf_l);
        /* t = alpha * Aj */
        __m256d th, tl;
        simd_dd::dd_mul(a_h, a_l, aj_h, aj_l, th, tl);
        /* For each i in diag block, update C[i, panel] += t * A(i, l) */
        for (int i = jc; i < jc + jb; ++i) {
            const int ir = i - jc;
            const T ail = A_(i, l);
            __m256d aih = _mm256_set1_pd(ail.limbs[0]);
            __m256d aili = _mm256_set1_pd(ail.limbs[1]);
            __m256d ph, pl;
            simd_dd::dd_mul(th, tl, aih, aili, ph, pl);
            __m256d ck_h = _mm256_load_pd(&ch[ir * kSimdLane]);
            __m256d ck_l = _mm256_load_pd(&cl[ir * kSimdLane]);
            __m256d nh, nl;
            simd_dd::dd_add(ck_h, ck_l, ph, pl, nh, nl);
            _mm256_store_pd(&ch[ir * kSimdLane], nh);
            _mm256_store_pd(&cl[ir * kSimdLane], nl);
        }
    }
}

/* TR='T' dot product SIMD: pre-pack Aj's 4 columns, then for each i
 * run a 4-wide SIMD accumulator across l, store to C[i, panel]. */
inline void simd_syrk_diag_tt(int jc, int jb, int K, T alpha,
                              const T *a, int lda,
                              int j_panel, int j_count,
                              const double *ajh, const double *ajl,
                              double *ch, double *cl)
{
    const __m256d a_h = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d a_l = _mm256_set1_pd(alpha.limbs[1]);
    for (int i = jc; i < jc + jb; ++i) {
        const int ir = i - jc;
        /* Ai column — read stride-1 */
        const T *Ai = a + static_cast<std::size_t>(i) * lda;
        __m256d sh = _mm256_setzero_pd();
        __m256d sl = _mm256_setzero_pd();
        for (int l = 0; l < K; ++l) {
            __m256d aih = _mm256_set1_pd(Ai[l].limbs[0]);
            __m256d aili = _mm256_set1_pd(Ai[l].limbs[1]);
            __m256d ajhv = _mm256_load_pd(&ajh[l * kSimdLane]);
            __m256d ajlv = _mm256_load_pd(&ajl[l * kSimdLane]);
            __m256d ph, pl;
            simd_dd::dd_mul(aih, aili, ajhv, ajlv, ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(sh, sl, ph, pl, nh, nl);
            sh = nh; sl = nl;
        }
        /* C[i, panel] += alpha · s */
        __m256d ph, pl;
        simd_dd::dd_mul(a_h, a_l, sh, sl, ph, pl);
        __m256d ck_h = _mm256_load_pd(&ch[ir * kSimdLane]);
        __m256d ck_l = _mm256_load_pd(&cl[ir * kSimdLane]);
        __m256d nh, nl;
        simd_dd::dd_add(ck_h, ck_l, ph, pl, nh, nl);
        _mm256_store_pd(&ch[ir * kSimdLane], nh);
        _mm256_store_pd(&cl[ir * kSimdLane], nl);
    }
}

inline void simd_syrk_diag_panels(int jc, int jb, int K, T alpha,
                                  const T *a, int lda,
                                  T *c, int ldc,
                                  char UPLO, char TR)
{
    alignas(32) double ch[kMaxBlockM * kSimdLane];
    alignas(32) double cl[kMaxBlockM * kSimdLane];
    /* For TR='T', also reserve scratch for the 4-column A pack. */
    alignas(32) static thread_local double ajh_scratch[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajl_scratch[kMaxK * kSimdLane];

    for (int j = jc; j < jc + jb; j += kSimdLane) {
        const int jcount = (jc + jb - j < kSimdLane) ? (jc + jb - j) : kSimdLane;
        pack_4col(jb, jc, c, ldc, j, jcount, ch, cl);
        if (TR == 'N') {
            simd_syrk_diag_tn(jc, jb, K, alpha, a, lda, j, jcount, ch, cl);
        } else {
            /* Pre-pack 4 columns of A: A[0..K-1, j..j+jcount] → SoA. */
            for (int jj = 0; jj < jcount; ++jj) {
                const T *col = a + static_cast<std::size_t>(j + jj) * lda;
                for (int l = 0; l < K; ++l) {
                    ajh_scratch[l * kSimdLane + jj] = col[l].limbs[0];
                    ajl_scratch[l * kSimdLane + jj] = col[l].limbs[1];
                }
            }
            for (int jj = jcount; jj < kSimdLane; ++jj)
                for (int l = 0; l < K; ++l) {
                    ajh_scratch[l * kSimdLane + jj] = 0.0;
                    ajl_scratch[l * kSimdLane + jj] = 0.0;
                }
            simd_syrk_diag_tt(jc, jb, K, alpha, a, lda, j, jcount,
                              ajh_scratch, ajl_scratch, ch, cl);
        }
        unpack_4col_triangle(jc, jb, j, jcount, UPLO, c, ldc, ch, cl);
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
                if (!dd_iszero(ajl)) {
                    const T t = alpha * ajl;
                    for (int i = i_lo; i < i_hi; ++i) cj[i] = cj[i] + t * A_(i, l);
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
                T s = zero_dd;
                for (int l = 0; l < K; ++l) s = s + Ai[l] * Aj[l];
                cj[i] = cj[i] + alpha * s;
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

int msyrk_block_nb(void) {
    static int nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void msyrk_scale_col(int j, int N, char UPLO, T beta, T *c, int ldc) {
    const int i_lo = (UPLO == 'L') ? j : 0;
    const int i_hi = (UPLO == 'L') ? N : j + 1;
    T *cj = c + static_cast<std::size_t>(j) * ldc;
    if (dd_iszero(beta)) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero_dd;
    else                 for (int i = i_lo; i < i_hi; ++i) cj[i] = cj[i] * beta;
}

void msyrk_block(int jc, int jb, int N, int K, char UPLO, char TR,
                 T alpha, T beta, const T *a, int lda, T *c, int ldc)
{
    /* Beta-scale this block's own triangle columns. */
    for (int j = jc; j < jc + jb; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (dd_iszero(beta))      for (int i = i_lo; i < i_hi; ++i) cj[i] = zero_dd;
        else if (!dd_isone(beta)) for (int i = i_lo; i < i_hi; ++i) cj[i] = cj[i] * beta;
    }

    diag_dispatch(jc, jb, K, alpha, a, lda, c, ldc, UPLO, TR);

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    if (UPLO == 'L') {
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            const int j0 = jc + jb;
            if (TR == 'N') {
                mgemm_serial(NN, TN, &trailing, &jb, &K, &alpha,
                             &A_(j0, 0), &lda, &A_(jc, 0), &lda,
                             &one_dd, &C_(j0, jc), &ldc, 1, 1);
            } else {
                mgemm_serial(TN, NN, &trailing, &jb, &K, &alpha,
                             &A_(0, j0), &lda, &A_(0, jc), &lda,
                             &one_dd, &C_(j0, jc), &ldc, 1, 1);
            }
        }
    } else {
        if (jc > 0) {
            if (TR == 'N') {
                mgemm_serial(NN, TN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &A_(jc, 0), &lda,
                             &one_dd, &C_(0, jc), &ldc, 1, 1);
            } else {
                mgemm_serial(TN, NN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &A_(0, jc), &lda,
                             &one_dd, &C_(0, jc), &ldc, 1, 1);
            }
        }
    }
}

extern "C" void msyrk_serial(
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
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    if (dd_iszero(alpha) || K == 0) {
        if (dd_isone(beta)) return;
        for (int j = 0; j < N; ++j) msyrk_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const int nb = msyrk_block_nb();
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        msyrk_block(jc, jb, N, K, UPLO, TR, alpha, beta, a, lda, c, ldc);
    }
}

#undef A_
#undef C_
