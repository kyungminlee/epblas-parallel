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
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]
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

/* TR='N' rank-2 update: for each l, broadcast α·A(j_panel..+4, l) → t1,
 * α·B(j_panel..+4, l) → t2, then for each i in diag block update
 * C[i, panel] += B(i,l)·t1 + A(i,l)·t2. */
inline void simd_syr2k_diag_tn(int jc, int jb, int K, T alpha,
                               const T *a, int lda, const T *b, int ldb,
                               int j_panel, int j_count,
                               double *ch, double *cl)
{
    const __m256d a_h = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d a_l = _mm256_set1_pd(alpha.limbs[1]);
    alignas(32) double aj_h[kSimdLane], aj_l[kSimdLane];
    alignas(32) double bj_h[kSimdLane], bj_l[kSimdLane];
    for (int ll = 0; ll < K; ++ll) {
        for (int j = 0; j < j_count; ++j) {
            aj_h[j] = A_(j_panel + j, ll).limbs[0];
            aj_l[j] = A_(j_panel + j, ll).limbs[1];
            bj_h[j] = B_(j_panel + j, ll).limbs[0];
            bj_l[j] = B_(j_panel + j, ll).limbs[1];
        }
        for (int j = j_count; j < kSimdLane; ++j) {
            aj_h[j] = 0.0; aj_l[j] = 0.0; bj_h[j] = 0.0; bj_l[j] = 0.0;
        }
        __m256d ajh = _mm256_load_pd(aj_h);
        __m256d ajl = _mm256_load_pd(aj_l);
        __m256d bjh = _mm256_load_pd(bj_h);
        __m256d bjl = _mm256_load_pd(bj_l);
        __m256d t1h, t1l, t2h, t2l;
        simd_dd::dd_mul(a_h, a_l, ajh, ajl, t1h, t1l);  /* t1 = α·Aj */
        simd_dd::dd_mul(a_h, a_l, bjh, bjl, t2h, t2l);  /* t2 = α·Bj */
        for (int i = jc; i < jc + jb; ++i) {
            const int ir = i - jc;
            const T ail = A_(i, ll);
            const T bil = B_(i, ll);
            __m256d aih = _mm256_set1_pd(ail.limbs[0]);
            __m256d aili = _mm256_set1_pd(ail.limbs[1]);
            __m256d bih = _mm256_set1_pd(bil.limbs[0]);
            __m256d bili = _mm256_set1_pd(bil.limbs[1]);
            __m256d p1h, p1l, p2h, p2l;
            simd_dd::dd_mul(bih, bili, t1h, t1l, p1h, p1l);  /* B(i,l)·t1 */
            simd_dd::dd_mul(aih, aili, t2h, t2l, p2h, p2l);  /* A(i,l)·t2 */
            __m256d sh, sl;
            simd_dd::dd_add(p1h, p1l, p2h, p2l, sh, sl);
            __m256d ck_h = _mm256_load_pd(&ch[ir * kSimdLane]);
            __m256d ck_l = _mm256_load_pd(&cl[ir * kSimdLane]);
            __m256d nh, nl;
            simd_dd::dd_add(ck_h, ck_l, sh, sl, nh, nl);
            _mm256_store_pd(&ch[ir * kSimdLane], nh);
            _mm256_store_pd(&cl[ir * kSimdLane], nl);
        }
    }
}

/* TR='T' dot product: pre-pack Aj's & Bj's 4 columns. For each i,
 * 4-wide SIMD accumulator s += Ai(l)·bj_4 + Bi(l)·aj_4. */
inline void simd_syr2k_diag_tt(int jc, int jb, int K, T alpha,
                               const T *a, int lda, const T *b, int ldb,
                               const double *ajh, const double *ajl,
                               const double *bjh, const double *bjl,
                               double *ch, double *cl)
{
    const __m256d a_h = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d a_l = _mm256_set1_pd(alpha.limbs[1]);
    for (int i = jc; i < jc + jb; ++i) {
        const int ir = i - jc;
        const T *Ai = a + static_cast<std::size_t>(i) * lda;
        const T *Bi = b + static_cast<std::size_t>(i) * ldb;
        __m256d sh = _mm256_setzero_pd();
        __m256d sl = _mm256_setzero_pd();
        for (int ll = 0; ll < K; ++ll) {
            __m256d aih = _mm256_set1_pd(Ai[ll].limbs[0]);
            __m256d aili = _mm256_set1_pd(Ai[ll].limbs[1]);
            __m256d bih = _mm256_set1_pd(Bi[ll].limbs[0]);
            __m256d bili = _mm256_set1_pd(Bi[ll].limbs[1]);
            __m256d ajv = _mm256_load_pd(&ajh[ll * kSimdLane]);
            __m256d ajvl = _mm256_load_pd(&ajl[ll * kSimdLane]);
            __m256d bjv = _mm256_load_pd(&bjh[ll * kSimdLane]);
            __m256d bjvl = _mm256_load_pd(&bjl[ll * kSimdLane]);
            __m256d p1h, p1l, p2h, p2l;
            simd_dd::dd_mul(aih, aili, bjv, bjvl, p1h, p1l);   /* Ai(l) · Bj */
            simd_dd::dd_mul(bih, bili, ajv, ajvl, p2h, p2l);   /* Bi(l) · Aj */
            __m256d ph, pl;
            simd_dd::dd_add(p1h, p1l, p2h, p2l, ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(sh, sl, ph, pl, nh, nl);
            sh = nh; sl = nl;
        }
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

inline void simd_syr2k_diag_panels(int jc, int jb, int K, T alpha,
                                   const T *a, int lda, const T *b, int ldb,
                                   T *c, int ldc, char UPLO, char TR)
{
    alignas(32) double ch[kMaxBlockM * kSimdLane];
    alignas(32) double cl[kMaxBlockM * kSimdLane];
    alignas(32) static thread_local double ajh[kMaxK * kSimdLane];
    alignas(32) static thread_local double ajl[kMaxK * kSimdLane];
    alignas(32) static thread_local double bjh[kMaxK * kSimdLane];
    alignas(32) static thread_local double bjl[kMaxK * kSimdLane];

    for (int j = jc; j < jc + jb; j += kSimdLane) {
        const int jcount = (jc + jb - j < kSimdLane) ? (jc + jb - j) : kSimdLane;
        pack_4col(jb, jc, c, ldc, j, jcount, ch, cl);
        if (TR == 'N') {
            simd_syr2k_diag_tn(jc, jb, K, alpha, a, lda, b, ldb, j, jcount, ch, cl);
        } else {
            for (int jj = 0; jj < jcount; ++jj) {
                const T *acol = a + static_cast<std::size_t>(j + jj) * lda;
                const T *bcol = b + static_cast<std::size_t>(j + jj) * ldb;
                for (int ll = 0; ll < K; ++ll) {
                    ajh[ll * kSimdLane + jj] = acol[ll].limbs[0];
                    ajl[ll * kSimdLane + jj] = acol[ll].limbs[1];
                    bjh[ll * kSimdLane + jj] = bcol[ll].limbs[0];
                    bjl[ll * kSimdLane + jj] = bcol[ll].limbs[1];
                }
            }
            for (int jj = jcount; jj < kSimdLane; ++jj)
                for (int ll = 0; ll < K; ++ll) {
                    ajh[ll * kSimdLane + jj] = 0.0; ajl[ll * kSimdLane + jj] = 0.0;
                    bjh[ll * kSimdLane + jj] = 0.0; bjl[ll * kSimdLane + jj] = 0.0;
                }
            simd_syr2k_diag_tt(jc, jb, K, alpha, a, lda, b, ldb,
                               ajh, ajl, bjh, bjl, ch, cl);
        }
        unpack_4col_triangle(jc, jb, j, jcount, UPLO, c, ldc, ch, cl);
    }
}

#endif  /* MBLAS_SIMD_DD */

void syr2k_diag_add(int jc, int jb, int K, T alpha,
                    const T *a, int lda,
                    const T *b, int ldb,
                    T *c, int ldc,
                    char UPLO, char TR)
{
    if (TR == 'N') {
        for (int j = jc; j < jc + jb; ++j) {
            const int i_lo = (UPLO == 'L') ? j     : jc;
            const int i_hi = (UPLO == 'L') ? jc+jb : j + 1;
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            for (int l = 0; l < K; ++l) {
                const T t1 = alpha * A_(j, l);
                const T t2 = alpha * B_(j, l);
                for (int i = i_lo; i < i_hi; ++i) {
                    cj[i] = cj[i] + B_(i, l) * t1 + A_(i, l) * t2;
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
                T s = zero_dd;
                for (int l = 0; l < K; ++l) s = s + Ai[l] * Bj[l] + Bi[l] * Aj[l];
                cj[i] = cj[i] + alpha * s;
            }
        }
    }
}

inline void diag_dispatch(int jc, int jb, int K, T alpha,
                          const T *a, int lda, const T *b, int ldb,
                          T *c, int ldc, char UPLO, char TR)
{
#ifdef MBLAS_SIMD_DD
    if (jb <= kMaxBlockM && K <= kMaxK) {
        simd_syr2k_diag_panels(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR);
        return;
    }
#endif
    syr2k_diag_add(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR);
}

} /* anonymous namespace */

int msyr2k_block_nb(void) {
    static int nb = 0;
    if (nb == 0) nb = 64;
    return nb;
}

void msyr2k_scale_col(int j, int N, char UPLO, T beta, T *c, int ldc) {
    const int i_lo = (UPLO == 'L') ? j : 0;
    const int i_hi = (UPLO == 'L') ? N : j + 1;
    T *cj = c + static_cast<std::size_t>(j) * ldc;
    if (dd_iszero(beta)) for (int i = i_lo; i < i_hi; ++i) cj[i] = zero_dd;
    else                 for (int i = i_lo; i < i_hi; ++i) cj[i] = cj[i] * beta;
}

void msyr2k_block(int jc, int jb, int N, int K, char UPLO, char TR,
                  T alpha, T beta, const T *a, int lda, const T *b, int ldb,
                  T *c, int ldc)
{
    /* Beta-scale this block's own triangle columns. */
    for (int j = jc; j < jc + jb; ++j) {
        const int i_lo = (UPLO == 'L') ? j : 0;
        const int i_hi = (UPLO == 'L') ? N : j + 1;
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (dd_iszero(beta))      for (int i = i_lo; i < i_hi; ++i) cj[i] = zero_dd;
        else if (!dd_isone(beta)) for (int i = i_lo; i < i_hi; ++i) cj[i] = cj[i] * beta;
    }

    diag_dispatch(jc, jb, K, alpha, a, lda, b, ldb, c, ldc, UPLO, TR);

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    if (UPLO == 'L') {
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            const int j0 = jc + jb;
            if (TR == 'N') {
                mgemm_serial(NN, TN, &trailing, &jb, &K, &alpha,
                             &A_(j0, 0), &lda, &B_(jc, 0), &ldb,
                             &one_dd, &C_(j0, jc), &ldc, 1, 1);
                mgemm_serial(NN, TN, &trailing, &jb, &K, &alpha,
                             &B_(j0, 0), &ldb, &A_(jc, 0), &lda,
                             &one_dd, &C_(j0, jc), &ldc, 1, 1);
            } else {
                mgemm_serial(TN, NN, &trailing, &jb, &K, &alpha,
                             &A_(0, j0), &lda, &B_(0, jc), &ldb,
                             &one_dd, &C_(j0, jc), &ldc, 1, 1);
                mgemm_serial(TN, NN, &trailing, &jb, &K, &alpha,
                             &B_(0, j0), &ldb, &A_(0, jc), &lda,
                             &one_dd, &C_(j0, jc), &ldc, 1, 1);
            }
        }
    } else {
        if (jc > 0) {
            if (TR == 'N') {
                mgemm_serial(NN, TN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &B_(jc, 0), &ldb,
                             &one_dd, &C_(0, jc), &ldc, 1, 1);
                mgemm_serial(NN, TN, &jc, &jb, &K, &alpha,
                             &B_(0, 0), &ldb, &A_(jc, 0), &lda,
                             &one_dd, &C_(0, jc), &ldc, 1, 1);
            } else {
                mgemm_serial(TN, NN, &jc, &jb, &K, &alpha,
                             &A_(0, 0), &lda, &B_(0, jc), &ldb,
                             &one_dd, &C_(0, jc), &ldc, 1, 1);
                mgemm_serial(TN, NN, &jc, &jb, &K, &alpha,
                             &B_(0, 0), &ldb, &A_(0, jc), &lda,
                             &one_dd, &C_(0, jc), &ldc, 1, 1);
            }
        }
    }
}

extern "C" void msyr2k_serial(
    const char *uplo, const char *trans,
    const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    std::size_t uplo_len, std::size_t trans_len)
{
    (void)uplo_len; (void)trans_len;
    const int N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const char UPLO = up(uplo);
    char TR = up(trans);
    if (TR == 'C') TR = 'T';

    if (N == 0) return;

    if (dd_iszero(alpha) || K == 0) {
        if (dd_isone(beta)) return;
        for (int j = 0; j < N; ++j) msyr2k_scale_col(j, N, UPLO, beta, c, ldc);
        return;
    }

    const int nb = msyr2k_block_nb();
    for (int jc = 0; jc < N; jc += nb) {
        const int jb = (N - jc < nb) ? (N - jc) : nb;
        msyr2k_block(jc, jb, N, K, UPLO, TR, alpha, beta, a, lda, b, ldb, c, ldc);
    }
}

#undef A_
#undef B_
#undef C_
