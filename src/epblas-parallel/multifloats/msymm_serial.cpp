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

int env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    int v = std::atoi(s);
    return v > 0 ? v : dflt;
}

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

constexpr int kSimdLane = simd_dd::NR;   /* 4 */
constexpr int kMaxBlockM = 256;

/* Pack `count` cells from B[ic..ic+count, j_start..j_start+j_count)
 * into SoA scratch [bh,bl][0..count-1, 0..3]. Zero-pad lanes ≥ j_count. */
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

inline void unpack_4col(int count, int row_start,
                        T *m, int ldm, int j_start, int j_count,
                        const double *h, const double *l)
{
    for (int j = 0; j < j_count; ++j) {
        T *col = m + static_cast<std::size_t>(j_start + j) * ldm;
        for (int i = 0; i < count; ++i) {
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
inline void simd_symm_diag_L(int ic, int ib, T alpha,
                             const T *a, int lda,
                             const double *bh, const double *bl,
                             double *ch, double *cl,
                             char UPLO)
{
    const __m256d alpha_h = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d alpha_l = _mm256_set1_pd(alpha.limbs[1]);

    auto body = [&](int i) {
        const int ir = i - ic;
        __m256d bi_h = _mm256_load_pd(&bh[ir * kSimdLane]);
        __m256d bi_l = _mm256_load_pd(&bl[ir * kSimdLane]);
        __m256d t1h, t1l;
        simd_dd::dd_mul(alpha_h, alpha_l, bi_h, bi_l, t1h, t1l);
        __m256d t2h = _mm256_setzero_pd();
        __m256d t2l = _mm256_setzero_pd();

        const int k_lo = (UPLO == 'L') ? ic       : i + 1;
        const int k_hi = (UPLO == 'L') ? i        : ic + ib;
        for (int k = k_lo; k < k_hi; ++k) {
            const int kr = k - ic;
            const T aik = A_(i, k);
            __m256d aih = _mm256_set1_pd(aik.limbs[0]);
            __m256d ail = _mm256_set1_pd(aik.limbs[1]);
            /* C[k,j] += temp1 · A(i,k) */
            __m256d ck_h = _mm256_load_pd(&ch[kr * kSimdLane]);
            __m256d ck_l = _mm256_load_pd(&cl[kr * kSimdLane]);
            __m256d ph, pl;
            simd_dd::dd_mul(t1h, t1l, aih, ail, ph, pl);
            __m256d new_ckh, new_ckl;
            simd_dd::dd_add(ck_h, ck_l, ph, pl, new_ckh, new_ckl);
            _mm256_store_pd(&ch[kr * kSimdLane], new_ckh);
            _mm256_store_pd(&cl[kr * kSimdLane], new_ckl);
            /* temp2 += B[k,j] · A(i,k) */
            __m256d bk_h = _mm256_load_pd(&bh[kr * kSimdLane]);
            __m256d bk_l = _mm256_load_pd(&bl[kr * kSimdLane]);
            __m256d qh, ql;
            simd_dd::dd_mul(bk_h, bk_l, aih, ail, qh, ql);
            __m256d new_t2h, new_t2l;
            simd_dd::dd_add(t2h, t2l, qh, ql, new_t2h, new_t2l);
            t2h = new_t2h; t2l = new_t2l;
        }
        /* Diagonal cell: C[i,j] += temp1·A(i,i) + alpha·temp2 */
        const T aii = A_(i, i);
        __m256d aii_h = _mm256_set1_pd(aii.limbs[0]);
        __m256d aii_l = _mm256_set1_pd(aii.limbs[1]);
        __m256d diag_h, diag_l;
        simd_dd::dd_mul(t1h, t1l, aii_h, aii_l, diag_h, diag_l);
        __m256d at2h, at2l;
        simd_dd::dd_mul(alpha_h, alpha_l, t2h, t2l, at2h, at2l);
        __m256d sum_h, sum_l;
        simd_dd::dd_add(diag_h, diag_l, at2h, at2l, sum_h, sum_l);
        __m256d ci_h = _mm256_load_pd(&ch[ir * kSimdLane]);
        __m256d ci_l = _mm256_load_pd(&cl[ir * kSimdLane]);
        __m256d new_cih, new_cil;
        simd_dd::dd_add(ci_h, ci_l, sum_h, sum_l, new_cih, new_cil);
        _mm256_store_pd(&ch[ir * kSimdLane], new_cih);
        _mm256_store_pd(&cl[ir * kSimdLane], new_cil);
    };

    if (UPLO == 'L') for (int i = ic;          i < ic + ib;  ++i) body(i);
    else             for (int i = ic + ib - 1; i >= ic;      --i) body(i);
}

/* Drive the SIMD diag over a column range [0..N) in 4-column panels. */
inline void simd_symm_diag_L_panels(int ic, int ib, int N, T alpha,
                                    const T *a, int lda,
                                    const T *b, int ldb,
                                    T *c, int ldc, char UPLO)
{
    alignas(32) double bh[kMaxBlockM * kSimdLane];
    alignas(32) double bl[kMaxBlockM * kSimdLane];
    alignas(32) double ch[kMaxBlockM * kSimdLane];
    alignas(32) double cl[kMaxBlockM * kSimdLane];
    for (int j = 0; j < N; j += kSimdLane) {
        const int jc = (N - j < kSimdLane) ? (N - j) : kSimdLane;
        pack_4col(ib, ic, b, ldb, j, jc, bh, bl);
        pack_4col(ib, ic, c, ldc, j, jc, ch, cl);
        simd_symm_diag_L(ic, ib, alpha, a, lda, bh, bl, ch, cl, UPLO);
        unpack_4col(ib, ic, c, ldc, j, jc, ch, cl);
    }
}

#endif  /* MBLAS_SIMD_DD */

/* Scalar fallback diag for SIDE='L' (also used when SIMD off or
 * block too big for stack scratch). */
void symm_diag_add_L(int ic, int ib, int N, T alpha,
                     const T *a, int lda, const T *b, int ldb,
                     T *c, int ldc, char UPLO)
{
    for (int j = 0; j < N; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        const T *bj = b + static_cast<std::size_t>(j) * ldb;
        if (UPLO == 'L') {
            for (int i = ic; i < ic + ib; ++i) {
                const T temp1 = alpha * bj[i];
                T temp2 = zero_dd;
                for (int k = ic; k < i; ++k) {
                    cj[k]  = cj[k]  + temp1 * A_(i, k);
                    temp2  = temp2  + bj[k] * A_(i, k);
                }
                cj[i] = cj[i] + temp1 * A_(i, i) + alpha * temp2;
            }
        } else {
            for (int i = ic + ib - 1; i >= ic; --i) {
                const T temp1 = alpha * bj[i];
                T temp2 = zero_dd;
                for (int k = i + 1; k < ic + ib; ++k) {
                    cj[k]  = cj[k]  + temp1 * A_(i, k);
                    temp2  = temp2  + bj[k] * A_(i, k);
                }
                cj[i] = cj[i] + temp1 * A_(i, i) + alpha * temp2;
            }
        }
    }
}

#ifdef MBLAS_SIMD_DD

/* AoS→SoA: load 4 DD cells from a column into (hi, lo) 4-lane vectors.
 * Memory layout per cell: [hi, lo] back-to-back. */
inline void load_4cell_soa(const T *col, int ofs,
                           __m256d &h, __m256d &l)
{
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs]));
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs + 2]));
    __m256d lo = _mm256_unpacklo_pd(v0, v1);   /* [c0h, c2h, c1h, c3h] */
    __m256d hi = _mm256_unpackhi_pd(v0, v1);   /* [c0l, c2l, c1l, c3l] */
    h = _mm256_permute4x64_pd(lo, 0xD8);       /* [c0h, c1h, c2h, c3h] */
    l = _mm256_permute4x64_pd(hi, 0xD8);       /* [c0l, c1l, c2l, c3l] */
}

inline void store_4cell_soa(T *col, int ofs, __m256d h, __m256d l)
{
    __m256d lo = _mm256_unpacklo_pd(h, l);     /* [c0h, c0l, c2h, c2l] */
    __m256d hi = _mm256_unpackhi_pd(h, l);     /* [c1h, c1l, c3h, c3l] */
    __m256d v0 = _mm256_permute2f128_pd(lo, hi, 0x20);  /* [c0h, c0l, c1h, c1l] */
    __m256d v1 = _mm256_permute2f128_pd(lo, hi, 0x31);  /* [c2h, c2l, c3h, c3l] */
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs]),     v0);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs + 2]), v1);
}

/* SIDE='R' symmetric diag-block kernel, 4-row SIMD.
 *
 * For each i-block of 4 rows: hold C[i..i+3, j] in 2 ymm regs across
 * the k loop, accumulate α·A_eff(j,k)·B[i..i+3, k] for k ∈ [jc, jc+jb),
 * store back. A_eff uses the symmetric mirror via UPLO.
 * Tail (M % 4 != 0) falls back to scalar. */
inline void simd_symm_diag_R(int jc, int jb, int M, T alpha,
                             const T *a, int lda, const T *b, int ldb,
                             T *c, int ldc, char UPLO)
{
    const int M4 = M & ~3;

    for (int ib = 0; ib < M4; ib += 4) {
        for (int j = jc; j < jc + jb; ++j) {
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            __m256d ch, cl;
            load_4cell_soa(cj, ib, ch, cl);

            for (int k = jc; k < jc + jb; ++k) {
                T tval;
                if (k == j)                  tval = alpha * A_(j, j);
                else if (UPLO == 'L')        tval = (k < j) ? (alpha * A_(j, k))
                                                            : (alpha * A_(k, j));
                else /* UPLO == 'U' */       tval = (k < j) ? (alpha * A_(k, j))
                                                            : (alpha * A_(j, k));
                if (dd_iszero(tval)) continue;
                const __m256d th = _mm256_set1_pd(tval.limbs[0]);
                const __m256d tl = _mm256_set1_pd(tval.limbs[1]);
                const T *bk = b + static_cast<std::size_t>(k) * ldb;
                __m256d bh, bl;
                load_4cell_soa(bk, ib, bh, bl);
                __m256d ph, pl;
                simd_dd::dd_mul(th, tl, bh, bl, ph, pl);
                __m256d nh, nl;
                simd_dd::dd_add(ch, cl, ph, pl, nh, nl);
                ch = nh; cl = nl;
            }

            store_4cell_soa(cj, ib, ch, cl);
        }
    }

    /* Scalar tail rows (at most 3) */
    if (M4 < M) {
        for (int j = jc; j < jc + jb; ++j) {
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            {
                const T t = alpha * A_(j, j);
                for (int i = M4; i < M; ++i) cj[i] = cj[i] + t * B_(i, j);
            }
            if (UPLO == 'L') {
                for (int k = jc; k < j; ++k) {
                    const T t = alpha * A_(j, k);
                    if (!dd_iszero(t)) for (int i = M4; i < M; ++i) cj[i] = cj[i] + t * B_(i, k);
                }
                for (int k = j + 1; k < jc + jb; ++k) {
                    const T t = alpha * A_(k, j);
                    if (!dd_iszero(t)) for (int i = M4; i < M; ++i) cj[i] = cj[i] + t * B_(i, k);
                }
            } else {
                for (int k = jc; k < j; ++k) {
                    const T t = alpha * A_(k, j);
                    if (!dd_iszero(t)) for (int i = M4; i < M; ++i) cj[i] = cj[i] + t * B_(i, k);
                }
                for (int k = j + 1; k < jc + jb; ++k) {
                    const T t = alpha * A_(j, k);
                    if (!dd_iszero(t)) for (int i = M4; i < M; ++i) cj[i] = cj[i] + t * B_(i, k);
                }
            }
        }
    }
}

#endif  /* MBLAS_SIMD_DD */

void symm_diag_add_R(int jc, int jb, int M, T alpha,
                     const T *a, int lda, const T *b, int ldb,
                     T *c, int ldc, char UPLO)
{
    for (int j = jc; j < jc + jb; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        {
            const T t = alpha * A_(j, j);
            for (int i = 0; i < M; ++i) cj[i] = cj[i] + t * B_(i, j);
        }
        if (UPLO == 'L') {
            for (int k = jc; k < j; ++k) {
                const T t = alpha * A_(j, k);
                if (!dd_iszero(t)) for (int i = 0; i < M; ++i) cj[i] = cj[i] + t * B_(i, k);
            }
            for (int k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * A_(k, j);
                if (!dd_iszero(t)) for (int i = 0; i < M; ++i) cj[i] = cj[i] + t * B_(i, k);
            }
        } else {
            for (int k = jc; k < j; ++k) {
                const T t = alpha * A_(k, j);
                if (!dd_iszero(t)) for (int i = 0; i < M; ++i) cj[i] = cj[i] + t * B_(i, k);
            }
            for (int k = j + 1; k < jc + jb; ++k) {
                const T t = alpha * A_(j, k);
                if (!dd_iszero(t)) for (int i = 0; i < M; ++i) cj[i] = cj[i] + t * B_(i, k);
            }
        }
    }
}

inline void diag_R_dispatch(int jc, int jb, int M, T alpha,
                            const T *a, int lda, const T *b, int ldb,
                            T *c, int ldc, char UPLO)
{
#ifdef MBLAS_SIMD_DD
    simd_symm_diag_R(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
    return;
#else
    symm_diag_add_R(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
#endif
}

inline void diag_L_dispatch(int ic, int ib, int N, T alpha,
                            const T *a, int lda, const T *b, int ldb,
                            T *c, int ldc, char UPLO)
{
#ifdef MBLAS_SIMD_DD
    if (ib <= kMaxBlockM) {
        simd_symm_diag_L_panels(ic, ib, N, alpha, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }
#endif
    symm_diag_add_L(ic, ib, N, alpha, a, lda, b, ldb, c, ldc, UPLO);
}

} /* anonymous namespace */

int msymm_block_nb(void) {
    static int nb = 0;
    if (nb == 0) nb = env_int("MSYMM_NB", 64);
    return nb;
}

void msymm_scale_col(int j, int M, T beta, T *c, int ldc) {
    T *cj = c + static_cast<std::size_t>(j) * ldc;
    if (dd_iszero(beta)) for (int i = 0; i < M; ++i) cj[i] = zero_dd;
    else                 for (int i = 0; i < M; ++i) cj[i] = cj[i] * beta;
}

void msymm_block_L(int ic, int ib, int M, int N, char UPLO,
                   T alpha, T beta, const T *a, int lda, const T *b, int ldb,
                   T *c, int ldc)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    /* beta-scale this block's rows across all columns */
    for (int j = 0; j < N; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (dd_iszero(beta))      for (int i = ic; i < ic + ib; ++i) cj[i] = zero_dd;
        else if (!dd_isone(beta)) for (int i = ic; i < ic + ib; ++i) cj[i] = cj[i] * beta;
    }
    if (UPLO == 'L') {
        if (ic > 0) {
            mgemm_serial(NN, NN, &ib, &N, &ic, &alpha,
                         &A_(ic, 0), &lda, &B_(0, 0), &ldb,
                         &one_dd, &C_(ic, 0), &ldc, 1, 1);
        }
        diag_L_dispatch(ic, ib, N, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const int trailing = M - ic - ib;
        if (trailing > 0) {
            mgemm_serial(TN, NN, &ib, &N, &trailing, &alpha,
                         &A_(ic + ib, ic), &lda, &B_(ic + ib, 0), &ldb,
                         &one_dd, &C_(ic, 0), &ldc, 1, 1);
        }
    } else {
        if (ic > 0) {
            mgemm_serial(TN, NN, &ib, &N, &ic, &alpha,
                         &A_(0, ic), &lda, &B_(0, 0), &ldb,
                         &one_dd, &C_(ic, 0), &ldc, 1, 1);
        }
        diag_L_dispatch(ic, ib, N, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const int trailing = M - ic - ib;
        if (trailing > 0) {
            mgemm_serial(NN, NN, &ib, &N, &trailing, &alpha,
                         &A_(ic, ic + ib), &lda, &B_(ic + ib, 0), &ldb,
                         &one_dd, &C_(ic, 0), &ldc, 1, 1);
        }
    }
}

void msymm_block_R(int jc, int jb, int M, int N, char UPLO,
                   T alpha, T beta, const T *a, int lda, const T *b, int ldb,
                   T *c, int ldc)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    /* beta-scale this block's columns over all rows */
    for (int j = jc; j < jc + jb; ++j) {
        T *cj = c + static_cast<std::size_t>(j) * ldc;
        if (dd_iszero(beta))      for (int i = 0; i < M; ++i) cj[i] = zero_dd;
        else if (!dd_isone(beta)) for (int i = 0; i < M; ++i) cj[i] = cj[i] * beta;
    }
    if (UPLO == 'L') {
        if (jc > 0) {
            mgemm_serial(NN, TN, &M, &jb, &jc, &alpha,
                         &B_(0, 0), &ldb, &A_(jc, 0), &lda,
                         &one_dd, &C_(0, jc), &ldc, 1, 1);
        }
        diag_R_dispatch(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            mgemm_serial(NN, NN, &M, &jb, &trailing, &alpha,
                         &B_(0, jc + jb), &ldb, &A_(jc + jb, jc), &lda,
                         &one_dd, &C_(0, jc), &ldc, 1, 1);
        }
    } else {
        if (jc > 0) {
            mgemm_serial(NN, NN, &M, &jb, &jc, &alpha,
                         &B_(0, 0), &ldb, &A_(0, jc), &lda,
                         &one_dd, &C_(0, jc), &ldc, 1, 1);
        }
        diag_R_dispatch(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
        const int trailing = N - jc - jb;
        if (trailing > 0) {
            mgemm_serial(NN, TN, &M, &jb, &trailing, &alpha,
                         &B_(0, jc + jb), &ldb, &A_(jc, jc + jb), &lda,
                         &one_dd, &C_(0, jc), &ldc, 1, 1);
        }
    }
}

extern "C" void msymm_serial(
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

    if (dd_iszero(alpha)) {
        if (dd_isone(beta)) return;
        for (int j = 0; j < N; ++j) msymm_scale_col(j, M, beta, c, ldc);
        return;
    }

    const int nb = msymm_block_nb();
    if (SIDE == 'L') {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            msymm_block_L(ic, ib, M, N, UPLO, alpha, beta, a, lda, b, ldb, c, ldc);
        }
    } else {
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            msymm_block_R(jc, jb, M, N, UPLO, alpha, beta, a, lda, b, ldb, c, ldc);
        }
    }
}

#undef A_
#undef B_
#undef C_
