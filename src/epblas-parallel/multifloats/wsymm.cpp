/*
 * wsymm — multifloats complex (DD) symmetric matrix multiply.
 * NOT Hermitian (see whemm).
 *
 * Same blocked SIMD strategy as msymm: AVX2 4-wide pack of 4 columns
 * of B and C into SoA scratch (one ymm-pair per limb × {re, im}), run
 * the symmetric "read A_IK once, use twice" rank-1 kernel using
 * simd_dd::cdd_mul / cdd_add, unpack C back. SIDE='R' kept scalar.
 */
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"
#include <immintrin.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {

#define WSYMM_OMP_MIN 32

int env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    int v = std::atoi(s);
    return v > 0 ? v : dflt;
}
int g_nb = 0;
int symm_nb(void) { if (g_nb == 0) g_nb = env_int("WSYMM_NB", 64); return g_nb; }

inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
const T one_cdd { R{1.0, 0.0}, R{0.0, 0.0} };

inline bool cdd_iszero(const T &x) {
    return x.re.limbs[0] == 0.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}
inline bool cdd_isone(const T &x) {
    return x.re.limbs[0] == 1.0 && x.re.limbs[1] == 0.0
        && x.im.limbs[0] == 0.0 && x.im.limbs[1] == 0.0;
}

inline T cmul(T const &a, T const &b) {
    return T{ a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re };
}
inline T cadd(T const &a, T const &b) { return T{ a.re + b.re, a.im + b.im }; }

extern "C" void wgemm_(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    std::size_t transa_len, std::size_t transb_len);

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]
#define C_(i, j)  c[static_cast<std::size_t>(j) * ldc + (i)]

#ifdef MBLAS_SIMD_DD

constexpr int kSimdLane = simd_dd::NR;   /* 4 */
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
inline void broadcast_cdd(const T &v,
                          __m256d &rh, __m256d &rl,
                          __m256d &ih, __m256d &il)
{
    rh = _mm256_set1_pd(v.re.limbs[0]);
    rl = _mm256_set1_pd(v.re.limbs[1]);
    ih = _mm256_set1_pd(v.im.limbs[0]);
    il = _mm256_set1_pd(v.im.limbs[1]);
}

/* SIDE='L' complex-symmetric diag-block kernel, 4 column lanes.
 * cf. msymm.cpp simd_symm_diag_L — same control flow, cdd primitives. */
inline void simd_symm_diag_L(int ic, int ib, T alpha,
                             const T *a, int lda,
                             const double *brh, const double *brl,
                             const double *bih, const double *bil,
                             double *crh, double *crl,
                             double *cih, double *cil,
                             char UPLO)
{
    __m256d a_rh, a_rl, a_ih, a_il;
    broadcast_cdd(alpha, a_rh, a_rl, a_ih, a_il);

    auto body = [&](int i) {
        const int ir = i - ic;
        __m256d bi_rh = _mm256_load_pd(&brh[ir * kSimdLane]);
        __m256d bi_rl = _mm256_load_pd(&brl[ir * kSimdLane]);
        __m256d bi_ih = _mm256_load_pd(&bih[ir * kSimdLane]);
        __m256d bi_il = _mm256_load_pd(&bil[ir * kSimdLane]);
        __m256d t1rh, t1rl, t1ih, t1il;
        simd_dd::cdd_mul(a_rh, a_rl, a_ih, a_il,
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
            broadcast_cdd(A_(i, k), ak_rh, ak_rl, ak_ih, ak_il);
            /* C[k,j] += temp1 · A(i,k) */
            __m256d ck_rh = _mm256_load_pd(&crh[kr * kSimdLane]);
            __m256d ck_rl = _mm256_load_pd(&crl[kr * kSimdLane]);
            __m256d ck_ih = _mm256_load_pd(&cih[kr * kSimdLane]);
            __m256d ck_il = _mm256_load_pd(&cil[kr * kSimdLane]);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(t1rh, t1rl, t1ih, t1il,
                             ak_rh, ak_rl, ak_ih, ak_il,
                             prh, prl, pih, pil);
            __m256d ncrh, ncrl, ncih, ncil;
            simd_dd::cdd_add(ck_rh, ck_rl, ck_ih, ck_il,
                             prh, prl, pih, pil,
                             ncrh, ncrl, ncih, ncil);
            _mm256_store_pd(&crh[kr * kSimdLane], ncrh);
            _mm256_store_pd(&crl[kr * kSimdLane], ncrl);
            _mm256_store_pd(&cih[kr * kSimdLane], ncih);
            _mm256_store_pd(&cil[kr * kSimdLane], ncil);
            /* temp2 += B[k,j] · A(i,k) */
            __m256d bk_rh = _mm256_load_pd(&brh[kr * kSimdLane]);
            __m256d bk_rl = _mm256_load_pd(&brl[kr * kSimdLane]);
            __m256d bk_ih = _mm256_load_pd(&bih[kr * kSimdLane]);
            __m256d bk_il = _mm256_load_pd(&bil[kr * kSimdLane]);
            __m256d qrh, qrl, qih, qil;
            simd_dd::cdd_mul(bk_rh, bk_rl, bk_ih, bk_il,
                             ak_rh, ak_rl, ak_ih, ak_il,
                             qrh, qrl, qih, qil);
            __m256d nt2rh, nt2rl, nt2ih, nt2il;
            simd_dd::cdd_add(t2rh, t2rl, t2ih, t2il,
                             qrh, qrl, qih, qil,
                             nt2rh, nt2rl, nt2ih, nt2il);
            t2rh = nt2rh; t2rl = nt2rl; t2ih = nt2ih; t2il = nt2il;
        }
        /* C[i,j] += temp1 · A(i,i) + alpha · temp2 */
        __m256d aii_rh, aii_rl, aii_ih, aii_il;
        broadcast_cdd(A_(i, i), aii_rh, aii_rl, aii_ih, aii_il);
        __m256d d_rh, d_rl, d_ih, d_il;
        simd_dd::cdd_mul(t1rh, t1rl, t1ih, t1il,
                         aii_rh, aii_rl, aii_ih, aii_il,
                         d_rh, d_rl, d_ih, d_il);
        __m256d at_rh, at_rl, at_ih, at_il;
        simd_dd::cdd_mul(a_rh, a_rl, a_ih, a_il,
                         t2rh, t2rl, t2ih, t2il,
                         at_rh, at_rl, at_ih, at_il);
        __m256d sum_rh, sum_rl, sum_ih, sum_il;
        simd_dd::cdd_add(d_rh, d_rl, d_ih, d_il,
                         at_rh, at_rl, at_ih, at_il,
                         sum_rh, sum_rl, sum_ih, sum_il);
        __m256d ci_rh = _mm256_load_pd(&crh[ir * kSimdLane]);
        __m256d ci_rl = _mm256_load_pd(&crl[ir * kSimdLane]);
        __m256d ci_ih = _mm256_load_pd(&cih[ir * kSimdLane]);
        __m256d ci_il = _mm256_load_pd(&cil[ir * kSimdLane]);
        __m256d ncirh, ncirl, nciih, nciil;
        simd_dd::cdd_add(ci_rh, ci_rl, ci_ih, ci_il,
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

inline void simd_symm_diag_L_panels(int ic, int ib, int N, T alpha,
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
        simd_symm_diag_L(ic, ib, alpha, a, lda,
                         brh, brl, bih, bil,
                         crh, crl, cih, cil, UPLO);
        unpack_4col_cdd(ib, ic, c, ldc, j, jc, crh, crl, cih, cil);
    }
}

#endif  /* MBLAS_SIMD_DD */

void symm_diag_add_L(int ic, int ib, int N, T alpha,
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
                    cj[k]  = cadd(cj[k], cmul(temp1, A_(i, k)));
                    temp2  = cadd(temp2, cmul(bj[k], A_(i, k)));
                }
                cj[i] = cadd(cj[i], cadd(cmul(temp1, A_(i, i)), cmul(alpha, temp2)));
            }
        } else {
            for (int i = ic + ib - 1; i >= ic; --i) {
                const T temp1 = cmul(alpha, bj[i]);
                T temp2 = zero_cdd;
                for (int k = i + 1; k < ic + ib; ++k) {
                    cj[k]  = cadd(cj[k], cmul(temp1, A_(i, k)));
                    temp2  = cadd(temp2, cmul(bj[k], A_(i, k)));
                }
                cj[i] = cadd(cj[i], cadd(cmul(temp1, A_(i, i)), cmul(alpha, temp2)));
            }
        }
    }
}

#ifdef MBLAS_SIMD_DD

/* AoS→SoA for 4 complex DD cells from a column. Each cell is 4 doubles
 * in memory: [re.hi, re.lo, im.hi, im.lo]. */
inline void load_4cell_csoa(const T *col, int ofs,
                            __m256d &rh, __m256d &rl,
                            __m256d &ih, __m256d &il)
{
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs]));     /* c0 */
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs + 1])); /* c1 */
    __m256d v2 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs + 2])); /* c2 */
    __m256d v3 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs + 3])); /* c3 */
    /* 4×4 transpose */
    __m256d t0 = _mm256_unpacklo_pd(v0, v1);  /* [c0.rh, c1.rh, c0.ih, c1.ih] */
    __m256d t1 = _mm256_unpackhi_pd(v0, v1);  /* [c0.rl, c1.rl, c0.il, c1.il] */
    __m256d t2 = _mm256_unpacklo_pd(v2, v3);  /* [c2.rh, c3.rh, c2.ih, c3.ih] */
    __m256d t3 = _mm256_unpackhi_pd(v2, v3);  /* [c2.rl, c3.rl, c2.il, c3.il] */
    rh = _mm256_permute2f128_pd(t0, t2, 0x20);  /* [c0.rh, c1.rh, c2.rh, c3.rh] */
    rl = _mm256_permute2f128_pd(t1, t3, 0x20);
    ih = _mm256_permute2f128_pd(t0, t2, 0x31);  /* [c0.ih, c1.ih, c2.ih, c3.ih] */
    il = _mm256_permute2f128_pd(t1, t3, 0x31);
}

inline void store_4cell_csoa(T *col, int ofs,
                             __m256d rh, __m256d rl,
                             __m256d ih, __m256d il)
{
    /* Inverse 4×4 transpose */
    __m256d t0 = _mm256_unpacklo_pd(rh, rl);    /* [c0.rh, c0.rl, c2.rh, c2.rl] */
    __m256d t1 = _mm256_unpackhi_pd(rh, rl);    /* [c1.rh, c1.rl, c3.rh, c3.rl] */
    __m256d t2 = _mm256_unpacklo_pd(ih, il);    /* [c0.ih, c0.il, c2.ih, c2.il] */
    __m256d t3 = _mm256_unpackhi_pd(ih, il);    /* [c1.ih, c1.il, c3.ih, c3.il] */
    __m256d v0 = _mm256_permute2f128_pd(t0, t2, 0x20);  /* [c0.rh, c0.rl, c0.ih, c0.il] */
    __m256d v1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    __m256d v2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    __m256d v3 = _mm256_permute2f128_pd(t1, t3, 0x31);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs]),     v0);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs + 1]), v1);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs + 2]), v2);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs + 3]), v3);
}

/* SIDE='R' complex symmetric diag, 4-row SIMD. */
inline void simd_symm_diag_R(int jc, int jb, int M, T alpha,
                             const T *a, int lda, const T *b, int ldb,
                             T *c, int ldc, char UPLO)
{
    const int M4 = M & ~3;

    for (int ib = 0; ib < M4; ib += 4) {
        for (int j = jc; j < jc + jb; ++j) {
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            __m256d crh, crl, cih, cil;
            load_4cell_csoa(cj, ib, crh, crl, cih, cil);

            for (int k = jc; k < jc + jb; ++k) {
                T tval;
                if (k == j)                  tval = cmul(alpha, A_(j, j));
                else if (UPLO == 'L')        tval = (k < j) ? cmul(alpha, A_(j, k))
                                                            : cmul(alpha, A_(k, j));
                else                         tval = (k < j) ? cmul(alpha, A_(k, j))
                                                            : cmul(alpha, A_(j, k));
                if (cdd_iszero(tval)) continue;
                __m256d trh = _mm256_set1_pd(tval.re.limbs[0]);
                __m256d trl = _mm256_set1_pd(tval.re.limbs[1]);
                __m256d tih = _mm256_set1_pd(tval.im.limbs[0]);
                __m256d til = _mm256_set1_pd(tval.im.limbs[1]);
                const T *bk = b + static_cast<std::size_t>(k) * ldb;
                __m256d brh, brl, bih, bil;
                load_4cell_csoa(bk, ib, brh, brl, bih, bil);
                __m256d prh, prl, pih, pil;
                simd_dd::cdd_mul(trh, trl, tih, til,
                                 brh, brl, bih, bil,
                                 prh, prl, pih, pil);
                __m256d nrh, nrl, nih, nil_;
                simd_dd::cdd_add(crh, crl, cih, cil,
                                 prh, prl, pih, pil,
                                 nrh, nrl, nih, nil_);
                crh = nrh; crl = nrl; cih = nih; cil = nil_;
            }

            store_4cell_csoa(cj, ib, crh, crl, cih, cil);
        }
    }

    /* Scalar tail */
    if (M4 < M) {
        for (int j = jc; j < jc + jb; ++j) {
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            {
                const T t = cmul(alpha, A_(j, j));
                for (int i = M4; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, j)));
            }
            if (UPLO == 'L') {
                for (int k = jc; k < j; ++k) {
                    const T t = cmul(alpha, A_(j, k));
                    if (!cdd_iszero(t)) for (int i = M4; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
                }
                for (int k = j + 1; k < jc + jb; ++k) {
                    const T t = cmul(alpha, A_(k, j));
                    if (!cdd_iszero(t)) for (int i = M4; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
                }
            } else {
                for (int k = jc; k < j; ++k) {
                    const T t = cmul(alpha, A_(k, j));
                    if (!cdd_iszero(t)) for (int i = M4; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
                }
                for (int k = j + 1; k < jc + jb; ++k) {
                    const T t = cmul(alpha, A_(j, k));
                    if (!cdd_iszero(t)) for (int i = M4; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
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
            const T t = cmul(alpha, A_(j, j));
            for (int i = 0; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, j)));
        }
        if (UPLO == 'L') {
            for (int k = jc; k < j; ++k) {
                const T t = cmul(alpha, A_(j, k));
                if (!cdd_iszero(t)) for (int i = 0; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
            }
            for (int k = j + 1; k < jc + jb; ++k) {
                const T t = cmul(alpha, A_(k, j));
                if (!cdd_iszero(t)) for (int i = 0; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
            }
        } else {
            for (int k = jc; k < j; ++k) {
                const T t = cmul(alpha, A_(k, j));
                if (!cdd_iszero(t)) for (int i = 0; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
            }
            for (int k = j + 1; k < jc + jb; ++k) {
                const T t = cmul(alpha, A_(j, k));
                if (!cdd_iszero(t)) for (int i = 0; i < M; ++i) cj[i] = cadd(cj[i], cmul(t, B_(i, k)));
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
    diag_R_dispatch(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
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

extern "C" void wsymm_(
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

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};

    if (cdd_iszero(alpha)) {
        if (cdd_isone(beta)) return;
#ifdef _OPENMP
        const int axis = (SIDE == 'L') ? M : N;
        const bool use_omp = (axis >= WSYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (int j = 0; j < N; ++j) {
            T *cj = c + static_cast<std::size_t>(j) * ldc;
            if (cdd_iszero(beta)) for (int i = 0; i < M; ++i) cj[i] = zero_cdd;
            else                  for (int i = 0; i < M; ++i) cj[i] = cmul(cj[i], beta);
        }
        return;
    }

    const int nb = symm_nb();

    if (SIDE == 'L') {
#ifdef _OPENMP
        const bool use_omp = (M >= WSYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            for (int j = 0; j < N; ++j) {
                T *cj = c + static_cast<std::size_t>(j) * ldc;
                if (cdd_iszero(beta))      for (int i = ic; i < ic + ib; ++i) cj[i] = zero_cdd;
                else if (!cdd_isone(beta)) for (int i = ic; i < ic + ib; ++i) cj[i] = cmul(cj[i], beta);
            }
            if (UPLO == 'L') {
                if (ic > 0) {
                    wgemm_(NN, NN, &ib, &N, &ic, &alpha,
                           &A_(ic, 0), &lda, &B_(0, 0), &ldb,
                           &one_cdd, &C_(ic, 0), &ldc, 1, 1);
                }
                diag_L_dispatch(ic, ib, N, alpha, a, lda, b, ldb, c, ldc, UPLO);
                const int trailing = M - ic - ib;
                if (trailing > 0) {
                    wgemm_(TN, NN, &ib, &N, &trailing, &alpha,
                           &A_(ic + ib, ic), &lda, &B_(ic + ib, 0), &ldb,
                           &one_cdd, &C_(ic, 0), &ldc, 1, 1);
                }
            } else {
                if (ic > 0) {
                    wgemm_(TN, NN, &ib, &N, &ic, &alpha,
                           &A_(0, ic), &lda, &B_(0, 0), &ldb,
                           &one_cdd, &C_(ic, 0), &ldc, 1, 1);
                }
                diag_L_dispatch(ic, ib, N, alpha, a, lda, b, ldb, c, ldc, UPLO);
                const int trailing = M - ic - ib;
                if (trailing > 0) {
                    wgemm_(NN, NN, &ib, &N, &trailing, &alpha,
                           &A_(ic, ic + ib), &lda, &B_(ic + ib, 0), &ldb,
                           &one_cdd, &C_(ic, 0), &ldc, 1, 1);
                }
            }
        }
    } else {
#ifdef _OPENMP
        const bool use_omp = (N >= WSYMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(dynamic, 1)
#endif
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            for (int j = jc; j < jc + jb; ++j) {
                T *cj = c + static_cast<std::size_t>(j) * ldc;
                if (cdd_iszero(beta))      for (int i = 0; i < M; ++i) cj[i] = zero_cdd;
                else if (!cdd_isone(beta)) for (int i = 0; i < M; ++i) cj[i] = cmul(cj[i], beta);
            }
            if (UPLO == 'L') {
                if (jc > 0) {
                    wgemm_(NN, TN, &M, &jb, &jc, &alpha,
                           &B_(0, 0), &ldb, &A_(jc, 0), &lda,
                           &one_cdd, &C_(0, jc), &ldc, 1, 1);
                }
                diag_R_dispatch(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
                const int trailing = N - jc - jb;
                if (trailing > 0) {
                    wgemm_(NN, NN, &M, &jb, &trailing, &alpha,
                           &B_(0, jc + jb), &ldb, &A_(jc + jb, jc), &lda,
                           &one_cdd, &C_(0, jc), &ldc, 1, 1);
                }
            } else {
                if (jc > 0) {
                    wgemm_(NN, NN, &M, &jb, &jc, &alpha,
                           &B_(0, 0), &ldb, &A_(0, jc), &lda,
                           &one_cdd, &C_(0, jc), &ldc, 1, 1);
                }
                diag_R_dispatch(jc, jb, M, alpha, a, lda, b, ldb, c, ldc, UPLO);
                const int trailing = N - jc - jb;
                if (trailing > 0) {
                    wgemm_(NN, TN, &M, &jb, &trailing, &alpha,
                           &B_(0, jc + jb), &ldb, &A_(jc, jc + jb), &lda,
                           &one_cdd, &C_(0, jc), &ldc, 1, 1);
                }
            }
        }
    }
}

#undef A_
#undef B_
#undef C_
