/*
 * mtrmm — multifloats real (double-double) triangular multiply.
 *
 * B := alpha · op(A) · B (SIDE='L') or B := alpha · B · op(A) (SIDE='R'),
 * where op(A) ∈ {A, Aᵀ}. Scalar-core diagonal blocks + mgemm trailing
 * update with coarse-N parallelism. No SIMD diagonal kernel (in
 * mtrsm that path was 5–13% of total; the SIMD gemm trailing
 * dominates and we get that for free via the mgemm_ call).
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
using T = mf::float64x2;

namespace {

#define MTRMM_OMP_MIN 32

int env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    int v = std::atoi(s);
    return v > 0 ? v : dflt;
}

int g_nb_trmm = 0;
int trmm_nb(void) {
    if (g_nb_trmm == 0) g_nb_trmm = env_int("MTRMM_NB", 64);
    return g_nb_trmm;
}

inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

const T zero_dd{0.0, 0.0};
const T one_dd {1.0, 0.0};

inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (T x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }

extern "C" void mgemm_(
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

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

#ifdef MBLAS_SIMD_DD

constexpr int kSimdLane = simd_dd::NR;
constexpr int kMaxBlockM = 256;

inline void pack_B_4col(int M, const T *b, int ldb, int j_start, int j_count,
                        double *bh, double *bl)
{
    for (int j = 0; j < j_count; ++j) {
        const T *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (int i = 0; i < M; ++i) {
            bh[i * kSimdLane + j] = col[i].limbs[0];
            bl[i * kSimdLane + j] = col[i].limbs[1];
        }
    }
    for (int j = j_count; j < kSimdLane; ++j)
        for (int i = 0; i < M; ++i) {
            bh[i * kSimdLane + j] = 0.0;
            bl[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_B_4col(int M, T *b, int ldb, int j_start, int j_count,
                          const double *bh, const double *bl)
{
    for (int j = 0; j < j_count; ++j) {
        T *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (int i = 0; i < M; ++i) {
            col[i].limbs[0] = bh[i * kSimdLane + j];
            col[i].limbs[1] = bl[i * kSimdLane + j];
        }
    }
}

/* SIMD LLN: for k = M-1..0: temp = α·B(k); for i>k: B(i) += temp·A(i,k);
 * if nounit: temp *= A(k,k); B(k) = temp. */
inline void simd_trmm_lln(int M, const T *a, int lda, T alpha, int nounit,
                          double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (int k = M - 1; k >= 0; --k) {
        __m256d bkh = _mm256_load_pd(&bh[k * kSimdLane]);
        __m256d bkl = _mm256_load_pd(&bl[k * kSimdLane]);
        __m256d th, tl;
        simd_dd::dd_mul(ah, al, bkh, bkl, th, tl);
        for (int i = M - 1; i > k; --i) {
            const T aik = A_(i, k);
            __m256d aih = _mm256_set1_pd(aik.limbs[0]);
            __m256d ail = _mm256_set1_pd(aik.limbs[1]);
            __m256d ph, pl;
            simd_dd::dd_mul(th, tl, aih, ail, ph, pl);
            __m256d bih = _mm256_load_pd(&bh[i * kSimdLane]);
            __m256d bil = _mm256_load_pd(&bl[i * kSimdLane]);
            __m256d nh, nl;
            simd_dd::dd_add(bih, bil, ph, pl, nh, nl);
            _mm256_store_pd(&bh[i * kSimdLane], nh);
            _mm256_store_pd(&bl[i * kSimdLane], nl);
        }
        if (nounit) {
            const T akk = A_(k, k);
            __m256d akh = _mm256_set1_pd(akk.limbs[0]);
            __m256d akl = _mm256_set1_pd(akk.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(th, tl, akh, akl, nh, nl);
            th = nh; tl = nl;
        }
        _mm256_store_pd(&bh[k * kSimdLane], th);
        _mm256_store_pd(&bl[k * kSimdLane], tl);
    }
}

/* SIMD LUN: for k = 0..M: temp = α·B(k); for i<k: B(i) += temp·A(i,k);
 * if nounit: temp *= A(k,k); B(k) = temp. */
inline void simd_trmm_lun(int M, const T *a, int lda, T alpha, int nounit,
                          double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (int k = 0; k < M; ++k) {
        __m256d bkh = _mm256_load_pd(&bh[k * kSimdLane]);
        __m256d bkl = _mm256_load_pd(&bl[k * kSimdLane]);
        __m256d th, tl;
        simd_dd::dd_mul(ah, al, bkh, bkl, th, tl);
        for (int i = 0; i < k; ++i) {
            const T aik = A_(i, k);
            __m256d aih = _mm256_set1_pd(aik.limbs[0]);
            __m256d ail = _mm256_set1_pd(aik.limbs[1]);
            __m256d ph, pl;
            simd_dd::dd_mul(th, tl, aih, ail, ph, pl);
            __m256d bih = _mm256_load_pd(&bh[i * kSimdLane]);
            __m256d bil = _mm256_load_pd(&bl[i * kSimdLane]);
            __m256d nh, nl;
            simd_dd::dd_add(bih, bil, ph, pl, nh, nl);
            _mm256_store_pd(&bh[i * kSimdLane], nh);
            _mm256_store_pd(&bl[i * kSimdLane], nl);
        }
        if (nounit) {
            const T akk = A_(k, k);
            __m256d akh = _mm256_set1_pd(akk.limbs[0]);
            __m256d akl = _mm256_set1_pd(akk.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(th, tl, akh, akl, nh, nl);
            th = nh; tl = nl;
        }
        _mm256_store_pd(&bh[k * kSimdLane], th);
        _mm256_store_pd(&bl[k * kSimdLane], tl);
    }
}

/* SIMD LLT: for i = 0..M: t = B(i); if nounit: t *= A(i,i);
 * for k>i: t += A(k,i)·B(k); B(i) = alpha·t. */
inline void simd_trmm_llt(int M, const T *a, int lda, T alpha, int nounit,
                          double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (int i = 0; i < M; ++i) {
        __m256d th = _mm256_load_pd(&bh[i * kSimdLane]);
        __m256d tl = _mm256_load_pd(&bl[i * kSimdLane]);
        if (nounit) {
            const T aii = A_(i, i);
            __m256d aih = _mm256_set1_pd(aii.limbs[0]);
            __m256d ail = _mm256_set1_pd(aii.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(th, tl, aih, ail, nh, nl);
            th = nh; tl = nl;
        }
        for (int k = i + 1; k < M; ++k) {
            const T aki = A_(k, i);
            __m256d akh = _mm256_set1_pd(aki.limbs[0]);
            __m256d akl = _mm256_set1_pd(aki.limbs[1]);
            __m256d bkh = _mm256_load_pd(&bh[k * kSimdLane]);
            __m256d bkl = _mm256_load_pd(&bl[k * kSimdLane]);
            __m256d ph, pl;
            simd_dd::dd_mul(akh, akl, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(th, tl, ph, pl, nh, nl);
            th = nh; tl = nl;
        }
        __m256d nh, nl;
        simd_dd::dd_mul(ah, al, th, tl, nh, nl);
        _mm256_store_pd(&bh[i * kSimdLane], nh);
        _mm256_store_pd(&bl[i * kSimdLane], nl);
    }
}

/* SIMD LUT: for i = M-1..0: t = B(i); if nounit: t *= A(i,i);
 * for k<i: t += A(k,i)·B(k); B(i) = alpha·t. */
inline void simd_trmm_lut(int M, const T *a, int lda, T alpha, int nounit,
                          double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (int i = M - 1; i >= 0; --i) {
        __m256d th = _mm256_load_pd(&bh[i * kSimdLane]);
        __m256d tl = _mm256_load_pd(&bl[i * kSimdLane]);
        if (nounit) {
            const T aii = A_(i, i);
            __m256d aih = _mm256_set1_pd(aii.limbs[0]);
            __m256d ail = _mm256_set1_pd(aii.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(th, tl, aih, ail, nh, nl);
            th = nh; tl = nl;
        }
        for (int k = 0; k < i; ++k) {
            const T aki = A_(k, i);
            __m256d akh = _mm256_set1_pd(aki.limbs[0]);
            __m256d akl = _mm256_set1_pd(aki.limbs[1]);
            __m256d bkh = _mm256_load_pd(&bh[k * kSimdLane]);
            __m256d bkl = _mm256_load_pd(&bl[k * kSimdLane]);
            __m256d ph, pl;
            simd_dd::dd_mul(akh, akl, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(th, tl, ph, pl, nh, nl);
            th = nh; tl = nl;
        }
        __m256d nh, nl;
        simd_dd::dd_mul(ah, al, th, tl, nh, nl);
        _mm256_store_pd(&bh[i * kSimdLane], nh);
        _mm256_store_pd(&bl[i * kSimdLane], nl);
    }
}

enum trmm_simd_op { SLLN, SLUN, SLLT, SLUT };

inline void mtrmm_simd_diag(trmm_simd_op op, int j_start, int j_end,
                            int M, T alpha,
                            const T *a, int lda, T *b, int ldb, int nounit)
{
    alignas(32) double bh[kMaxBlockM * kSimdLane];
    alignas(32) double bl[kMaxBlockM * kSimdLane];
    for (int j = j_start; j < j_end; j += kSimdLane) {
        const int jc = (j_end - j < kSimdLane) ? (j_end - j) : kSimdLane;
        pack_B_4col(M, b, ldb, j, jc, bh, bl);
        switch (op) {
        case SLLN: simd_trmm_lln(M, a, lda, alpha, nounit, bh, bl); break;
        case SLUN: simd_trmm_lun(M, a, lda, alpha, nounit, bh, bl); break;
        case SLLT: simd_trmm_llt(M, a, lda, alpha, nounit, bh, bl); break;
        case SLUT: simd_trmm_lut(M, a, lda, alpha, nounit, bh, bl); break;
        }
        unpack_B_4col(M, b, ldb, j, jc, bh, bl);
    }
}

#endif  /* MBLAS_SIMD_DD */

inline void mtrmm_lln_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = M - 1; k >= 0; --k) {
            if (!dd_iszero(B_(k, j))) {
                T temp = alpha * B_(k, j);
                for (int i = M - 1; i > k; --i)
                    B_(i, j) = B_(i, j) + temp * A_(i, k);
                if (nounit) temp = temp * A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

inline void mtrmm_lun_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = 0; k < M; ++k) {
            if (!dd_iszero(B_(k, j))) {
                T temp = alpha * B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) = B_(i, j) + temp * A_(i, k);
                if (nounit) temp = temp * A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

inline void mtrmm_llt_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = B_(i, j);
            if (nounit) t = t * A_(i, i);
            for (int k = i + 1; k < M; ++k) t = t + A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

inline void mtrmm_lut_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t = t * A_(i, i);
            for (int k = 0; k < i; ++k) t = t + A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

#ifdef MBLAS_SIMD_DD

/* Forward decls for scalar tails (defined below). */
inline void mtrmm_rln_core(int, int, int, T, const T*, int, T*, int, int);
inline void mtrmm_run_core(int, int, int, T, const T*, int, T*, int, int);
inline void mtrmm_rlt_core(int, int, int, T, const T*, int, T*, int, int);
inline void mtrmm_rut_core(int, int, int, T, const T*, int, T*, int, int);

inline void load_4cell_soa(const T *col, int ofs, __m256d &h, __m256d &l)
{
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs]));
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs + 2]));
    __m256d lo = _mm256_unpacklo_pd(v0, v1);
    __m256d hi = _mm256_unpackhi_pd(v0, v1);
    h = _mm256_permute4x64_pd(lo, 0xD8);
    l = _mm256_permute4x64_pd(hi, 0xD8);
}

inline void store_4cell_soa(T *col, int ofs, __m256d h, __m256d l)
{
    __m256d lo = _mm256_unpacklo_pd(h, l);
    __m256d hi = _mm256_unpackhi_pd(h, l);
    __m256d v0 = _mm256_permute2f128_pd(lo, hi, 0x20);
    __m256d v1 = _mm256_permute2f128_pd(lo, hi, 0x31);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs]),     v0);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs + 2]), v1);
}

/* R-side trmm SIMD: 4-row chunks of B[i_start:i_end, 0..N).
 * For each 4-row chunk, run the trmm column-by-column with broadcast A. */
inline void simd_trmm_r4_rln(int ib, int N, T alpha, const T *a, int lda,
                             T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        T *bj = b + static_cast<std::size_t>(j) * ldb;
        T t = alpha;
        if (nounit) t = t * A_(j, j);
        __m256d bjh, bjl;
        load_4cell_soa(bj, ib, bjh, bjl);
        if (!dd_isone(t)) {
            __m256d th = _mm256_set1_pd(t.limbs[0]);
            __m256d tl = _mm256_set1_pd(t.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(bjh, bjl, th, tl, nh, nl);
            bjh = nh; bjl = nl;
        }
        for (int k = j + 1; k < N; ++k) {
            const T akj_v = A_(k, j);
            if (dd_iszero(akj_v)) continue;
            const T akj = alpha * akj_v;
            __m256d akh = _mm256_set1_pd(akj.limbs[0]);
            __m256d akl = _mm256_set1_pd(akj.limbs[1]);
            const T *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkh, bkl;
            load_4cell_soa(bk, ib, bkh, bkl);
            __m256d ph, pl;
            simd_dd::dd_mul(akh, akl, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(bjh, bjl, ph, pl, nh, nl);
            bjh = nh; bjl = nl;
        }
        store_4cell_soa(bj, ib, bjh, bjl);
    }
}

inline void simd_trmm_r4_run(int ib, int N, T alpha, const T *a, int lda,
                             T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        T *bj = b + static_cast<std::size_t>(j) * ldb;
        T t = alpha;
        if (nounit) t = t * A_(j, j);
        __m256d bjh, bjl;
        load_4cell_soa(bj, ib, bjh, bjl);
        if (!dd_isone(t)) {
            __m256d th = _mm256_set1_pd(t.limbs[0]);
            __m256d tl = _mm256_set1_pd(t.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(bjh, bjl, th, tl, nh, nl);
            bjh = nh; bjl = nl;
        }
        for (int k = 0; k < j; ++k) {
            const T akj_v = A_(k, j);
            if (dd_iszero(akj_v)) continue;
            const T akj = alpha * akj_v;
            __m256d akh = _mm256_set1_pd(akj.limbs[0]);
            __m256d akl = _mm256_set1_pd(akj.limbs[1]);
            const T *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkh, bkl;
            load_4cell_soa(bk, ib, bkh, bkl);
            __m256d ph, pl;
            simd_dd::dd_mul(akh, akl, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(bjh, bjl, ph, pl, nh, nl);
            bjh = nh; bjl = nl;
        }
        store_4cell_soa(bj, ib, bjh, bjl);
    }
}

/* For RLT/RUT, the column-update order goes k=N-1..0 (RLT) or 0..N-1 (RUT);
 * for each k we update columns j != k by adding α·A(j,k)·B(:,k), then scale B(:,k)
 * by α (×A(k,k) if nounit). Need to read B(:,k) before scaling. */
inline void simd_trmm_r4_rlt(int ib, int N, T alpha, const T *a, int lda,
                             T *b, int ldb, int nounit)
{
    for (int k = N - 1; k >= 0; --k) {
        const T *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkh, bkl;
        load_4cell_soa(bk, ib, bkh, bkl);  /* original B(:,k) before scaling */
        for (int j = k + 1; j < N; ++j) {
            const T ajk_v = A_(j, k);
            if (dd_iszero(ajk_v)) continue;
            const T ajk = alpha * ajk_v;
            __m256d ah = _mm256_set1_pd(ajk.limbs[0]);
            __m256d al = _mm256_set1_pd(ajk.limbs[1]);
            T *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjh, bjl;
            load_4cell_soa(bj, ib, bjh, bjl);
            __m256d ph, pl;
            simd_dd::dd_mul(ah, al, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(bjh, bjl, ph, pl, nh, nl);
            store_4cell_soa(bj, ib, nh, nl);
        }
        T t = alpha;
        if (nounit) t = t * A_(k, k);
        if (!dd_isone(t)) {
            __m256d th = _mm256_set1_pd(t.limbs[0]);
            __m256d tl = _mm256_set1_pd(t.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(bkh, bkl, th, tl, nh, nl);
            store_4cell_soa(const_cast<T*>(bk), ib, nh, nl);
        }
    }
}

inline void simd_trmm_r4_rut(int ib, int N, T alpha, const T *a, int lda,
                             T *b, int ldb, int nounit)
{
    for (int k = 0; k < N; ++k) {
        const T *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkh, bkl;
        load_4cell_soa(bk, ib, bkh, bkl);
        for (int j = 0; j < k; ++j) {
            const T ajk_v = A_(j, k);
            if (dd_iszero(ajk_v)) continue;
            const T ajk = alpha * ajk_v;
            __m256d ah = _mm256_set1_pd(ajk.limbs[0]);
            __m256d al = _mm256_set1_pd(ajk.limbs[1]);
            T *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjh, bjl;
            load_4cell_soa(bj, ib, bjh, bjl);
            __m256d ph, pl;
            simd_dd::dd_mul(ah, al, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(bjh, bjl, ph, pl, nh, nl);
            store_4cell_soa(bj, ib, nh, nl);
        }
        T t = alpha;
        if (nounit) t = t * A_(k, k);
        if (!dd_isone(t)) {
            __m256d th = _mm256_set1_pd(t.limbs[0]);
            __m256d tl = _mm256_set1_pd(t.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(bkh, bkl, th, tl, nh, nl);
            store_4cell_soa(const_cast<T*>(bk), ib, nh, nl);
        }
    }
}

enum trmm_r_op { RLN_OP, RUN_OP, RLT_OP, RUT_OP };

inline void mtrmm_simd_diag_R(trmm_r_op op, int i_start, int i_end, int N, T alpha,
                              const T *a, int lda, T *b, int ldb, int nounit)
{
    const int i4_end = i_start + ((i_end - i_start) & ~3);
    for (int ib = i_start; ib < i4_end; ib += 4) {
        switch (op) {
        case RLN_OP: simd_trmm_r4_rln(ib, N, alpha, a, lda, b, ldb, nounit); break;
        case RUN_OP: simd_trmm_r4_run(ib, N, alpha, a, lda, b, ldb, nounit); break;
        case RLT_OP: simd_trmm_r4_rlt(ib, N, alpha, a, lda, b, ldb, nounit); break;
        case RUT_OP: simd_trmm_r4_rut(ib, N, alpha, a, lda, b, ldb, nounit); break;
        }
    }
    /* Scalar tail rows */
    if (i4_end < i_end) {
        switch (op) {
        case RLN_OP: mtrmm_rln_core(i4_end, i_end, N, alpha, a, lda, b, ldb, nounit); break;
        case RUN_OP: mtrmm_run_core(i4_end, i_end, N, alpha, a, lda, b, ldb, nounit); break;
        case RLT_OP: mtrmm_rlt_core(i4_end, i_end, N, alpha, a, lda, b, ldb, nounit); break;
        case RUT_OP: mtrmm_rut_core(i4_end, i_end, N, alpha, a, lda, b, ldb, nounit); break;
        }
    }
}

#endif  /* MBLAS_SIMD_DD */

inline void mtrmm_rln_core(int i_start, int i_end, int N, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        T t = alpha;
        if (nounit) t = t * A_(j, j);
        if (!dd_isone(t))
            for (int i = i_start; i < i_end; ++i) B_(i, j) = B_(i, j) * t;
        for (int k = j + 1; k < N; ++k) {
            if (!dd_iszero(A_(k, j))) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) = B_(i, j) + akj * B_(i, k);
            }
        }
    }
}

inline void mtrmm_run_core(int i_start, int i_end, int N, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t = t * A_(j, j);
        if (!dd_isone(t))
            for (int i = i_start; i < i_end; ++i) B_(i, j) = B_(i, j) * t;
        for (int k = 0; k < j; ++k) {
            if (!dd_iszero(A_(k, j))) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) = B_(i, j) + akj * B_(i, k);
            }
        }
    }
}

inline void mtrmm_rlt_core(int i_start, int i_end, int N, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = N - 1; k >= 0; --k) {
        for (int j = k + 1; j < N; ++j) {
            if (!dd_iszero(A_(j, k))) {
                const T ajk = alpha * A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) = B_(i, j) + ajk * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t = t * A_(k, k);
        if (!dd_isone(t))
            for (int i = i_start; i < i_end; ++i) B_(i, k) = B_(i, k) * t;
    }
}

inline void mtrmm_rut_core(int i_start, int i_end, int N, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < k; ++j) {
            if (!dd_iszero(A_(j, k))) {
                const T ajk = alpha * A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) = B_(i, j) + ajk * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t = t * A_(k, k);
        if (!dd_isone(t))
            for (int i = i_start; i < i_end; ++i) B_(i, k) = B_(i, k) * t;
    }
}

/* ── Standalone OMP wrappers (small-M fallback). */

#ifdef _OPENMP
#define MTRMM_OMP_WRAP_L(name, core)                                       \
    void name(int M, int N, T alpha,                                        \
              const T *a, int lda, T *b, int ldb, int nounit) {             \
        if (N >= MTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = static_cast<int>(static_cast<long long>(N) * tid / nt);            \
                int je  = static_cast<int>(static_cast<long long>(N) * (tid + 1) / nt);      \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }            \
    }
#define MTRMM_OMP_WRAP_R(name, core)                                       \
    void name(int M, int N, T alpha,                                        \
              const T *a, int lda, T *b, int ldb, int nounit) {             \
        if (M >= MTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int is  = static_cast<int>(static_cast<long long>(M) * tid / nt);            \
                int ie  = static_cast<int>(static_cast<long long>(M) * (tid + 1) / nt);      \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }            \
    }
#else
#define MTRMM_OMP_WRAP_L(name, core)                                       \
    void name(int M, int N, T alpha,                                        \
              const T *a, int lda, T *b, int ldb, int nounit) {             \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#define MTRMM_OMP_WRAP_R(name, core)                                       \
    void name(int M, int N, T alpha,                                        \
              const T *a, int lda, T *b, int ldb, int nounit) {             \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                       \
    }
#endif

MTRMM_OMP_WRAP_L(mtrmm_lln, mtrmm_lln_core)
MTRMM_OMP_WRAP_L(mtrmm_lun, mtrmm_lun_core)
MTRMM_OMP_WRAP_L(mtrmm_llt, mtrmm_llt_core)
MTRMM_OMP_WRAP_L(mtrmm_lut, mtrmm_lut_core)
MTRMM_OMP_WRAP_R(mtrmm_rln, mtrmm_rln_core)
MTRMM_OMP_WRAP_R(mtrmm_run, mtrmm_run_core)
MTRMM_OMP_WRAP_R(mtrmm_rlt, mtrmm_rlt_core)
MTRMM_OMP_WRAP_R(mtrmm_rut, mtrmm_rut_core)

/* ── Blocked SIDE='L' ────────────────────────────────────────────── */

enum trmm_variant_L { LLN, LUN, LLT, LUT };

void blocked_chunk_L(trmm_variant_L V, int j_start, int j_end,
                     int M, int nb, T alpha,
                     const T *a, int lda, T *b, int ldb, int nounit)
{
    const int my_N = j_end - j_start;
    if (my_N <= 0) return;

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    T *B_chunk = &B_(0, j_start);

    if (V == LLN) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                mtrmm_simd_diag(SLLN, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            mtrmm_lln_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                mgemm_(NN, NN, &ib, &my_N, &ic, &alpha,
                       &A_(ic, 0), &lda,
                       B_chunk, &ldb, &one_dd,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    } else if (V == LUN) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                mtrmm_simd_diag(SLUN, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            mtrmm_lun_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int j0 = ic + ib;
                mgemm_(NN, NN, &ib, &my_N, &trailing, &alpha,
                       &A_(ic, j0), &lda,
                       &B_chunk[j0], &ldb, &one_dd,
                       &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else if (V == LLT) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                mtrmm_simd_diag(SLLT, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            mtrmm_llt_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int i0 = ic + ib;
                mgemm_(TN, NN, &ib, &my_N, &trailing, &alpha,
                       &A_(i0, ic), &lda,
                       &B_chunk[i0], &ldb, &one_dd,
                       &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else { /* LUT */
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                mtrmm_simd_diag(SLUT, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            mtrmm_lut_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                mgemm_(TN, NN, &ib, &my_N, &ic, &alpha,
                       &A_(0, ic), &lda,
                       B_chunk, &ldb, &one_dd,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    }
}

void blocked_dispatch_L(trmm_variant_L V, int M, int N, T alpha,
                        const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = trmm_nb();
#ifdef _OPENMP
    if (N >= MTRMM_OMP_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int js  = static_cast<int>(static_cast<long long>(N) * tid / nt);
            int je  = static_cast<int>(static_cast<long long>(N) * (tid + 1) / nt);
            blocked_chunk_L(V, js, je, M, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    blocked_chunk_L(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
}

/* ── Blocked SIDE='R' ────────────────────────────────────────────── */

enum trmm_variant_R { RLN, RUN, RLT, RUT };

void blocked_chunk_R(trmm_variant_R V, int i_start, int i_end,
                     int N, int nb, T alpha,
                     const T *a, int lda, T *b, int ldb, int nounit)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const int my_M = i_end - i_start;
    if (my_M <= 0) return;
    T *B_chunk = &B_(i_start, 0);

    if (V == RLN) {
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            mtrmm_simd_diag_R(RLN_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            mtrmm_rln_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#endif
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                mgemm_(NN, NN, &my_M, &jb, &trailing, &alpha,
                       &B_chunk[static_cast<std::size_t>(k0) * ldb], &ldb,
                       &A_(k0, jc), &lda, &one_dd,
                       &B_chunk[static_cast<std::size_t>(jc) * ldb], &ldb, 1, 1);
            }
        }
    } else if (V == RUN) {
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            mtrmm_simd_diag_R(RUN_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            mtrmm_run_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#endif
            if (jc > 0) {
                mgemm_(NN, NN, &my_M, &jb, &jc, &alpha,
                       B_chunk, &ldb,
                       &A_(0, jc), &lda, &one_dd,
                       &B_chunk[static_cast<std::size_t>(jc) * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else if (V == RLT) {
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            mtrmm_simd_diag_R(RLT_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            mtrmm_rlt_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#endif
            if (jc > 0) {
                mgemm_(NN, TN, &my_M, &jb, &jc, &alpha,
                       B_chunk, &ldb,
                       &A_(jc, 0), &lda, &one_dd,
                       &B_chunk[static_cast<std::size_t>(jc) * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else { /* RUT */
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            mtrmm_simd_diag_R(RUT_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            mtrmm_rut_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#endif
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                mgemm_(NN, TN, &my_M, &jb, &trailing, &alpha,
                       &B_chunk[static_cast<std::size_t>(k0) * ldb], &ldb,
                       &A_(jc, k0), &lda, &one_dd,
                       &B_chunk[static_cast<std::size_t>(jc) * ldb], &ldb, 1, 1);
            }
        }
    }
}

void blocked_dispatch_R(trmm_variant_R V, int M, int N, T alpha,
                        const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = trmm_nb();
#ifdef _OPENMP
    if (M >= MTRMM_OMP_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int is  = static_cast<int>(static_cast<long long>(M) * tid / nt);
            int ie  = static_cast<int>(static_cast<long long>(M) * (tid + 1) / nt);
            blocked_chunk_R(V, is, ie, N, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    blocked_chunk_R(V, 0, M, N, nb, alpha, a, lda, b, ldb, nounit);
}

} /* anonymous namespace */

extern "C" void mtrmm_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    std::size_t side_len, std::size_t uplo_len, std::size_t transa_len, std::size_t diag_len)
{
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;
    const char SIDE = up(side);
    const char UPLO = up(uplo);
    char TR         = up(transa);
    if (TR == 'C') TR = 'T';   /* real DD: conj-trans ≡ trans */
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (dd_iszero(alpha)) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = zero_dd;
        return;
    }

    const int nb = trmm_nb();

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * nb);
#ifdef MBLAS_SIMD_DD
        /* For small M (single-block regime), route the unblocked path
         * through the SIMD diag too — same kernel, full N column range. */
        const int simd_unblocked = (!use_blocked) && (M <= kMaxBlockM);
        if (simd_unblocked) {
            trmm_simd_op op = SLLN;
            if (TR == 'N') op = (UPLO == 'L') ? SLLN : SLUN;
            else           op = (UPLO == 'L') ? SLLT : SLUT;
            mtrmm_simd_diag(op, 0, N, M, alpha, a, lda, b, ldb, nounit);
        } else
#endif
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_L(LLN, M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrmm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_L(LUN, M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrmm_lun(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_L(LLT, M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrmm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_L(LUT, M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrmm_lut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    } else {
        const int use_blocked = (N >= 2 * nb);
#ifdef MBLAS_SIMD_DD
        if (!use_blocked) {
            trmm_r_op op = RLN_OP;
            if (TR == 'N') op = (UPLO == 'L') ? RLN_OP : RUN_OP;
            else           op = (UPLO == 'L') ? RLT_OP : RUT_OP;
            mtrmm_simd_diag_R(op, 0, M, N, alpha, a, lda, b, ldb, nounit);
            return;
        }
#endif
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_R(RLN, M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrmm_rln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_R(RUN, M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrmm_run(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_R(RLT, M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrmm_rlt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_R(RUT, M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrmm_rut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    }
}

#undef A_
#undef B_
