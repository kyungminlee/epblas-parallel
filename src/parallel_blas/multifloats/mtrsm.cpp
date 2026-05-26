/*
 * mtrsm — multifloats real (double-double) triangular solve.
 *
 * Solves one of:
 *   op(A) · X = α · B          (SIDE='L')
 *   X · op(A) = α · B          (SIDE='R')
 *
 * where op(A) ∈ {A, Aᵀ, Aᴴ}; for real DD types Aᴴ ≡ Aᵀ. A is M×M
 * (or N×N) triangular (upper or lower; optionally unit-diagonal).
 * B is overwritten with the solution X.
 *
 * Implementation stages (this file lands stages 1+2 together, with
 * stage 3 — SIMD diagonal kernel — in a follow-up):
 *
 *   1. Scalar unblocked, all 16 distinct variants. C++ scalar code
 *      with multifloats overloaded operators inlined into the hot
 *      loop — already beats migrated heavily because the migrated
 *      mtrsm goes through gfortran elementals (one call per DD op).
 *
 *   2. Blocked SIDE='L' (4 variants) with mgemm trailing update +
 *      coarse-N parallelism (one outer omp parallel, threads
 *      partition columns of B). mgemm inside each thread runs with
 *      a 1-thread inner team due to OMP_NESTED=false default — so
 *      we get SIMD GEMM trailing-update for free.
 *
 *   3. SIMD 4-wide AVX2 diagonal kernel — separate follow-up.
 *
 * Fortran ABI: extern "C" symbol `mtrsm_`. Character args have
 * hidden trailing size_t lengths.
 */

#include <cstddef>
#include <cstdlib>
#include <cctype>
#include <multifloats.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"   /* dd_mul, dd_add, dd_neg primitives */
#include <immintrin.h>
#include "../common/blas_omp.h"
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {

#define MTRSM_OMP_N_MIN 32

int env_int(const char *name, int dflt) {
    const char *s = std::getenv(name);
    if (!s || !*s) return dflt;
    int v = std::atoi(s);
    return v > 0 ? v : dflt;
}

int g_nb_trsm = 0;
int trsm_nb(void) {
    if (g_nb_trsm == 0) g_nb_trsm = env_int("MTRSM_NB", 64);
    return g_nb_trsm;
}

inline char up(const char *p) {
    return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
}

const T zero_dd{0.0, 0.0};
const T one_dd {1.0, 0.0};

inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (T x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }

/* mgemm extern — we call it for trailing-matrix updates. */
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

#ifdef MBLAS_SIMD_DD

/* ── SIMD 4-wide diagonal kernel for SIDE='L' variants.
 *
 * Layout: pack 4 columns of B into thread-local SoA hi[] / lo[]
 * arrays (interleaved by row), do forward/back substitution in
 * SIMD across the 4 column lanes, unpack back to B. Partial
 * trailing 4-panel (j_count < 4) is zero-padded so the kernel
 * always runs full 4-wide; only j_count lanes are written back.
 *
 * Operates on M ≤ kMaxBlockM (block size cap from trsm_nb). Stack
 * scratch is 2 · M · 4 doubles = 4KB at M=64.
 */
constexpr int kSimdLane = simd_dd::NR;   /* 4 */
constexpr int kMaxBlockM = 256;          /* upper bound for stack scratch */

/* Pack [M, j_start..j_start+j_count) of B into SoA scratch (bh, bl).
 * Zero-pad lanes ≥ j_count. */
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

/* SIMD alpha prescale on packed scratch. */
inline void simd_prescale(int M, T alpha, double *bh, double *bl)
{
    if (dd_isone(alpha)) return;
    if (dd_iszero(alpha)) {
        const __m256d z = _mm256_setzero_pd();
        for (int k = 0; k < M; ++k) {
            _mm256_storeu_pd(&bh[k * kSimdLane], z);
            _mm256_storeu_pd(&bl[k * kSimdLane], z);
        }
        return;
    }
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (int k = 0; k < M; ++k) {
        __m256d bkh = _mm256_loadu_pd(&bh[k * kSimdLane]);
        __m256d bkl = _mm256_loadu_pd(&bl[k * kSimdLane]);
        __m256d nh, nl;
        simd_dd::dd_mul(bkh, bkl, ah, al, nh, nl);
        _mm256_storeu_pd(&bh[k * kSimdLane], nh);
        _mm256_storeu_pd(&bl[k * kSimdLane], nl);
    }
}

/* Forward substitution (L, L, N): for k = 0..M-1 :
 *   if nounit: bk /= A[k,k]
 *   for i > k: bi -= A[i,k] * bk */
inline void simd_fwd_sub_lln(int M, const T *a, int lda, int nounit,
                             double *bh, double *bl)
{
    for (int k = 0; k < M; ++k) {
        __m256d bkh = _mm256_loadu_pd(&bh[k * kSimdLane]);
        __m256d bkl = _mm256_loadu_pd(&bl[k * kSimdLane]);
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(bkh, bkl, ih, il, nh, nl);
            bkh = nh; bkl = nl;
            _mm256_storeu_pd(&bh[k * kSimdLane], bkh);
            _mm256_storeu_pd(&bl[k * kSimdLane], bkl);
        }
        for (int i = k + 1; i < M; ++i) {
            __m256d aih = _mm256_set1_pd(A_(i, k).limbs[0]);
            __m256d ail = _mm256_set1_pd(A_(i, k).limbs[1]);
            __m256d ph, pl;
            simd_dd::dd_mul(aih, ail, bkh, bkl, ph, pl);
            simd_dd::dd_neg(ph, pl);
            __m256d bih = _mm256_loadu_pd(&bh[i * kSimdLane]);
            __m256d bil = _mm256_loadu_pd(&bl[i * kSimdLane]);
            __m256d nh, nl;
            simd_dd::dd_add(bih, bil, ph, pl, nh, nl);
            _mm256_storeu_pd(&bh[i * kSimdLane], nh);
            _mm256_storeu_pd(&bl[i * kSimdLane], nl);
        }
    }
}

/* Back substitution (L, U, N): for k = M-1..0 :
 *   if nounit: bk /= A[k,k]
 *   for i < k: bi -= A[i,k] * bk */
inline void simd_bwd_sub_lun(int M, const T *a, int lda, int nounit,
                             double *bh, double *bl)
{
    for (int k = M - 1; k >= 0; --k) {
        __m256d bkh = _mm256_loadu_pd(&bh[k * kSimdLane]);
        __m256d bkl = _mm256_loadu_pd(&bl[k * kSimdLane]);
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(bkh, bkl, ih, il, nh, nl);
            bkh = nh; bkl = nl;
            _mm256_storeu_pd(&bh[k * kSimdLane], bkh);
            _mm256_storeu_pd(&bl[k * kSimdLane], bkl);
        }
        for (int i = 0; i < k; ++i) {
            __m256d aih = _mm256_set1_pd(A_(i, k).limbs[0]);
            __m256d ail = _mm256_set1_pd(A_(i, k).limbs[1]);
            __m256d ph, pl;
            simd_dd::dd_mul(aih, ail, bkh, bkl, ph, pl);
            simd_dd::dd_neg(ph, pl);
            __m256d bih = _mm256_loadu_pd(&bh[i * kSimdLane]);
            __m256d bil = _mm256_loadu_pd(&bl[i * kSimdLane]);
            __m256d nh, nl;
            simd_dd::dd_add(bih, bil, ph, pl, nh, nl);
            _mm256_storeu_pd(&bh[i * kSimdLane], nh);
            _mm256_storeu_pd(&bl[i * kSimdLane], nl);
        }
    }
}

/* Forward sub on Aᵀ (L, L, T): inner-product form, scans i = M-1..0.
 *   t = α·B[i,j]; for k > i: t -= A[k,i]·B[k,j]; B[i,j] = t / A[i,i] */
inline void simd_fwd_sub_llt(int M, const T *a, int lda, T alpha, int nounit,
                             double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (int i = M - 1; i >= 0; --i) {
        __m256d bih = _mm256_loadu_pd(&bh[i * kSimdLane]);
        __m256d bil = _mm256_loadu_pd(&bl[i * kSimdLane]);
        __m256d th, tl;
        simd_dd::dd_mul(ah, al, bih, bil, th, tl);
        for (int k = i + 1; k < M; ++k) {
            __m256d akh = _mm256_set1_pd(A_(k, i).limbs[0]);
            __m256d akl = _mm256_set1_pd(A_(k, i).limbs[1]);
            __m256d bkh = _mm256_loadu_pd(&bh[k * kSimdLane]);
            __m256d bkl = _mm256_loadu_pd(&bl[k * kSimdLane]);
            __m256d ph, pl;
            simd_dd::dd_mul(akh, akl, bkh, bkl, ph, pl);
            simd_dd::dd_neg(ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(th, tl, ph, pl, nh, nl);
            th = nh; tl = nl;
        }
        if (nounit) {
            const T inv = one_dd / A_(i, i);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(th, tl, ih, il, nh, nl);
            th = nh; tl = nl;
        }
        _mm256_storeu_pd(&bh[i * kSimdLane], th);
        _mm256_storeu_pd(&bl[i * kSimdLane], tl);
    }
}

/* (L, U, T): scans i = 0..M-1, k = 0..i-1. */
inline void simd_bwd_sub_lut(int M, const T *a, int lda, T alpha, int nounit,
                             double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (int i = 0; i < M; ++i) {
        __m256d bih = _mm256_loadu_pd(&bh[i * kSimdLane]);
        __m256d bil = _mm256_loadu_pd(&bl[i * kSimdLane]);
        __m256d th, tl;
        simd_dd::dd_mul(ah, al, bih, bil, th, tl);
        for (int k = 0; k < i; ++k) {
            __m256d akh = _mm256_set1_pd(A_(k, i).limbs[0]);
            __m256d akl = _mm256_set1_pd(A_(k, i).limbs[1]);
            __m256d bkh = _mm256_loadu_pd(&bh[k * kSimdLane]);
            __m256d bkl = _mm256_loadu_pd(&bl[k * kSimdLane]);
            __m256d ph, pl;
            simd_dd::dd_mul(akh, akl, bkh, bkl, ph, pl);
            simd_dd::dd_neg(ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(th, tl, ph, pl, nh, nl);
            th = nh; tl = nl;
        }
        if (nounit) {
            const T inv = one_dd / A_(i, i);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(th, tl, ih, il, nh, nl);
            th = nh; tl = nl;
        }
        _mm256_storeu_pd(&bh[i * kSimdLane], th);
        _mm256_storeu_pd(&bl[i * kSimdLane], tl);
    }
}

enum trsm_simd_op { SLLN, SLUN, SLLT, SLUT };

/* SIMD diagonal solver for a column range. Replaces mtrsm_*_core on
 * the blocked path. Scratch is per-call stack (small, ≤4KB).
 * For LLN/LUN: alpha is applied to bh/bl via simd_prescale before
 *              the forward/back sub (matches the rank-1 form).
 * For LLT/LUT: alpha is folded into the i-loop (matches scalar form). */
inline void mtrsm_simd_diag(trsm_simd_op op, int j_start, int j_end,
                            int M, T alpha,
                            const T *a, int lda, T *b, int ldb, int nounit)
{
    alignas(32) double bh[kMaxBlockM * kSimdLane];
    alignas(32) double bl[kMaxBlockM * kSimdLane];
    for (int j = j_start; j < j_end; j += kSimdLane) {
        const int jc = (j_end - j < kSimdLane) ? (j_end - j) : kSimdLane;
        pack_B_4col(M, b, ldb, j, jc, bh, bl);
        switch (op) {
        case SLLN:
            simd_prescale(M, alpha, bh, bl);
            simd_fwd_sub_lln(M, a, lda, nounit, bh, bl);
            break;
        case SLUN:
            simd_prescale(M, alpha, bh, bl);
            simd_bwd_sub_lun(M, a, lda, nounit, bh, bl);
            break;
        case SLLT:
            simd_fwd_sub_llt(M, a, lda, alpha, nounit, bh, bl);
            break;
        case SLUT:
            simd_bwd_sub_lut(M, a, lda, alpha, nounit, bh, bl);
            break;
        }
        unpack_B_4col(M, b, ldb, j, jc, bh, bl);
    }
}

#endif  /* MBLAS_SIMD_DD */

/* ── Column-range "core" kernels: serial work over j ∈ [j_start, j_end). */

inline void mtrsm_lln_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (!dd_isone(alpha)) for (int i = 0; i < M; ++i) B_(i, j) = B_(i, j) * alpha;
        for (int k = 0; k < M; ++k) {
            if (!dd_iszero(B_(k, j))) {
                if (nounit) B_(k, j) = B_(k, j) / A_(k, k);
                const T bk = B_(k, j);
                for (int i = k + 1; i < M; ++i)
                    B_(i, j) = B_(i, j) - bk * A_(i, k);
            }
        }
    }
}

inline void mtrsm_lun_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (!dd_isone(alpha)) for (int i = 0; i < M; ++i) B_(i, j) = B_(i, j) * alpha;
        for (int k = M - 1; k >= 0; --k) {
            if (!dd_iszero(B_(k, j))) {
                if (nounit) B_(k, j) = B_(k, j) / A_(k, k);
                const T bk = B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) = B_(i, j) - bk * A_(i, k);
            }
        }
    }
}

inline void mtrsm_llt_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (int k = i + 1; k < M; ++k) t = t - A_(k, i) * B_(k, j);
            if (nounit) t = t / A_(i, i);
            B_(i, j) = t;
        }
    }
}

inline void mtrsm_lut_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = alpha * B_(i, j);
            for (int k = 0; k < i; ++k) t = t - A_(k, i) * B_(k, j);
            if (nounit) t = t / A_(i, i);
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R': solve X op(A) = α B. SIMD over 4-row chunks of B;
 * scalar tail for remaining rows. Same column-major loop structure
 * as reference DTRSM 'R',*,*. */

#ifdef MBLAS_SIMD_DD

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

/* RLN: B := α·B / L (R-side, lower-tri L, no transpose).
 * For each ib (4-row chunk), iterate j = N-1..0:
 *   B(:,j) *= α; B(:,j) -= sum_{k>j} A(k,j) · B(:,k); B(:,j) /= A(j,j) */
inline void simd_trsm_r4_rln(int ib, int N, T alpha,
                             const T *a, int lda, T *b, int ldb, int nounit)
{
    __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const bool alpha_nontriv = !dd_isone(alpha);
    for (int j = N - 1; j >= 0; --j) {
        T *bj = b + static_cast<std::size_t>(j) * ldb;
        __m256d bjh, bjl;
        load_4cell_soa(bj, ib, bjh, bjl);
        if (alpha_nontriv) {
            __m256d nh, nl;
            simd_dd::dd_mul(bjh, bjl, ah, al, nh, nl);
            bjh = nh; bjl = nl;
        }
        for (int k = j + 1; k < N; ++k) {
            const T akj = A_(k, j);
            if (dd_iszero(akj)) continue;
            __m256d akh = _mm256_set1_pd(akj.limbs[0]);
            __m256d akl = _mm256_set1_pd(akj.limbs[1]);
            const T *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkh, bkl;
            load_4cell_soa(bk, ib, bkh, bkl);
            __m256d ph, pl;
            simd_dd::dd_mul(akh, akl, bkh, bkl, ph, pl);
            simd_dd::dd_neg(ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(bjh, bjl, ph, pl, nh, nl);
            bjh = nh; bjl = nl;
        }
        if (nounit) {
            const T inv = one_dd / A_(j, j);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(bjh, bjl, ih, il, nh, nl);
            bjh = nh; bjl = nl;
        }
        store_4cell_soa(bj, ib, bjh, bjl);
    }
}

inline void simd_trsm_r4_run(int ib, int N, T alpha,
                             const T *a, int lda, T *b, int ldb, int nounit)
{
    __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const bool alpha_nontriv = !dd_isone(alpha);
    for (int j = 0; j < N; ++j) {
        T *bj = b + static_cast<std::size_t>(j) * ldb;
        __m256d bjh, bjl;
        load_4cell_soa(bj, ib, bjh, bjl);
        if (alpha_nontriv) {
            __m256d nh, nl;
            simd_dd::dd_mul(bjh, bjl, ah, al, nh, nl);
            bjh = nh; bjl = nl;
        }
        for (int k = 0; k < j; ++k) {
            const T akj = A_(k, j);
            if (dd_iszero(akj)) continue;
            __m256d akh = _mm256_set1_pd(akj.limbs[0]);
            __m256d akl = _mm256_set1_pd(akj.limbs[1]);
            const T *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkh, bkl;
            load_4cell_soa(bk, ib, bkh, bkl);
            __m256d ph, pl;
            simd_dd::dd_mul(akh, akl, bkh, bkl, ph, pl);
            simd_dd::dd_neg(ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(bjh, bjl, ph, pl, nh, nl);
            bjh = nh; bjl = nl;
        }
        if (nounit) {
            const T inv = one_dd / A_(j, j);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(bjh, bjl, ih, il, nh, nl);
            bjh = nh; bjl = nl;
        }
        store_4cell_soa(bj, ib, bjh, bjl);
    }
}

/* RLT: B := α·B / Lᵀ. Iterate k = 0..N-1:
 *   B(:,k) /= A(k,k) (if nounit); subtract from B(:,j) for j > k; α-scale B(:,k) */
inline void simd_trsm_r4_rlt(int ib, int N, T alpha,
                             const T *a, int lda, T *b, int ldb, int nounit)
{
    __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const bool alpha_nontriv = !dd_isone(alpha);
    for (int k = 0; k < N; ++k) {
        T *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkh, bkl;
        load_4cell_soa(bk, ib, bkh, bkl);
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(bkh, bkl, ih, il, nh, nl);
            bkh = nh; bkl = nl;
            store_4cell_soa(bk, ib, bkh, bkl);
        }
        for (int j = k + 1; j < N; ++j) {
            const T ajk = A_(j, k);
            if (dd_iszero(ajk)) continue;
            __m256d ajh = _mm256_set1_pd(ajk.limbs[0]);
            __m256d ajl = _mm256_set1_pd(ajk.limbs[1]);
            T *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjh, bjl;
            load_4cell_soa(bj, ib, bjh, bjl);
            __m256d ph, pl;
            simd_dd::dd_mul(ajh, ajl, bkh, bkl, ph, pl);
            simd_dd::dd_neg(ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(bjh, bjl, ph, pl, nh, nl);
            store_4cell_soa(bj, ib, nh, nl);
        }
        if (alpha_nontriv) {
            __m256d nh, nl;
            simd_dd::dd_mul(bkh, bkl, ah, al, nh, nl);
            store_4cell_soa(bk, ib, nh, nl);
        }
    }
}

inline void simd_trsm_r4_rut(int ib, int N, T alpha,
                             const T *a, int lda, T *b, int ldb, int nounit)
{
    __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const bool alpha_nontriv = !dd_isone(alpha);
    for (int k = N - 1; k >= 0; --k) {
        T *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkh, bkl;
        load_4cell_soa(bk, ib, bkh, bkl);
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_dd::dd_mul(bkh, bkl, ih, il, nh, nl);
            bkh = nh; bkl = nl;
            store_4cell_soa(bk, ib, bkh, bkl);
        }
        for (int j = 0; j < k; ++j) {
            const T ajk = A_(j, k);
            if (dd_iszero(ajk)) continue;
            __m256d ajh = _mm256_set1_pd(ajk.limbs[0]);
            __m256d ajl = _mm256_set1_pd(ajk.limbs[1]);
            T *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjh, bjl;
            load_4cell_soa(bj, ib, bjh, bjl);
            __m256d ph, pl;
            simd_dd::dd_mul(ajh, ajl, bkh, bkl, ph, pl);
            simd_dd::dd_neg(ph, pl);
            __m256d nh, nl;
            simd_dd::dd_add(bjh, bjl, ph, pl, nh, nl);
            store_4cell_soa(bj, ib, nh, nl);
        }
        if (alpha_nontriv) {
            __m256d nh, nl;
            simd_dd::dd_mul(bkh, bkl, ah, al, nh, nl);
            store_4cell_soa(bk, ib, nh, nl);
        }
    }
}

enum trsm_r_op { TRSM_RLN, TRSM_RUN, TRSM_RLT, TRSM_RUT };

/* Forward decl scalar tails (defined below). */
inline void mtrsm_rln_core(int, int, int, T, const T*, int, T*, int, int);
inline void mtrsm_run_core(int, int, int, T, const T*, int, T*, int, int);
inline void mtrsm_rlt_core(int, int, int, T, const T*, int, T*, int, int);
inline void mtrsm_rut_core(int, int, int, T, const T*, int, T*, int, int);

inline void mtrsm_simd_diag_R(trsm_r_op op, int M, int N, T alpha,
                              const T *a, int lda, T *b, int ldb, int nounit)
{
    const int M4 = M & ~3;
    for (int ib = 0; ib < M4; ib += 4) {
        switch (op) {
        case TRSM_RLN: simd_trsm_r4_rln(ib, N, alpha, a, lda, b, ldb, nounit); break;
        case TRSM_RUN: simd_trsm_r4_run(ib, N, alpha, a, lda, b, ldb, nounit); break;
        case TRSM_RLT: simd_trsm_r4_rlt(ib, N, alpha, a, lda, b, ldb, nounit); break;
        case TRSM_RUT: simd_trsm_r4_rut(ib, N, alpha, a, lda, b, ldb, nounit); break;
        }
    }
    /* Scalar tail rows: each *_core processes full B[i:M, *]. Need to constrain
     * to i ∈ [M4, M). The scalar cores take (j_start, j_end) for SIDE='L', but
     * SIDE='R' cores use j_end as N and process all M rows. We need a row-range
     * variant. Simplest: directly inline the scalar logic over rows [M4, M). */
    if (M4 < M) {
        const int Mt = M;  /* keep names consistent */
        switch (op) {
        case TRSM_RLN: {
            for (int j = N - 1; j >= 0; --j) {
                if (!dd_isone(alpha)) for (int i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) * alpha;
                for (int k = j + 1; k < N; ++k) {
                    if (!dd_iszero(A_(k, j))) {
                        const T akj = A_(k, j);
                        for (int i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) - akj * B_(i, k);
                    }
                }
                if (nounit) { const T inv = one_dd / A_(j, j);
                    for (int i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) * inv; }
            }
        } break;
        case TRSM_RUN: {
            for (int j = 0; j < N; ++j) {
                if (!dd_isone(alpha)) for (int i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) * alpha;
                for (int k = 0; k < j; ++k) {
                    if (!dd_iszero(A_(k, j))) {
                        const T akj = A_(k, j);
                        for (int i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) - akj * B_(i, k);
                    }
                }
                if (nounit) { const T inv = one_dd / A_(j, j);
                    for (int i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) * inv; }
            }
        } break;
        case TRSM_RLT: {
            for (int k = 0; k < N; ++k) {
                if (nounit) { const T inv = one_dd / A_(k, k);
                    for (int i = M4; i < Mt; ++i) B_(i, k) = B_(i, k) * inv; }
                for (int j = k + 1; j < N; ++j) {
                    if (!dd_iszero(A_(j, k))) {
                        const T ajk = A_(j, k);
                        for (int i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) - ajk * B_(i, k);
                    }
                }
                if (!dd_isone(alpha)) for (int i = M4; i < Mt; ++i) B_(i, k) = B_(i, k) * alpha;
            }
        } break;
        case TRSM_RUT: {
            for (int k = N - 1; k >= 0; --k) {
                if (nounit) { const T inv = one_dd / A_(k, k);
                    for (int i = M4; i < Mt; ++i) B_(i, k) = B_(i, k) * inv; }
                for (int j = 0; j < k; ++j) {
                    if (!dd_iszero(A_(j, k))) {
                        const T ajk = A_(j, k);
                        for (int i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) - ajk * B_(i, k);
                    }
                }
                if (!dd_isone(alpha)) for (int i = M4; i < Mt; ++i) B_(i, k) = B_(i, k) * alpha;
            }
        } break;
        }
    }
}

#endif  /* MBLAS_SIMD_DD */

inline void mtrsm_rln_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    /* For SIDE='R', the algorithm reorders columns of B (j loop is
     * outermost over A's columns), so j_start/j_end indicate which
     * B-rows this thread owns. (Actually for SIDE='R' the natural
     * partition is rows of B, not columns.) Handled by callers. */
    (void)j_start; (void)j_end;
    /* Serial implementation matches reference DTRSM 'R','L','N': */
    const int N = j_end;  /* placeholder — see caller */
    for (int j = N - 1; j >= 0; --j) {
        if (!dd_isone(alpha)) for (int i = 0; i < M; ++i) B_(i, j) = B_(i, j) * alpha;
        for (int k = j + 1; k < N; ++k) {
            if (!dd_iszero(A_(k, j))) {
                const T akj = A_(k, j);
                for (int i = 0; i < M; ++i)
                    B_(i, j) = B_(i, j) - akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = one_dd / A_(j, j);
            for (int i = 0; i < M; ++i) B_(i, j) = B_(i, j) * inv;
        }
    }
}

inline void mtrsm_run_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    (void)j_start;
    const int N = j_end;
    for (int j = 0; j < N; ++j) {
        if (!dd_isone(alpha)) for (int i = 0; i < M; ++i) B_(i, j) = B_(i, j) * alpha;
        for (int k = 0; k < j; ++k) {
            if (!dd_iszero(A_(k, j))) {
                const T akj = A_(k, j);
                for (int i = 0; i < M; ++i)
                    B_(i, j) = B_(i, j) - akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = one_dd / A_(j, j);
            for (int i = 0; i < M; ++i) B_(i, j) = B_(i, j) * inv;
        }
    }
}

inline void mtrsm_rlt_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    (void)j_start;
    const int N = j_end;
    for (int k = 0; k < N; ++k) {
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            for (int i = 0; i < M; ++i) B_(i, k) = B_(i, k) * inv;
        }
        for (int j = k + 1; j < N; ++j) {
            if (!dd_iszero(A_(j, k))) {
                const T ajk = A_(j, k);
                for (int i = 0; i < M; ++i)
                    B_(i, j) = B_(i, j) - ajk * B_(i, k);
            }
        }
        if (!dd_isone(alpha)) for (int i = 0; i < M; ++i) B_(i, k) = B_(i, k) * alpha;
    }
}

inline void mtrsm_rut_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    (void)j_start;
    const int N = j_end;
    for (int k = N - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            for (int i = 0; i < M; ++i) B_(i, k) = B_(i, k) * inv;
        }
        for (int j = 0; j < k; ++j) {
            if (!dd_iszero(A_(j, k))) {
                const T ajk = A_(j, k);
                for (int i = 0; i < M; ++i)
                    B_(i, j) = B_(i, j) - ajk * B_(i, k);
            }
        }
        if (!dd_isone(alpha)) for (int i = 0; i < M; ++i) B_(i, k) = B_(i, k) * alpha;
    }
}

/* Standalone unblocked SIDE='L' entries: wrap _core in own parallel region. */
#ifdef _OPENMP
#define MTRSM_OMP_WRAP(name, core)                                          \
    void name(int M, int N, T alpha,                                        \
              const T *a, int lda, T *b, int ldb, int nounit)               \
    {                                                                       \
        if (N >= MTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel")                                         \
            {                                                               \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = static_cast<int>((long long)N * tid / nt);        \
                int je  = static_cast<int>((long long)N * (tid + 1) / nt);  \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else {                                                            \
            core(0, N, M, alpha, a, lda, b, ldb, nounit);                   \
        }                                                                   \
    }
#else
#define MTRSM_OMP_WRAP(name, core)                                          \
    void name(int M, int N, T alpha,                                        \
              const T *a, int lda, T *b, int ldb, int nounit)               \
    {                                                                       \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#endif

MTRSM_OMP_WRAP(mtrsm_lln, mtrsm_lln_core)
MTRSM_OMP_WRAP(mtrsm_lun, mtrsm_lun_core)
MTRSM_OMP_WRAP(mtrsm_llt, mtrsm_llt_core)
MTRSM_OMP_WRAP(mtrsm_lut, mtrsm_lut_core)

/* SIDE='R' kernels just take the full N — no column partitioning. */
void mtrsm_rln(int M, int N, T alpha,
               const T *a, int lda, T *b, int ldb, int nounit) {
    mtrsm_rln_core(0, N, M, alpha, a, lda, b, ldb, nounit);
}
void mtrsm_run(int M, int N, T alpha,
               const T *a, int lda, T *b, int ldb, int nounit) {
    mtrsm_run_core(0, N, M, alpha, a, lda, b, ldb, nounit);
}
void mtrsm_rlt(int M, int N, T alpha,
               const T *a, int lda, T *b, int ldb, int nounit) {
    mtrsm_rlt_core(0, N, M, alpha, a, lda, b, ldb, nounit);
}
void mtrsm_rut(int M, int N, T alpha,
               const T *a, int lda, T *b, int ldb, int nounit) {
    mtrsm_rut_core(0, N, M, alpha, a, lda, b, ldb, nounit);
}

/* ── Blocked SIDE='L' variants: coarse-grain parallelism across N.
 *
 * One outer omp parallel partitions columns of B across threads.
 * Each thread runs serial blocked-TRSM on its own column chunk:
 *   - mgemm trailing update (auto runs single-threaded internally due
 *     to OMP_NESTED=false → no inner team. The SIMD micro-kernel
 *     still runs at full SIMD width per call.)
 *   - scalar core diagonal solve on the thread's column range.
 */

inline void prescale_chunk(int j_start, int j_end, int M, T alpha,
                           T *b, int ldb)
{
    if (dd_isone(alpha)) return;
    if (dd_iszero(alpha)) {
        for (int j = j_start; j < j_end; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = zero_dd;
        return;
    }
    for (int j = j_start; j < j_end; ++j)
        for (int i = 0; i < M; ++i) B_(i, j) = B_(i, j) * alpha;
}

enum trsm_variant { LLN, LUN, LLT, LUT };

void blocked_chunk(trsm_variant V, int j_start, int j_end,
                   int M, int nb, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    const int my_N = j_end - j_start;
    if (my_N <= 0) return;
    prescale_chunk(j_start, j_end, M, alpha, b, ldb);

    const T m_one = T{-1.0, 0.0};
    const T one   = T{ 1.0, 0.0};
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    T *B_chunk = &B_(0, j_start);

/* Diagonal-solve helper: SIMD path if available, scalar otherwise. */
#ifdef MBLAS_SIMD_DD
#define DIAG_SOLVE(op, scalar_core, ib_arg, alpha_arg)               \
    do {                                                              \
        if ((ib_arg) <= kMaxBlockM)                                   \
            mtrsm_simd_diag(op, j_start, j_end, (ib_arg), (alpha_arg),\
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);\
        else                                                          \
            scalar_core(j_start, j_end, (ib_arg), (alpha_arg),        \
                        &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);   \
    } while (0)
#else
#define DIAG_SOLVE(op, scalar_core, ib_arg, alpha_arg)               \
    scalar_core(j_start, j_end, (ib_arg), (alpha_arg),                \
                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit)
#endif

    if (V == LLN) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                mgemm_(NN, NN, &ib, &my_N, &ic, &m_one,
                       &A_(ic, 0), &lda,
                       B_chunk, &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            DIAG_SOLVE(SLLN, mtrsm_lln_core, ib, one_dd);
        }
    } else if (V == LUN) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int j0 = ic + ib;
                mgemm_(NN, NN, &ib, &my_N, &trailing, &m_one,
                       &A_(ic, j0), &lda,
                       &B_chunk[j0], &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            DIAG_SOLVE(SLUN, mtrsm_lun_core, ib, one_dd);
            ic -= nb;
        }
    } else if (V == LLT) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int i0 = ic + ib;
                mgemm_(TN, NN, &ib, &my_N, &trailing, &m_one,
                       &A_(i0, ic), &lda,
                       &B_chunk[i0], &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            DIAG_SOLVE(SLLT, mtrsm_llt_core, ib, one_dd);
            ic -= nb;
        }
    } else { /* LUT */
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                mgemm_(TN, NN, &ib, &my_N, &ic, &m_one,
                       &A_(0, ic), &lda,
                       B_chunk, &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            DIAG_SOLVE(SLUT, mtrsm_lut_core, ib, one_dd);
        }
    }
#undef DIAG_SOLVE
}

void blocked_dispatch(trsm_variant V, int M, int N, T alpha,
                      const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = trsm_nb();
#ifdef _OPENMP
    if (N >= MTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int js  = static_cast<int>((long long)N * tid / nt);
            int je  = static_cast<int>((long long)N * (tid + 1) / nt);
            blocked_chunk(V, js, je, M, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    blocked_chunk(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
}

void blocked_lln(int M, int N, T alpha, const T *a, int lda, T *b, int ldb, int nounit) {
    blocked_dispatch(LLN, M, N, alpha, a, lda, b, ldb, nounit);
}
void blocked_lun(int M, int N, T alpha, const T *a, int lda, T *b, int ldb, int nounit) {
    blocked_dispatch(LUN, M, N, alpha, a, lda, b, ldb, nounit);
}
void blocked_llt(int M, int N, T alpha, const T *a, int lda, T *b, int ldb, int nounit) {
    blocked_dispatch(LLT, M, N, alpha, a, lda, b, ldb, nounit);
}
void blocked_lut(int M, int N, T alpha, const T *a, int lda, T *b, int ldb, int nounit) {
    blocked_dispatch(LUT, M, N, alpha, a, lda, b, ldb, nounit);
}

}  // namespace

extern "C" void mtrsm_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    std::size_t side_len, std::size_t uplo_len,
    std::size_t transa_len, std::size_t diag_len)
{
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;
    const char SIDE = up(side);
    const char UPLO = up(uplo);
    char TR = up(transa);
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (dd_iszero(alpha)) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = zero_dd;
        return;
    }

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * trsm_nb());
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_lln(M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrsm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_lun(M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrsm_lun(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else {
            if (UPLO == 'L') {
                if (use_blocked) blocked_llt(M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrsm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_lut(M, N, alpha, a, lda, b, ldb, nounit);
                else             mtrsm_lut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    } else {
        /* SIDE='R': partition over rows of B. The j (column) loop walks
         * the diagonal serially, but every row of B is processed
         * identically — each thread takes a disjoint row slice and the
         * work is race-free with no barriers.
         *
         * Round slice boundaries to multiples of 4 where possible so
         * the SIMD 4-row chunks inside mtrsm_simd_diag_R stay aligned;
         * the last thread absorbs any non-4-aligned tail. */
#ifdef _OPENMP
        const int use_omp = (M >= MTRSM_OMP_N_MIN && blas_omp_max_threads() > 1
                             && !omp_in_parallel());
#else
        const int use_omp = 0;
#endif

#ifdef MBLAS_SIMD_DD
        trsm_r_op op;
        if (TR == 'N') op = (UPLO == 'L') ? TRSM_RLN : TRSM_RUN;
        else           op = (UPLO == 'L') ? TRSM_RLT : TRSM_RUT;
#endif

#ifdef _OPENMP
        #pragma omp parallel if(use_omp)
#endif
        {
            int tid = 0, nt = 1;
#ifdef _OPENMP
            if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
#endif
            int i_lo = (int)((long long)M * tid / nt);
            int i_hi = (int)((long long)M * (tid + 1) / nt);
            /* Round i_lo down to a multiple of 4 (except thread 0), and
             * i_hi up to a multiple of 4 — except the last thread, which
             * absorbs the M&3 tail. This keeps SIMD chunks aligned for
             * interior threads. */
            if (tid > 0)      i_lo &= ~3;
            if (tid < nt - 1) i_hi &= ~3;
            const int Mslice = i_hi - i_lo;
            if (Mslice > 0) {
                T *b_slice = b + i_lo;
#ifdef MBLAS_SIMD_DD
                mtrsm_simd_diag_R(op, Mslice, N, alpha, a, lda, b_slice, ldb, nounit);
#else
                if (TR == 'N') {
                    if (UPLO == 'L') mtrsm_rln_core(0, N, Mslice, alpha, a, lda, b_slice, ldb, nounit);
                    else             mtrsm_run_core(0, N, Mslice, alpha, a, lda, b_slice, ldb, nounit);
                } else {
                    if (UPLO == 'L') mtrsm_rlt_core(0, N, Mslice, alpha, a, lda, b_slice, ldb, nounit);
                    else             mtrsm_rut_core(0, N, Mslice, alpha, a, lda, b_slice, ldb, nounit);
                }
#endif
            }
        }
    }
}

#undef A_
#undef B_
