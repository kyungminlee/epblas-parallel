/*
 * mtrsm_serial.cpp — multifloats real (double-double) triangular solve,
 * single-thread core. Owns ALL the numerics shared by the serial and parallel
 * entries:
 *
 *   - scalar column "core" kernels for SIDE='L' (LLN/LUN/LLT/LUT) and the
 *     SIDE='R' cores (RLN/RUN/RLT/RUT),
 *   - the AVX2 4-wide SIMD diagonal kernels (SIDE='L' packed-SoA and SIDE='R'
 *     4-row chunks), under MBLAS_SIMD_DD,
 *   - the block-size policy and the blocked SIDE='L' chunk worker, whose
 *     trailing-matrix update routes through mgemm_serial (no nested OpenMP),
 *   - the per-slice workers mtrsm_L_slice / mtrsm_R_slice (declared in
 *     mtrsm_kernel.h) that the parallel entry fans across a team, plus the
 *     public `mtrsm_serial` entry.
 *
 * There is NO OpenMP on this path. Threading lives entirely in
 * mtrsm_parallel.cpp; both paths drive these workers, so a static partition
 * is bitwise-identical to the serial sweep.
 */

#include "mtrsm_kernel.h"
#include "mgemm_kernel.h"   /* mgemm_serial for the trailing update */
#include <cstddef>
#include <cstdlib>
#include <cctype>

#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"   /* dd_mul, dd_add, dd_neg primitives */
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {

int g_nb_trsm = 0;
int trsm_nb(void) {
    if (g_nb_trsm == 0) g_nb_trsm = 64;
    return g_nb_trsm;
}

const T zero_dd{0.0, 0.0};
const T one_dd {1.0, 0.0};

inline bool dd_iszero(T x) { return x.limbs[0] == 0.0 && x.limbs[1] == 0.0; }
inline bool dd_isone (T x) { return x.limbs[0] == 1.0 && x.limbs[1] == 0.0; }

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
    /* Scalar tail rows i ∈ [M4, M): the SIDE='R' diagonal walks all rows of
     * B identically, so the tail is just the same recurrence over the
     * non-4-aligned rows. */
    if (M4 < M) {
        const int Mt = M;
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

inline void mtrsm_rln_core(int N, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
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

inline void mtrsm_run_core(int N, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
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

inline void mtrsm_rlt_core(int N, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
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

inline void mtrsm_rut_core(int N, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
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

/* ── Blocked SIDE='L' chunk worker: serial blocked-TRSM over one column slice
 * [j_start, j_end). The mgemm trailing update routes through mgemm_serial. */

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
                mgemm_serial(NN, NN, &ib, &my_N, &ic, &m_one,
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
                mgemm_serial(NN, NN, &ib, &my_N, &trailing, &m_one,
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
                mgemm_serial(TN, NN, &ib, &my_N, &trailing, &m_one,
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
                mgemm_serial(TN, NN, &ib, &my_N, &ic, &m_one,
                             &A_(0, ic), &lda,
                             B_chunk, &ldb, &one,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            DIAG_SOLVE(SLUT, mtrsm_lut_core, ib, one_dd);
        }
    }
#undef DIAG_SOLVE
}

/* Map (UPLO, TR) → blocked variant. */
inline trsm_variant l_variant(char UPLO, char TR) {
    if (TR == 'N') return (UPLO == 'L') ? LLN : LUN;
    return (UPLO == 'L') ? LLT : LUT;
}

}  // namespace

/* ── Exposed surface (mtrsm_kernel.h). ─────────────────────────────────── */

int mtrsm_block_nb(void) { return trsm_nb(); }

void mtrsm_zero_B(int M, int N, T *b, int ldb)
{
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i) B_(i, j) = zero_dd;
}

void mtrsm_L_slice(char UPLO, char TR, int use_blocked,
                   int j_start, int j_end, int M, int nb, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    if (j_start >= j_end) return;
    const trsm_variant V = l_variant(UPLO, TR);
    if (use_blocked) {
        blocked_chunk(V, j_start, j_end, M, nb, alpha, a, lda, b, ldb, nounit);
        return;
    }
    switch (V) {
    case LLN: mtrsm_lln_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit); break;
    case LUN: mtrsm_lun_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit); break;
    case LLT: mtrsm_llt_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit); break;
    case LUT: mtrsm_lut_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit); break;
    }
}

void mtrsm_R_slice(char UPLO, char TR, int row_lo, int row_hi,
                   int N, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
{
    const int Mslice = row_hi - row_lo;
    if (Mslice <= 0) return;
    T *b_slice = b + row_lo;
#ifdef MBLAS_SIMD_DD
    trsm_r_op op;
    if (TR == 'N') op = (UPLO == 'L') ? TRSM_RLN : TRSM_RUN;
    else           op = (UPLO == 'L') ? TRSM_RLT : TRSM_RUT;
    mtrsm_simd_diag_R(op, Mslice, N, alpha, a, lda, b_slice, ldb, nounit);
#else
    if (TR == 'N') {
        if (UPLO == 'L') mtrsm_rln_core(N, Mslice, alpha, a, lda, b_slice, ldb, nounit);
        else             mtrsm_run_core(N, Mslice, alpha, a, lda, b_slice, ldb, nounit);
    } else {
        if (UPLO == 'L') mtrsm_rlt_core(N, Mslice, alpha, a, lda, b_slice, ldb, nounit);
        else             mtrsm_rut_core(N, Mslice, alpha, a, lda, b_slice, ldb, nounit);
    }
#endif
}

extern "C" void mtrsm_serial(
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
    auto up = [](const char *p) {
        return static_cast<char>(std::toupper(static_cast<unsigned char>(*p)));
    };
    const char SIDE = up(side);
    const char UPLO = up(uplo);
    char TR = up(transa);
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (dd_iszero(alpha)) { mtrsm_zero_B(M, N, b, ldb); return; }

    if (SIDE == 'L') {
        const int nb = trsm_nb();
        const int use_blocked = (M >= 2 * nb);
        mtrsm_L_slice(UPLO, TR, use_blocked, 0, N, M, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        mtrsm_R_slice(UPLO, TR, 0, M, N, alpha, a, lda, b, ldb, nounit);
    }
}

#undef A_
#undef B_
