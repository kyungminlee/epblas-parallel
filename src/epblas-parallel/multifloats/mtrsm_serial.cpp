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
#include "mf_pred.h"
#include "mgemm_kernel.h"   /* mgemm_serial for the trailing update */
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include "mf_util.h"

#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"   /* mul, add, neg primitives */
#include "mf_simd_exact.h"  /* canonical load_dd4 */
#include <immintrin.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::eq0;
using mf_pred::eq1;
namespace {

std::ptrdiff_t g_nb_trsm = 0;
std::ptrdiff_t trsm_nb(void) {
    if (g_nb_trsm == 0) g_nb_trsm = 64;
    return g_nb_trsm;
}

const T zero_dd{0.0, 0.0};
const T one_dd {1.0, 0.0};


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
constexpr std::ptrdiff_t kSimdLane = simd_fast::NR;   /* 4 */
constexpr std::ptrdiff_t kMaxBlockM = 256;          /* upper bound for stack scratch */

/* Pack [M, j_start..j_start+j_count) of B into SoA scratch (bh, bl).
 * Zero-pad lanes ≥ j_count. */
inline void pack_B_4col(std::ptrdiff_t M, const T *b, std::ptrdiff_t ldb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                        double *bh, double *bl)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        const T *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (std::ptrdiff_t i = 0; i < M; ++i) {
            bh[i * kSimdLane + j] = col[i].limbs[0];
            bl[i * kSimdLane + j] = col[i].limbs[1];
        }
    }
    for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j)
        for (std::ptrdiff_t i = 0; i < M; ++i) {
            bh[i * kSimdLane + j] = 0.0;
            bl[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_B_4col(std::ptrdiff_t M, T *b, std::ptrdiff_t ldb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                          const double *bh, const double *bl)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        T *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (std::ptrdiff_t i = 0; i < M; ++i) {
            col[i].limbs[0] = bh[i * kSimdLane + j];
            col[i].limbs[1] = bl[i * kSimdLane + j];
        }
    }
}

/* SIMD alpha prescale on packed scratch. */
inline void simd_prescale(std::ptrdiff_t M, T alpha, double *bh, double *bl)
{
    if (eq1(alpha)) return;
    if (eq0(alpha)) {
        const __m256d z = _mm256_setzero_pd();
        for (std::ptrdiff_t k = 0; k < M; ++k) {
            _mm256_storeu_pd(&bh[k * kSimdLane], z);
            _mm256_storeu_pd(&bl[k * kSimdLane], z);
        }
        return;
    }
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (std::ptrdiff_t k = 0; k < M; ++k) {
        __m256d bkh = _mm256_loadu_pd(&bh[k * kSimdLane]);
        __m256d bkl = _mm256_loadu_pd(&bl[k * kSimdLane]);
        __m256d nh, nl;
        simd_fast::mul(bkh, bkl, ah, al, nh, nl);
        _mm256_storeu_pd(&bh[k * kSimdLane], nh);
        _mm256_storeu_pd(&bl[k * kSimdLane], nl);
    }
}

/* Forward substitution (L, L, N): for k = 0..M-1 :
 *   if nounit: bk /= A[k,k]
 *   for i > k: bi -= A[i,k] * bk */
inline void simd_fwd_sub_lln(std::ptrdiff_t M, const T *a, std::ptrdiff_t lda, std::ptrdiff_t nounit,
                             double *bh, double *bl)
{
    for (std::ptrdiff_t k = 0; k < M; ++k) {
        __m256d bkh = _mm256_loadu_pd(&bh[k * kSimdLane]);
        __m256d bkl = _mm256_loadu_pd(&bl[k * kSimdLane]);
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(bkh, bkl, ih, il, nh, nl);
            bkh = nh; bkl = nl;
            _mm256_storeu_pd(&bh[k * kSimdLane], bkh);
            _mm256_storeu_pd(&bl[k * kSimdLane], bkl);
        }
        for (std::ptrdiff_t i = k + 1; i < M; ++i) {
            __m256d aih = _mm256_set1_pd(A_(i, k).limbs[0]);
            __m256d ail = _mm256_set1_pd(A_(i, k).limbs[1]);
            __m256d ph, pl;
            simd_fast::mul(aih, ail, bkh, bkl, ph, pl);
            simd_fast::neg(ph, pl);
            __m256d bih = _mm256_loadu_pd(&bh[i * kSimdLane]);
            __m256d bil = _mm256_loadu_pd(&bl[i * kSimdLane]);
            __m256d nh, nl;
            simd_fast::add(bih, bil, ph, pl, nh, nl);
            _mm256_storeu_pd(&bh[i * kSimdLane], nh);
            _mm256_storeu_pd(&bl[i * kSimdLane], nl);
        }
    }
}

/* Back substitution (L, U, N): for k = M-1..0 :
 *   if nounit: bk /= A[k,k]
 *   for i < k: bi -= A[i,k] * bk */
inline void simd_bwd_sub_lun(std::ptrdiff_t M, const T *a, std::ptrdiff_t lda, std::ptrdiff_t nounit,
                             double *bh, double *bl)
{
    for (std::ptrdiff_t k = M - 1; k >= 0; --k) {
        __m256d bkh = _mm256_loadu_pd(&bh[k * kSimdLane]);
        __m256d bkl = _mm256_loadu_pd(&bl[k * kSimdLane]);
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(bkh, bkl, ih, il, nh, nl);
            bkh = nh; bkl = nl;
            _mm256_storeu_pd(&bh[k * kSimdLane], bkh);
            _mm256_storeu_pd(&bl[k * kSimdLane], bkl);
        }
        for (std::ptrdiff_t i = 0; i < k; ++i) {
            __m256d aih = _mm256_set1_pd(A_(i, k).limbs[0]);
            __m256d ail = _mm256_set1_pd(A_(i, k).limbs[1]);
            __m256d ph, pl;
            simd_fast::mul(aih, ail, bkh, bkl, ph, pl);
            simd_fast::neg(ph, pl);
            __m256d bih = _mm256_loadu_pd(&bh[i * kSimdLane]);
            __m256d bil = _mm256_loadu_pd(&bl[i * kSimdLane]);
            __m256d nh, nl;
            simd_fast::add(bih, bil, ph, pl, nh, nl);
            _mm256_storeu_pd(&bh[i * kSimdLane], nh);
            _mm256_storeu_pd(&bl[i * kSimdLane], nl);
        }
    }
}

/* Forward sub on Aᵀ (L, L, T): inner-product form, scans i = M-1..0.
 *   t = α·B[i,j]; for k > i: t -= A[k,i]·B[k,j]; B[i,j] = t / A[i,i] */
inline void simd_fwd_sub_llt(std::ptrdiff_t M, const T *a, std::ptrdiff_t lda, T alpha, std::ptrdiff_t nounit,
                             double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (std::ptrdiff_t i = M - 1; i >= 0; --i) {
        __m256d bih = _mm256_loadu_pd(&bh[i * kSimdLane]);
        __m256d bil = _mm256_loadu_pd(&bl[i * kSimdLane]);
        __m256d th, tl;
        simd_fast::mul(ah, al, bih, bil, th, tl);
        for (std::ptrdiff_t k = i + 1; k < M; ++k) {
            __m256d akh = _mm256_set1_pd(A_(k, i).limbs[0]);
            __m256d akl = _mm256_set1_pd(A_(k, i).limbs[1]);
            __m256d bkh = _mm256_loadu_pd(&bh[k * kSimdLane]);
            __m256d bkl = _mm256_loadu_pd(&bl[k * kSimdLane]);
            __m256d ph, pl;
            simd_fast::mul(akh, akl, bkh, bkl, ph, pl);
            simd_fast::neg(ph, pl);
            __m256d nh, nl;
            simd_fast::add(th, tl, ph, pl, nh, nl);
            th = nh; tl = nl;
        }
        if (nounit) {
            const T inv = one_dd / A_(i, i);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(th, tl, ih, il, nh, nl);
            th = nh; tl = nl;
        }
        _mm256_storeu_pd(&bh[i * kSimdLane], th);
        _mm256_storeu_pd(&bl[i * kSimdLane], tl);
    }
}

/* (L, U, T): scans i = 0..M-1, k = 0..i-1. */
inline void simd_bwd_sub_lut(std::ptrdiff_t M, const T *a, std::ptrdiff_t lda, T alpha, std::ptrdiff_t nounit,
                             double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (std::ptrdiff_t i = 0; i < M; ++i) {
        __m256d bih = _mm256_loadu_pd(&bh[i * kSimdLane]);
        __m256d bil = _mm256_loadu_pd(&bl[i * kSimdLane]);
        __m256d th, tl;
        simd_fast::mul(ah, al, bih, bil, th, tl);
        for (std::ptrdiff_t k = 0; k < i; ++k) {
            __m256d akh = _mm256_set1_pd(A_(k, i).limbs[0]);
            __m256d akl = _mm256_set1_pd(A_(k, i).limbs[1]);
            __m256d bkh = _mm256_loadu_pd(&bh[k * kSimdLane]);
            __m256d bkl = _mm256_loadu_pd(&bl[k * kSimdLane]);
            __m256d ph, pl;
            simd_fast::mul(akh, akl, bkh, bkl, ph, pl);
            simd_fast::neg(ph, pl);
            __m256d nh, nl;
            simd_fast::add(th, tl, ph, pl, nh, nl);
            th = nh; tl = nl;
        }
        if (nounit) {
            const T inv = one_dd / A_(i, i);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(th, tl, ih, il, nh, nl);
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
inline void mtrsm_simd_diag(trsm_simd_op op, std::ptrdiff_t j_start, std::ptrdiff_t j_end,
                            std::ptrdiff_t M, T alpha,
                            const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    alignas(32) double bh[kMaxBlockM * kSimdLane];
    alignas(32) double bl[kMaxBlockM * kSimdLane];
    for (std::ptrdiff_t j = j_start; j < j_end; j += kSimdLane) {
        const std::ptrdiff_t jc = (j_end - j < kSimdLane) ? (j_end - j) : kSimdLane;
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

inline void mtrsm_lln_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        if (!eq1(alpha)) for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, j) = B_(i, j) * alpha;
        for (std::ptrdiff_t k = 0; k < M; ++k) {
            if (!eq0(B_(k, j))) {
                if (nounit) B_(k, j) = B_(k, j) / A_(k, k);
                const T bk = B_(k, j);
                for (std::ptrdiff_t i = k + 1; i < M; ++i)
                    B_(i, j) = B_(i, j) - bk * A_(i, k);
            }
        }
    }
}

inline void mtrsm_lun_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        if (!eq1(alpha)) for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, j) = B_(i, j) * alpha;
        for (std::ptrdiff_t k = M - 1; k >= 0; --k) {
            if (!eq0(B_(k, j))) {
                if (nounit) B_(k, j) = B_(k, j) / A_(k, k);
                const T bk = B_(k, j);
                for (std::ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) = B_(i, j) - bk * A_(i, k);
            }
        }
    }
}

inline void mtrsm_llt_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t i = M - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (std::ptrdiff_t k = i + 1; k < M; ++k) t = t - A_(k, i) * B_(k, j);
            if (nounit) t = t / A_(i, i);
            B_(i, j) = t;
        }
    }
}

inline void mtrsm_lut_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t i = 0; i < M; ++i) {
            T t = alpha * B_(i, j);
            for (std::ptrdiff_t k = 0; k < i; ++k) t = t - A_(k, i) * B_(k, j);
            if (nounit) t = t / A_(i, i);
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R': solve X op(A) = α B. SIMD over 4-row chunks of B;
 * scalar tail for remaining rows. Same column-major loop structure
 * as reference DTRSM 'R',*,*. */

#ifdef MBLAS_SIMD_DD

/* AoS→SoA 4-cell transpose load: canonical simd_exact::load_dd4 (col + ofs). */
using simd_exact::load_dd4;
using simd_exact::store_dd4;

/* RLN: B := α·B / L (R-side, lower-tri L, no transpose).
 * For each ib (4-row chunk), iterate j = N-1..0:
 *   B(:,j) *= α; B(:,j) -= sum_{k>j} A(k,j) · B(:,k); B(:,j) /= A(j,j) */
inline void simd_trsm_r4_rln(std::ptrdiff_t ib, std::ptrdiff_t N, T alpha,
                             const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const bool alpha_nontriv = !eq1(alpha);
    for (std::ptrdiff_t j = N - 1; j >= 0; --j) {
        T *bj = b + static_cast<std::size_t>(j) * ldb;
        __m256d bjh, bjl;
        load_dd4(bj + ib, bjh, bjl);
        if (alpha_nontriv) {
            __m256d nh, nl;
            simd_fast::mul(bjh, bjl, ah, al, nh, nl);
            bjh = nh; bjl = nl;
        }
        for (std::ptrdiff_t k = j + 1; k < N; ++k) {
            const T akj = A_(k, j);
            if (eq0(akj)) continue;
            __m256d akh = _mm256_set1_pd(akj.limbs[0]);
            __m256d akl = _mm256_set1_pd(akj.limbs[1]);
            const T *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkh, bkl;
            load_dd4(bk + ib, bkh, bkl);
            __m256d ph, pl;
            simd_fast::mul(akh, akl, bkh, bkl, ph, pl);
            simd_fast::neg(ph, pl);
            __m256d nh, nl;
            simd_fast::add(bjh, bjl, ph, pl, nh, nl);
            bjh = nh; bjl = nl;
        }
        if (nounit) {
            const T inv = one_dd / A_(j, j);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(bjh, bjl, ih, il, nh, nl);
            bjh = nh; bjl = nl;
        }
        store_dd4(bj + ib, bjh, bjl);
    }
}

inline void simd_trsm_r4_run(std::ptrdiff_t ib, std::ptrdiff_t N, T alpha,
                             const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const bool alpha_nontriv = !eq1(alpha);
    for (std::ptrdiff_t j = 0; j < N; ++j) {
        T *bj = b + static_cast<std::size_t>(j) * ldb;
        __m256d bjh, bjl;
        load_dd4(bj + ib, bjh, bjl);
        if (alpha_nontriv) {
            __m256d nh, nl;
            simd_fast::mul(bjh, bjl, ah, al, nh, nl);
            bjh = nh; bjl = nl;
        }
        for (std::ptrdiff_t k = 0; k < j; ++k) {
            const T akj = A_(k, j);
            if (eq0(akj)) continue;
            __m256d akh = _mm256_set1_pd(akj.limbs[0]);
            __m256d akl = _mm256_set1_pd(akj.limbs[1]);
            const T *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkh, bkl;
            load_dd4(bk + ib, bkh, bkl);
            __m256d ph, pl;
            simd_fast::mul(akh, akl, bkh, bkl, ph, pl);
            simd_fast::neg(ph, pl);
            __m256d nh, nl;
            simd_fast::add(bjh, bjl, ph, pl, nh, nl);
            bjh = nh; bjl = nl;
        }
        if (nounit) {
            const T inv = one_dd / A_(j, j);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(bjh, bjl, ih, il, nh, nl);
            bjh = nh; bjl = nl;
        }
        store_dd4(bj + ib, bjh, bjl);
    }
}

/* RLT: B := α·B / Lᵀ. Iterate k = 0..N-1:
 *   B(:,k) /= A(k,k) (if nounit); subtract from B(:,j) for j > k; α-scale B(:,k) */
inline void simd_trsm_r4_rlt(std::ptrdiff_t ib, std::ptrdiff_t N, T alpha,
                             const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const bool alpha_nontriv = !eq1(alpha);
    for (std::ptrdiff_t k = 0; k < N; ++k) {
        T *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkh, bkl;
        load_dd4(bk + ib, bkh, bkl);
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(bkh, bkl, ih, il, nh, nl);
            bkh = nh; bkl = nl;
            store_dd4(bk + ib, bkh, bkl);
        }
        for (std::ptrdiff_t j = k + 1; j < N; ++j) {
            const T ajk = A_(j, k);
            if (eq0(ajk)) continue;
            __m256d ajh = _mm256_set1_pd(ajk.limbs[0]);
            __m256d ajl = _mm256_set1_pd(ajk.limbs[1]);
            T *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjh, bjl;
            load_dd4(bj + ib, bjh, bjl);
            __m256d ph, pl;
            simd_fast::mul(ajh, ajl, bkh, bkl, ph, pl);
            simd_fast::neg(ph, pl);
            __m256d nh, nl;
            simd_fast::add(bjh, bjl, ph, pl, nh, nl);
            store_dd4(bj + ib, nh, nl);
        }
        if (alpha_nontriv) {
            __m256d nh, nl;
            simd_fast::mul(bkh, bkl, ah, al, nh, nl);
            store_dd4(bk + ib, nh, nl);
        }
    }
}

inline void simd_trsm_r4_rut(std::ptrdiff_t ib, std::ptrdiff_t N, T alpha,
                             const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    const bool alpha_nontriv = !eq1(alpha);
    for (std::ptrdiff_t k = N - 1; k >= 0; --k) {
        T *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkh, bkl;
        load_dd4(bk + ib, bkh, bkl);
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            __m256d ih = _mm256_set1_pd(inv.limbs[0]);
            __m256d il = _mm256_set1_pd(inv.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(bkh, bkl, ih, il, nh, nl);
            bkh = nh; bkl = nl;
            store_dd4(bk + ib, bkh, bkl);
        }
        for (std::ptrdiff_t j = 0; j < k; ++j) {
            const T ajk = A_(j, k);
            if (eq0(ajk)) continue;
            __m256d ajh = _mm256_set1_pd(ajk.limbs[0]);
            __m256d ajl = _mm256_set1_pd(ajk.limbs[1]);
            T *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjh, bjl;
            load_dd4(bj + ib, bjh, bjl);
            __m256d ph, pl;
            simd_fast::mul(ajh, ajl, bkh, bkl, ph, pl);
            simd_fast::neg(ph, pl);
            __m256d nh, nl;
            simd_fast::add(bjh, bjl, ph, pl, nh, nl);
            store_dd4(bj + ib, nh, nl);
        }
        if (alpha_nontriv) {
            __m256d nh, nl;
            simd_fast::mul(bkh, bkl, ah, al, nh, nl);
            store_dd4(bk + ib, nh, nl);
        }
    }
}

enum trsm_r_op { TRSM_RLN, TRSM_RUN, TRSM_RLT, TRSM_RUT };

inline void mtrsm_simd_diag_R(trsm_r_op op, std::ptrdiff_t M, std::ptrdiff_t N, T alpha,
                              const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    const std::ptrdiff_t M4 = M & ~3;
    for (std::ptrdiff_t ib = 0; ib < M4; ib += 4) {
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
        const std::ptrdiff_t Mt = M;
        switch (op) {
        case TRSM_RLN: {
            for (std::ptrdiff_t j = N - 1; j >= 0; --j) {
                if (!eq1(alpha)) for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) * alpha;
                for (std::ptrdiff_t k = j + 1; k < N; ++k) {
                    if (!eq0(A_(k, j))) {
                        const T akj = A_(k, j);
                        for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) - akj * B_(i, k);
                    }
                }
                if (nounit) { const T inv = one_dd / A_(j, j);
                    for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) * inv; }
            }
        } break;
        case TRSM_RUN: {
            for (std::ptrdiff_t j = 0; j < N; ++j) {
                if (!eq1(alpha)) for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) * alpha;
                for (std::ptrdiff_t k = 0; k < j; ++k) {
                    if (!eq0(A_(k, j))) {
                        const T akj = A_(k, j);
                        for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) - akj * B_(i, k);
                    }
                }
                if (nounit) { const T inv = one_dd / A_(j, j);
                    for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) * inv; }
            }
        } break;
        case TRSM_RLT: {
            for (std::ptrdiff_t k = 0; k < N; ++k) {
                if (nounit) { const T inv = one_dd / A_(k, k);
                    for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, k) = B_(i, k) * inv; }
                for (std::ptrdiff_t j = k + 1; j < N; ++j) {
                    if (!eq0(A_(j, k))) {
                        const T ajk = A_(j, k);
                        for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) - ajk * B_(i, k);
                    }
                }
                if (!eq1(alpha)) for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, k) = B_(i, k) * alpha;
            }
        } break;
        case TRSM_RUT: {
            for (std::ptrdiff_t k = N - 1; k >= 0; --k) {
                if (nounit) { const T inv = one_dd / A_(k, k);
                    for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, k) = B_(i, k) * inv; }
                for (std::ptrdiff_t j = 0; j < k; ++j) {
                    if (!eq0(A_(j, k))) {
                        const T ajk = A_(j, k);
                        for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = B_(i, j) - ajk * B_(i, k);
                    }
                }
                if (!eq1(alpha)) for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, k) = B_(i, k) * alpha;
            }
        } break;
        }
    }
}

#endif  /* MBLAS_SIMD_DD */

inline void mtrsm_rln_core(std::ptrdiff_t N, std::ptrdiff_t M, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    for (std::ptrdiff_t j = N - 1; j >= 0; --j) {
        if (!eq1(alpha)) for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, j) = B_(i, j) * alpha;
        for (std::ptrdiff_t k = j + 1; k < N; ++k) {
            if (!eq0(A_(k, j))) {
                const T akj = A_(k, j);
                for (std::ptrdiff_t i = 0; i < M; ++i)
                    B_(i, j) = B_(i, j) - akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = one_dd / A_(j, j);
            for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, j) = B_(i, j) * inv;
        }
    }
}

inline void mtrsm_run_core(std::ptrdiff_t N, std::ptrdiff_t M, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    for (std::ptrdiff_t j = 0; j < N; ++j) {
        if (!eq1(alpha)) for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, j) = B_(i, j) * alpha;
        for (std::ptrdiff_t k = 0; k < j; ++k) {
            if (!eq0(A_(k, j))) {
                const T akj = A_(k, j);
                for (std::ptrdiff_t i = 0; i < M; ++i)
                    B_(i, j) = B_(i, j) - akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = one_dd / A_(j, j);
            for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, j) = B_(i, j) * inv;
        }
    }
}

inline void mtrsm_rlt_core(std::ptrdiff_t N, std::ptrdiff_t M, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    for (std::ptrdiff_t k = 0; k < N; ++k) {
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, k) = B_(i, k) * inv;
        }
        for (std::ptrdiff_t j = k + 1; j < N; ++j) {
            if (!eq0(A_(j, k))) {
                const T ajk = A_(j, k);
                for (std::ptrdiff_t i = 0; i < M; ++i)
                    B_(i, j) = B_(i, j) - ajk * B_(i, k);
            }
        }
        if (!eq1(alpha)) for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, k) = B_(i, k) * alpha;
    }
}

inline void mtrsm_rut_core(std::ptrdiff_t N, std::ptrdiff_t M, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    for (std::ptrdiff_t k = N - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = one_dd / A_(k, k);
            for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, k) = B_(i, k) * inv;
        }
        for (std::ptrdiff_t j = 0; j < k; ++j) {
            if (!eq0(A_(j, k))) {
                const T ajk = A_(j, k);
                for (std::ptrdiff_t i = 0; i < M; ++i)
                    B_(i, j) = B_(i, j) - ajk * B_(i, k);
            }
        }
        if (!eq1(alpha)) for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, k) = B_(i, k) * alpha;
    }
}

/* ── Blocked SIDE='L' chunk worker: serial blocked-TRSM over one column slice
 * [j_start, j_end). The mgemm trailing update routes through mgemm_serial. */

inline void prescale_chunk(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, T alpha,
                           T *b, std::ptrdiff_t ldb)
{
    if (eq1(alpha)) return;
    if (eq0(alpha)) {
        for (std::ptrdiff_t j = j_start; j < j_end; ++j)
            for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, j) = zero_dd;
        return;
    }
    for (std::ptrdiff_t j = j_start; j < j_end; ++j)
        for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, j) = B_(i, j) * alpha;
}

enum trsm_variant { LLN, LUN, LLT, LUT };

void blocked_chunk(trsm_variant V, std::ptrdiff_t j_start, std::ptrdiff_t j_end,
                   std::ptrdiff_t M, std::ptrdiff_t nb, T alpha,
                   const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    const std::ptrdiff_t my_N = j_end - j_start;
    if (my_N <= 0) return;
    prescale_chunk(j_start, j_end, M, alpha, b, ldb);

    const T m_one = T{-1.0, 0.0};
    const T one   = T{ 1.0, 0.0};
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
        for (std::ptrdiff_t ic = 0; ic < M; ic += nb) {
            const std::ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                mgemm_serial('N', 'N', ib, my_N, ic, &m_one,
                             &A_(ic, 0), lda,
                             B_chunk, ldb, &one,
                             &B_chunk[ic], ldb);
            }
            DIAG_SOLVE(SLLN, mtrsm_lln_core, ib, one_dd);
        }
    } else if (V == LUN) {
        std::ptrdiff_t ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const std::ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            const std::ptrdiff_t trailing = M - (ic + ib);
            if (trailing > 0) {
                const std::ptrdiff_t j0 = ic + ib;
                mgemm_serial('N', 'N', ib, my_N, trailing, &m_one,
                             &A_(ic, j0), lda,
                             &B_chunk[j0], ldb, &one,
                             &B_chunk[ic], ldb);
            }
            DIAG_SOLVE(SLUN, mtrsm_lun_core, ib, one_dd);
            ic -= nb;
        }
    } else if (V == LLT) {
        std::ptrdiff_t ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const std::ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            const std::ptrdiff_t trailing = M - (ic + ib);
            if (trailing > 0) {
                const std::ptrdiff_t i0 = ic + ib;
                mgemm_serial('T', 'N', ib, my_N, trailing, &m_one,
                             &A_(i0, ic), lda,
                             &B_chunk[i0], ldb, &one,
                             &B_chunk[ic], ldb);
            }
            DIAG_SOLVE(SLLT, mtrsm_llt_core, ib, one_dd);
            ic -= nb;
        }
    } else { /* LUT */
        for (std::ptrdiff_t ic = 0; ic < M; ic += nb) {
            const std::ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                mgemm_serial('T', 'N', ib, my_N, ic, &m_one,
                             &A_(0, ic), lda,
                             B_chunk, ldb, &one,
                             &B_chunk[ic], ldb);
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

std::ptrdiff_t mtrsm_block_nb(void) { return trsm_nb(); }

void mtrsm_zero_B(std::ptrdiff_t M, std::ptrdiff_t N, T *b, std::ptrdiff_t ldb)
{
    for (std::ptrdiff_t j = 0; j < N; ++j)
        for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, j) = zero_dd;
}

void mtrsm_L_slice(char UPLO, char TR, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, std::ptrdiff_t nb, T alpha,
                   const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    if (j_start >= j_end) return;
    const trsm_variant V = l_variant(UPLO, TR);
    if (use_blocked) {
        blocked_chunk(V, j_start, j_end, M, nb, alpha, a, lda, b, ldb, nounit);
        return;
    }
#ifdef MBLAS_SIMD_DD
    /* Single-block (M < 2·nb) path: route through the SIMD diagonal kernel
     * rather than the scalar cores. The transpose cores (LLT/LUT) are
     * single-accumulator dot loops — latency-bound for double-double — so the
     * scalar path lost the small-M Left/Transpose cells (~1.5-1.6× vs ob); the
     * 4-column-lane SIMD substitution recovers them. M ≤ kMaxBlockM holds here
     * because use_blocked is false ⇒ M < 2·nb (= 128) ≤ kMaxBlockM (256).
     * Same kernel the blocked path already drives, so serial and parallel stay
     * bitwise-identical. */
    if (M <= kMaxBlockM) {
        static const trsm_simd_op op_of[4] = { SLLN, SLUN, SLLT, SLUT };
        mtrsm_simd_diag(op_of[V], j_start, j_end, M, alpha, a, lda, b, ldb, nounit);
        return;
    }
#endif
    switch (V) {
    case LLN: mtrsm_lln_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit); break;
    case LUN: mtrsm_lun_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit); break;
    case LLT: mtrsm_llt_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit); break;
    case LUT: mtrsm_lut_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit); break;
    }
}

void mtrsm_R_slice(char UPLO, char TR, std::ptrdiff_t row_lo, std::ptrdiff_t row_hi,
                   std::ptrdiff_t N, T alpha,
                   const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, std::ptrdiff_t nounit)
{
    const std::ptrdiff_t Mslice = row_hi - row_lo;
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
    char side, char uplo, char transa, char diag,
    std::ptrdiff_t M, std::ptrdiff_t N,
    const T *alpha_,
    const T *a, std::ptrdiff_t lda,
    T *b, std::ptrdiff_t ldb)
{
    const T alpha = *alpha_;
    using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);
    char TR = up(&transa);
    if (TR == 'C') TR = 'T';
    const std::ptrdiff_t nounit = (up(&diag) != 'U');

    if (M == 0 || N == 0) return;

    if (eq0(alpha)) { mtrsm_zero_B(M, N, b, ldb); return; }

    if (SIDE == 'L') {
        const std::ptrdiff_t nb = trsm_nb();
        const std::ptrdiff_t use_blocked = (M >= 2 * nb);
        mtrsm_L_slice(UPLO, TR, use_blocked, 0, N, M, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        mtrsm_R_slice(UPLO, TR, 0, M, N, alpha, a, lda, b, ldb, nounit);
    }
}

#undef A_
#undef B_
