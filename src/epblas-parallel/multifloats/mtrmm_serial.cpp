/*
 * mtrmm_serial.cpp — multifloats real (double-double) triangular multiply,
 * single-thread core. Owns ALL the numerics shared by the serial and parallel
 * entries:
 *
 *   - scalar column "core" kernels for SIDE='L' (LLN/LUN/LLT/LUT) and the
 *     SIDE='R' cores (RLN/RUN/RLT/RUT),
 *   - the AVX2 4-wide SIMD diagonal kernels (SIDE='L' packed-SoA and SIDE='R'
 *     4-row chunks), under MBLAS_SIMD_DD,
 *   - the block-size policy and the blocked chunk workers for BOTH sides,
 *     whose trailing-matrix update routes through mgemm_serial (no nested
 *     OpenMP),
 *   - the per-slice workers mtrmm_L_slice / mtrmm_R_slice (declared in
 *     mtrmm_kernel.h) that the parallel entry fans across a team, plus the
 *     public `mtrmm_serial` entry.
 *
 * There is NO OpenMP on this path. Threading lives entirely in
 * mtrmm_parallel.cpp; both paths drive these workers, so a static partition
 * is bitwise-identical to the serial sweep.
 */

#include "mtrmm_kernel.h"
#include "mf_pred.h"
#include "mgemm_kernel.h"   /* mgemm_serial for the trailing update */
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include "mf_util.h"
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */

#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"   /* mul, add primitives */
#include "mf_simd_exact.h"  /* canonical load_dd4 */
#include <immintrin.h>
#endif

namespace mf = multifloats;
using TR = mf::float64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::eq0;
using mf_pred::eq1;
namespace {

/* Triangular-axis block size for the blocked paths — compile-time constant
 * (nothing writes it). */
constexpr std::ptrdiff_t g_nb_trmm = 64;
std::ptrdiff_t trmm_nb(void) { return g_nb_trmm; }

using mf_pred::zero_dd;   /* shared DD constants — mf_pred.h */
using mf_pred::one_dd;


#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

#ifdef MBLAS_SIMD_DD

/* AVX2+FMA under a possibly pre-Haswell baseline -march: these SIMD kernels and
 * their helpers are compiled with the feature enabled and reached only behind
 * mf_have_avx2_fma() at the call sites below. See mf_dispatch.h. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")

constexpr std::ptrdiff_t kSimdLane = simd_fast::NR;
constexpr std::ptrdiff_t kMaxBlockM = 256;

inline void pack_B_4col(std::ptrdiff_t m, const TR *b, std::ptrdiff_t ldb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                        double *bh, double *bl)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        const TR *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            bh[i * kSimdLane + j] = col[i].limbs[0];
            bl[i * kSimdLane + j] = col[i].limbs[1];
        }
    }
    for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j)
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            bh[i * kSimdLane + j] = 0.0;
            bl[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_B_4col(std::ptrdiff_t m, TR *b, std::ptrdiff_t ldb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                          const double *bh, const double *bl)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        TR *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            col[i].limbs[0] = bh[i * kSimdLane + j];
            col[i].limbs[1] = bl[i * kSimdLane + j];
        }
    }
}

/* SIMD LLN: for k = M-1..0: temp = α·B(k); for i>k: B(i) += temp·A(i,k);
 * if nounit: temp *= A(k,k); B(k) = temp. */
inline void simd_trmm_lln(std::ptrdiff_t m, const TR *a, std::ptrdiff_t lda, TR alpha, bool nounit,
                          double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (std::ptrdiff_t k = m - 1; k >= 0; --k) {
        __m256d bkh = _mm256_load_pd(&bh[k * kSimdLane]);
        __m256d bkl = _mm256_load_pd(&bl[k * kSimdLane]);
        __m256d th, tl;
        simd_fast::mul(ah, al, bkh, bkl, th, tl);
        for (std::ptrdiff_t i = m - 1; i > k; --i) {
            const TR aik = A_(i, k);
            __m256d aih = _mm256_set1_pd(aik.limbs[0]);
            __m256d ail = _mm256_set1_pd(aik.limbs[1]);
            __m256d ph, pl;
            simd_fast::mul(th, tl, aih, ail, ph, pl);
            __m256d bih = _mm256_load_pd(&bh[i * kSimdLane]);
            __m256d bil = _mm256_load_pd(&bl[i * kSimdLane]);
            __m256d nh, nl;
            simd_fast::add(bih, bil, ph, pl, nh, nl);
            _mm256_store_pd(&bh[i * kSimdLane], nh);
            _mm256_store_pd(&bl[i * kSimdLane], nl);
        }
        if (nounit) {
            const TR akk = A_(k, k);
            __m256d akh = _mm256_set1_pd(akk.limbs[0]);
            __m256d akl = _mm256_set1_pd(akk.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(th, tl, akh, akl, nh, nl);
            th = nh; tl = nl;
        }
        _mm256_store_pd(&bh[k * kSimdLane], th);
        _mm256_store_pd(&bl[k * kSimdLane], tl);
    }
}

/* SIMD LUN: for k = 0..M: temp = α·B(k); for i<k: B(i) += temp·A(i,k);
 * if nounit: temp *= A(k,k); B(k) = temp. */
inline void simd_trmm_lun(std::ptrdiff_t m, const TR *a, std::ptrdiff_t lda, TR alpha, bool nounit,
                          double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (std::ptrdiff_t k = 0; k < m; ++k) {
        __m256d bkh = _mm256_load_pd(&bh[k * kSimdLane]);
        __m256d bkl = _mm256_load_pd(&bl[k * kSimdLane]);
        __m256d th, tl;
        simd_fast::mul(ah, al, bkh, bkl, th, tl);
        for (std::ptrdiff_t i = 0; i < k; ++i) {
            const TR aik = A_(i, k);
            __m256d aih = _mm256_set1_pd(aik.limbs[0]);
            __m256d ail = _mm256_set1_pd(aik.limbs[1]);
            __m256d ph, pl;
            simd_fast::mul(th, tl, aih, ail, ph, pl);
            __m256d bih = _mm256_load_pd(&bh[i * kSimdLane]);
            __m256d bil = _mm256_load_pd(&bl[i * kSimdLane]);
            __m256d nh, nl;
            simd_fast::add(bih, bil, ph, pl, nh, nl);
            _mm256_store_pd(&bh[i * kSimdLane], nh);
            _mm256_store_pd(&bl[i * kSimdLane], nl);
        }
        if (nounit) {
            const TR akk = A_(k, k);
            __m256d akh = _mm256_set1_pd(akk.limbs[0]);
            __m256d akl = _mm256_set1_pd(akk.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(th, tl, akh, akl, nh, nl);
            th = nh; tl = nl;
        }
        _mm256_store_pd(&bh[k * kSimdLane], th);
        _mm256_store_pd(&bl[k * kSimdLane], tl);
    }
}

/* SIMD LLT: for i = 0..M: t = B(i); if nounit: t *= A(i,i);
 * for k>i: t += A(k,i)·B(k); B(i) = alpha·t. */
inline void simd_trmm_llt(std::ptrdiff_t m, const TR *a, std::ptrdiff_t lda, TR alpha, bool nounit,
                          double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (std::ptrdiff_t i = 0; i < m; ++i) {
        __m256d th = _mm256_load_pd(&bh[i * kSimdLane]);
        __m256d tl = _mm256_load_pd(&bl[i * kSimdLane]);
        if (nounit) {
            const TR aii = A_(i, i);
            __m256d aih = _mm256_set1_pd(aii.limbs[0]);
            __m256d ail = _mm256_set1_pd(aii.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(th, tl, aih, ail, nh, nl);
            th = nh; tl = nl;
        }
        for (std::ptrdiff_t k = i + 1; k < m; ++k) {
            const TR aki = A_(k, i);
            __m256d akh = _mm256_set1_pd(aki.limbs[0]);
            __m256d akl = _mm256_set1_pd(aki.limbs[1]);
            __m256d bkh = _mm256_load_pd(&bh[k * kSimdLane]);
            __m256d bkl = _mm256_load_pd(&bl[k * kSimdLane]);
            __m256d ph, pl;
            simd_fast::mul(akh, akl, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_fast::add(th, tl, ph, pl, nh, nl);
            th = nh; tl = nl;
        }
        __m256d nh, nl;
        simd_fast::mul(ah, al, th, tl, nh, nl);
        _mm256_store_pd(&bh[i * kSimdLane], nh);
        _mm256_store_pd(&bl[i * kSimdLane], nl);
    }
}

/* SIMD LUT: for i = M-1..0: t = B(i); if nounit: t *= A(i,i);
 * for k<i: t += A(k,i)·B(k); B(i) = alpha·t. */
inline void simd_trmm_lut(std::ptrdiff_t m, const TR *a, std::ptrdiff_t lda, TR alpha, bool nounit,
                          double *bh, double *bl)
{
    const __m256d ah = _mm256_set1_pd(alpha.limbs[0]);
    const __m256d al = _mm256_set1_pd(alpha.limbs[1]);
    for (std::ptrdiff_t i = m - 1; i >= 0; --i) {
        __m256d th = _mm256_load_pd(&bh[i * kSimdLane]);
        __m256d tl = _mm256_load_pd(&bl[i * kSimdLane]);
        if (nounit) {
            const TR aii = A_(i, i);
            __m256d aih = _mm256_set1_pd(aii.limbs[0]);
            __m256d ail = _mm256_set1_pd(aii.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(th, tl, aih, ail, nh, nl);
            th = nh; tl = nl;
        }
        for (std::ptrdiff_t k = 0; k < i; ++k) {
            const TR aki = A_(k, i);
            __m256d akh = _mm256_set1_pd(aki.limbs[0]);
            __m256d akl = _mm256_set1_pd(aki.limbs[1]);
            __m256d bkh = _mm256_load_pd(&bh[k * kSimdLane]);
            __m256d bkl = _mm256_load_pd(&bl[k * kSimdLane]);
            __m256d ph, pl;
            simd_fast::mul(akh, akl, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_fast::add(th, tl, ph, pl, nh, nl);
            th = nh; tl = nl;
        }
        __m256d nh, nl;
        simd_fast::mul(ah, al, th, tl, nh, nl);
        _mm256_store_pd(&bh[i * kSimdLane], nh);
        _mm256_store_pd(&bl[i * kSimdLane], nl);
    }
}

enum trmm_simd_op { SLLN, SLUN, SLLT, SLUT };

inline void mtrmm_simd_diag(trmm_simd_op op, std::ptrdiff_t j_start, std::ptrdiff_t j_end,
                            std::ptrdiff_t m, TR alpha,
                            const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    alignas(32) double bh[kMaxBlockM * kSimdLane];
    alignas(32) double bl[kMaxBlockM * kSimdLane];
    for (std::ptrdiff_t j = j_start; j < j_end; j += kSimdLane) {
        const std::ptrdiff_t jc = (j_end - j < kSimdLane) ? (j_end - j) : kSimdLane;
        pack_B_4col(m, b, ldb, j, jc, bh, bl);
        switch (op) {
        case SLLN: simd_trmm_lln(m, a, lda, alpha, nounit, bh, bl); break;
        case SLUN: simd_trmm_lun(m, a, lda, alpha, nounit, bh, bl); break;
        case SLLT: simd_trmm_llt(m, a, lda, alpha, nounit, bh, bl); break;
        case SLUT: simd_trmm_lut(m, a, lda, alpha, nounit, bh, bl); break;
        }
        unpack_B_4col(m, b, ldb, j, jc, bh, bl);
    }
}

#pragma GCC pop_options

#endif  /* MBLAS_SIMD_DD */

inline void mtrmm_lln_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TR alpha,
                           const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t k = m - 1; k >= 0; --k) {
            if (!eq0(B_(k, j))) {
                TR temp = alpha * B_(k, j);
                for (std::ptrdiff_t i = m - 1; i > k; --i)
                    B_(i, j) = B_(i, j) + temp * A_(i, k);
                if (nounit) temp = temp * A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

inline void mtrmm_lun_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TR alpha,
                           const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t k = 0; k < m; ++k) {
            if (!eq0(B_(k, j))) {
                TR temp = alpha * B_(k, j);
                for (std::ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) = B_(i, j) + temp * A_(i, k);
                if (nounit) temp = temp * A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

inline void mtrmm_llt_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TR alpha,
                           const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            TR t = B_(i, j);
            if (nounit) t = t * A_(i, i);
            for (std::ptrdiff_t k = i + 1; k < m; ++k) t = t + A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

inline void mtrmm_lut_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TR alpha,
                           const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t i = m - 1; i >= 0; --i) {
            TR t = B_(i, j);
            if (nounit) t = t * A_(i, i);
            for (std::ptrdiff_t k = 0; k < i; ++k) t = t + A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

#ifdef MBLAS_SIMD_DD

/* Forward decls for scalar tails (defined below). These MUST be declared at the
 * baseline target — outside the push_options region — so their out-of-region
 * definitions do NOT inherit the avx2,fma target attribute (GCC binds the
 * attribute at a function's first declaration). */
inline void mtrmm_rln_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, TR, const TR*, std::ptrdiff_t, TR*, std::ptrdiff_t, bool);
inline void mtrmm_run_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, TR, const TR*, std::ptrdiff_t, TR*, std::ptrdiff_t, bool);
inline void mtrmm_rlt_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, TR, const TR*, std::ptrdiff_t, TR*, std::ptrdiff_t, bool);
inline void mtrmm_rut_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, TR, const TR*, std::ptrdiff_t, TR*, std::ptrdiff_t, bool);

/* AVX2+FMA under a possibly pre-Haswell baseline -march; see mf_dispatch.h. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")

/* AoS→SoA 4-cell transpose load: canonical simd_exact::load_dd4 (col + ofs). */
using simd_exact::load_dd4;
using simd_exact::store_dd4;

/* R-side trmm SIMD: 4-row chunks of B[i_start:i_end, 0..N).
 * For each 4-row chunk, run the trmm column-by-column with broadcast A. */
inline void simd_trmm_r4_rln(std::ptrdiff_t ib, std::ptrdiff_t n, TR alpha, const TR *a, std::ptrdiff_t lda,
                             TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        TR *bj = b + static_cast<std::size_t>(j) * ldb;
        TR t = alpha;
        if (nounit) t = t * A_(j, j);
        __m256d bjh, bjl;
        load_dd4(bj + ib, bjh, bjl);
        if (!eq1(t)) {
            __m256d th = _mm256_set1_pd(t.limbs[0]);
            __m256d tl = _mm256_set1_pd(t.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(bjh, bjl, th, tl, nh, nl);
            bjh = nh; bjl = nl;
        }
        for (std::ptrdiff_t k = j + 1; k < n; ++k) {
            const TR akj_v = A_(k, j);
            if (eq0(akj_v)) continue;
            const TR akj = alpha * akj_v;
            __m256d akh = _mm256_set1_pd(akj.limbs[0]);
            __m256d akl = _mm256_set1_pd(akj.limbs[1]);
            const TR *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkh, bkl;
            load_dd4(bk + ib, bkh, bkl);
            __m256d ph, pl;
            simd_fast::mul(akh, akl, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_fast::add(bjh, bjl, ph, pl, nh, nl);
            bjh = nh; bjl = nl;
        }
        store_dd4(bj + ib, bjh, bjl);
    }
}

inline void simd_trmm_r4_run(std::ptrdiff_t ib, std::ptrdiff_t n, TR alpha, const TR *a, std::ptrdiff_t lda,
                             TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
        TR *bj = b + static_cast<std::size_t>(j) * ldb;
        TR t = alpha;
        if (nounit) t = t * A_(j, j);
        __m256d bjh, bjl;
        load_dd4(bj + ib, bjh, bjl);
        if (!eq1(t)) {
            __m256d th = _mm256_set1_pd(t.limbs[0]);
            __m256d tl = _mm256_set1_pd(t.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(bjh, bjl, th, tl, nh, nl);
            bjh = nh; bjl = nl;
        }
        for (std::ptrdiff_t k = 0; k < j; ++k) {
            const TR akj_v = A_(k, j);
            if (eq0(akj_v)) continue;
            const TR akj = alpha * akj_v;
            __m256d akh = _mm256_set1_pd(akj.limbs[0]);
            __m256d akl = _mm256_set1_pd(akj.limbs[1]);
            const TR *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkh, bkl;
            load_dd4(bk + ib, bkh, bkl);
            __m256d ph, pl;
            simd_fast::mul(akh, akl, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_fast::add(bjh, bjl, ph, pl, nh, nl);
            bjh = nh; bjl = nl;
        }
        store_dd4(bj + ib, bjh, bjl);
    }
}

/* For RLT/RUT, the column-update order goes k=N-1..0 (RLT) or 0..N-1 (RUT);
 * for each k we update columns j != k by adding α·A(j,k)·B(:,k), then scale B(:,k)
 * by α (×A(k,k) if nounit). Need to read B(:,k) before scaling. */
inline void simd_trmm_r4_rlt(std::ptrdiff_t ib, std::ptrdiff_t n, TR alpha, const TR *a, std::ptrdiff_t lda,
                             TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t k = n - 1; k >= 0; --k) {
        TR *bk = b + static_cast<std::size_t>(k) * ldb;   /* written through below */
        __m256d bkh, bkl;
        load_dd4(bk + ib, bkh, bkl);  /* original B(:,k) before scaling */
        for (std::ptrdiff_t j = k + 1; j < n; ++j) {
            const TR ajk_v = A_(j, k);
            if (eq0(ajk_v)) continue;
            const TR ajk = alpha * ajk_v;
            __m256d ah = _mm256_set1_pd(ajk.limbs[0]);
            __m256d al = _mm256_set1_pd(ajk.limbs[1]);
            TR *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjh, bjl;
            load_dd4(bj + ib, bjh, bjl);
            __m256d ph, pl;
            simd_fast::mul(ah, al, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_fast::add(bjh, bjl, ph, pl, nh, nl);
            store_dd4(bj + ib, nh, nl);
        }
        TR t = alpha;
        if (nounit) t = t * A_(k, k);
        if (!eq1(t)) {
            __m256d th = _mm256_set1_pd(t.limbs[0]);
            __m256d tl = _mm256_set1_pd(t.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(bkh, bkl, th, tl, nh, nl);
            store_dd4(bk + ib, nh, nl);
        }
    }
}

inline void simd_trmm_r4_rut(std::ptrdiff_t ib, std::ptrdiff_t n, TR alpha, const TR *a, std::ptrdiff_t lda,
                             TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t k = 0; k < n; ++k) {
        TR *bk = b + static_cast<std::size_t>(k) * ldb;   /* written through below */
        __m256d bkh, bkl;
        load_dd4(bk + ib, bkh, bkl);
        for (std::ptrdiff_t j = 0; j < k; ++j) {
            const TR ajk_v = A_(j, k);
            if (eq0(ajk_v)) continue;
            const TR ajk = alpha * ajk_v;
            __m256d ah = _mm256_set1_pd(ajk.limbs[0]);
            __m256d al = _mm256_set1_pd(ajk.limbs[1]);
            TR *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjh, bjl;
            load_dd4(bj + ib, bjh, bjl);
            __m256d ph, pl;
            simd_fast::mul(ah, al, bkh, bkl, ph, pl);
            __m256d nh, nl;
            simd_fast::add(bjh, bjl, ph, pl, nh, nl);
            store_dd4(bj + ib, nh, nl);
        }
        TR t = alpha;
        if (nounit) t = t * A_(k, k);
        if (!eq1(t)) {
            __m256d th = _mm256_set1_pd(t.limbs[0]);
            __m256d tl = _mm256_set1_pd(t.limbs[1]);
            __m256d nh, nl;
            simd_fast::mul(bkh, bkl, th, tl, nh, nl);
            store_dd4(bk + ib, nh, nl);
        }
    }
}

enum trmm_r_op { RLN_OP, RUN_OP, RLT_OP, RUT_OP };

inline void mtrmm_simd_diag_R(trmm_r_op op, std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t n, TR alpha,
                              const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    const std::ptrdiff_t i4_end = i_start + ((i_end - i_start) & ~3);
    for (std::ptrdiff_t ib = i_start; ib < i4_end; ib += 4) {
        switch (op) {
        case RLN_OP: simd_trmm_r4_rln(ib, n, alpha, a, lda, b, ldb, nounit); break;
        case RUN_OP: simd_trmm_r4_run(ib, n, alpha, a, lda, b, ldb, nounit); break;
        case RLT_OP: simd_trmm_r4_rlt(ib, n, alpha, a, lda, b, ldb, nounit); break;
        case RUT_OP: simd_trmm_r4_rut(ib, n, alpha, a, lda, b, ldb, nounit); break;
        }
    }
    /* Scalar tail rows */
    if (i4_end < i_end) {
        switch (op) {
        case RLN_OP: mtrmm_rln_core(i4_end, i_end, n, alpha, a, lda, b, ldb, nounit); break;
        case RUN_OP: mtrmm_run_core(i4_end, i_end, n, alpha, a, lda, b, ldb, nounit); break;
        case RLT_OP: mtrmm_rlt_core(i4_end, i_end, n, alpha, a, lda, b, ldb, nounit); break;
        case RUT_OP: mtrmm_rut_core(i4_end, i_end, n, alpha, a, lda, b, ldb, nounit); break;
        }
    }
}

#pragma GCC pop_options

#endif  /* MBLAS_SIMD_DD */

inline void mtrmm_rln_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t n, TR alpha,
                           const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        TR t = alpha;
        if (nounit) t = t * A_(j, j);
        if (!eq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) = B_(i, j) * t;
        for (std::ptrdiff_t k = j + 1; k < n; ++k) {
            if (!eq0(A_(k, j))) {
                const TR akj = alpha * A_(k, j);
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = B_(i, j) + akj * B_(i, k);
            }
        }
    }
}

inline void mtrmm_run_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t n, TR alpha,
                           const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
        TR t = alpha;
        if (nounit) t = t * A_(j, j);
        if (!eq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) = B_(i, j) * t;
        for (std::ptrdiff_t k = 0; k < j; ++k) {
            if (!eq0(A_(k, j))) {
                const TR akj = alpha * A_(k, j);
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = B_(i, j) + akj * B_(i, k);
            }
        }
    }
}

inline void mtrmm_rlt_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t n, TR alpha,
                           const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t k = n - 1; k >= 0; --k) {
        for (std::ptrdiff_t j = k + 1; j < n; ++j) {
            if (!eq0(A_(j, k))) {
                const TR ajk = alpha * A_(j, k);
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = B_(i, j) + ajk * B_(i, k);
            }
        }
        TR t = alpha;
        if (nounit) t = t * A_(k, k);
        if (!eq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) = B_(i, k) * t;
    }
}

inline void mtrmm_rut_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t n, TR alpha,
                           const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t k = 0; k < n; ++k) {
        for (std::ptrdiff_t j = 0; j < k; ++j) {
            if (!eq0(A_(j, k))) {
                const TR ajk = alpha * A_(j, k);
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = B_(i, j) + ajk * B_(i, k);
            }
        }
        TR t = alpha;
        if (nounit) t = t * A_(k, k);
        if (!eq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) = B_(i, k) * t;
    }
}

/* ── Blocked SIDE='L' chunk worker: serial blocked-TRMM over one column slice
 * [j_start, j_end). The mgemm trailing update routes through mgemm_serial. */

enum trmm_variant_L { LLN, LUN, LLT, LUT };

void blocked_chunk_L(trmm_variant_L V, std::ptrdiff_t j_start, std::ptrdiff_t j_end,
                     std::ptrdiff_t m, std::ptrdiff_t nb, TR alpha,
                     const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    const std::ptrdiff_t my_N = j_end - j_start;
    if (my_N <= 0) return;

    TR *B_chunk = &B_(0, j_start);

    if (V == LLN) {
        std::ptrdiff_t ic = ((m - 1) / nb) * nb;
        while (ic >= 0) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma() && ib <= kMaxBlockM) {
                mtrmm_simd_diag(SLLN, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            mtrmm_lln_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                mgemm_serial('N', 'N', ib, my_N, ic, &alpha,
                             &A_(ic, 0), lda,
                             B_chunk, ldb, &one_dd,
                             &B_chunk[ic], ldb);
            }
            ic -= nb;
        }
    } else if (V == LUN) {
        for (std::ptrdiff_t ic = 0; ic < m; ic += nb) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma() && ib <= kMaxBlockM) {
                mtrmm_simd_diag(SLUN, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            mtrmm_lun_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const std::ptrdiff_t trailing = m - (ic + ib);
            if (trailing > 0) {
                const std::ptrdiff_t j0 = ic + ib;
                mgemm_serial('N', 'N', ib, my_N, trailing, &alpha,
                             &A_(ic, j0), lda,
                             &B_chunk[j0], ldb, &one_dd,
                             &B_chunk[ic], ldb);
            }
        }
    } else if (V == LLT) {
        for (std::ptrdiff_t ic = 0; ic < m; ic += nb) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma() && ib <= kMaxBlockM) {
                mtrmm_simd_diag(SLLT, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            mtrmm_llt_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const std::ptrdiff_t trailing = m - (ic + ib);
            if (trailing > 0) {
                const std::ptrdiff_t i0 = ic + ib;
                mgemm_serial('T', 'N', ib, my_N, trailing, &alpha,
                             &A_(i0, ic), lda,
                             &B_chunk[i0], ldb, &one_dd,
                             &B_chunk[ic], ldb);
            }
        }
    } else { /* LUT */
        std::ptrdiff_t ic = ((m - 1) / nb) * nb;
        while (ic >= 0) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma() && ib <= kMaxBlockM) {
                mtrmm_simd_diag(SLUT, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            mtrmm_lut_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                mgemm_serial('T', 'N', ib, my_N, ic, &alpha,
                             &A_(0, ic), lda,
                             B_chunk, ldb, &one_dd,
                             &B_chunk[ic], ldb);
            }
            ic -= nb;
        }
    }
}

/* ── Blocked SIDE='R' chunk worker: serial blocked-TRMM over one row slice
 * [i_start, i_end). The mgemm trailing update routes through mgemm_serial. */

enum trmm_variant_R { RLN, RUN, RLT, RUT };

void blocked_chunk_R(trmm_variant_R V, std::ptrdiff_t i_start, std::ptrdiff_t i_end,
                     std::ptrdiff_t n, std::ptrdiff_t nb, TR alpha,
                     const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    const std::ptrdiff_t my_M = i_end - i_start;
    if (my_M <= 0) return;
    TR *B_chunk = &B_(i_start, 0);

    if (V == RLN) {
        for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
            const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma()) {
                mtrmm_simd_diag_R(RLN_OP, i_start, i_end, jb, alpha,
                                  &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            } else
#endif
            mtrmm_rln_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            const std::ptrdiff_t trailing = n - (jc + jb);
            if (trailing > 0) {
                const std::ptrdiff_t k0 = jc + jb;
                mgemm_serial('N', 'N', my_M, jb, trailing, &alpha,
                             &B_chunk[static_cast<std::size_t>(k0) * ldb], ldb,
                             &A_(k0, jc), lda, &one_dd,
                             &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
        }
    } else if (V == RUN) {
        std::ptrdiff_t jc = ((n - 1) / nb) * nb;
        while (jc >= 0) {
            const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma()) {
                mtrmm_simd_diag_R(RUN_OP, i_start, i_end, jb, alpha,
                                  &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            } else
#endif
            mtrmm_run_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            if (jc > 0) {
                mgemm_serial('N', 'N', my_M, jb, jc, &alpha,
                             B_chunk, ldb,
                             &A_(0, jc), lda, &one_dd,
                             &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
            jc -= nb;
        }
    } else if (V == RLT) {
        std::ptrdiff_t jc = ((n - 1) / nb) * nb;
        while (jc >= 0) {
            const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma()) {
                mtrmm_simd_diag_R(RLT_OP, i_start, i_end, jb, alpha,
                                  &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            } else
#endif
            mtrmm_rlt_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            if (jc > 0) {
                mgemm_serial('N', 'T', my_M, jb, jc, &alpha,
                             B_chunk, ldb,
                             &A_(jc, 0), lda, &one_dd,
                             &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
            jc -= nb;
        }
    } else { /* RUT */
        for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
            const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma()) {
                mtrmm_simd_diag_R(RUT_OP, i_start, i_end, jb, alpha,
                                  &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            } else
#endif
            mtrmm_rut_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            const std::ptrdiff_t trailing = n - (jc + jb);
            if (trailing > 0) {
                const std::ptrdiff_t k0 = jc + jb;
                mgemm_serial('N', 'T', my_M, jb, trailing, &alpha,
                             &B_chunk[static_cast<std::size_t>(k0) * ldb], ldb,
                             &A_(jc, k0), lda, &one_dd,
                             &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
        }
    }
}

/* Map (UPLO, TRANS) → blocked variant. */
inline trmm_variant_L l_variant(char UPLO, char TRANS) {
    if (TRANS == 'N') return (UPLO == 'L') ? LLN : LUN;
    return (UPLO == 'L') ? LLT : LUT;
}
inline trmm_variant_R r_variant(char UPLO, char TRANS) {
    if (TRANS == 'N') return (UPLO == 'L') ? RLN : RUN;
    return (UPLO == 'L') ? RLT : RUT;
}

}  // namespace

/* ── Exposed surface (mtrmm_kernel.h). ─────────────────────────────────── */

std::ptrdiff_t mtrmm_block_nb(void) { return trmm_nb(); }

void mtrmm_zero_B(std::ptrdiff_t m, std::ptrdiff_t n, TR *b, std::ptrdiff_t ldb)
{
    for (std::ptrdiff_t j = 0; j < n; ++j)
        for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = zero_dd;
}

void mtrmm_L_slice(char UPLO, char TRANS, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, std::ptrdiff_t nb, TR alpha,
                   const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    if (j_start >= j_end) return;
    const trmm_variant_L V = l_variant(UPLO, TRANS);
    if (use_blocked) {
        blocked_chunk_L(V, j_start, j_end, m, nb, alpha, a, lda, b, ldb, nounit);
        return;
    }
#ifdef MBLAS_SIMD_DD
    /* Small-M (single-block) regime: route the unblocked path through the SIMD
     * diag too — same kernel, over this slice's column range. Runtime-gated on
     * AVX2/FMA; the scalar switch below is always compiled. */
    if (mf_have_avx2_fma() && m <= kMaxBlockM) {
        trmm_simd_op op;
        if (TRANS == 'N') op = (UPLO == 'L') ? SLLN : SLUN;
        else           op = (UPLO == 'L') ? SLLT : SLUT;
        mtrmm_simd_diag(op, j_start, j_end, m, alpha, a, lda, b, ldb, nounit);
        return;
    }
#endif
    switch (V) {
    case LLN: mtrmm_lln_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit); break;
    case LUN: mtrmm_lun_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit); break;
    case LLT: mtrmm_llt_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit); break;
    case LUT: mtrmm_lut_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit); break;
    }
}

void mtrmm_R_slice(char UPLO, char TRANS, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t row_lo, std::ptrdiff_t row_hi, std::ptrdiff_t n, std::ptrdiff_t nb, TR alpha,
                   const TR *a, std::ptrdiff_t lda, TR *b, std::ptrdiff_t ldb, bool nounit)
{
    if (row_lo >= row_hi) return;
    const trmm_variant_R V = r_variant(UPLO, TRANS);
    if (use_blocked) {
        blocked_chunk_R(V, row_lo, row_hi, n, nb, alpha, a, lda, b, ldb, nounit);
        return;
    }
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) {
        trmm_r_op op;
        if (TRANS == 'N') op = (UPLO == 'L') ? RLN_OP : RUN_OP;
        else           op = (UPLO == 'L') ? RLT_OP : RUT_OP;
        mtrmm_simd_diag_R(op, row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit);
        return;
    }
#endif
    switch (V) {
    case RLN: mtrmm_rln_core(row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit); break;
    case RUN: mtrmm_run_core(row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit); break;
    case RLT: mtrmm_rlt_core(row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit); break;
    case RUT: mtrmm_rut_core(row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit); break;
    }
}

extern "C" void mtrmm_serial(
    char side, char uplo, char transa, char diag,
    std::ptrdiff_t m, std::ptrdiff_t n,
    const TR *alpha_,
    const TR *a, std::ptrdiff_t lda,
    TR *b, std::ptrdiff_t ldb)
{
    const TR alpha = *alpha_;
    using mf_util::up;  /* char flag uppercase — mf_util.h */
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);
    char TRANS = up(&transa);
    if (TRANS == 'C') TRANS = 'T';   /* real DD: conj-trans ≡ trans */
    const bool nounit = (up(&diag) != 'U');

    if (m == 0 || n == 0) return;

    if (eq0(alpha)) { mtrmm_zero_B(m, n, b, ldb); return; }

    const std::ptrdiff_t nb = trmm_nb();

    if (SIDE == 'L') {
        const std::ptrdiff_t use_blocked = (m >= 2 * nb);
        mtrmm_L_slice(UPLO, TRANS, use_blocked, 0, n, m, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        const std::ptrdiff_t use_blocked = (n >= 2 * nb);
        mtrmm_R_slice(UPLO, TRANS, use_blocked, 0, m, n, nb, alpha,
                      a, lda, b, ldb, nounit);
    }
}

#undef A_
#undef B_
