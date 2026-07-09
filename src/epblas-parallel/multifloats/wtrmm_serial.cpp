/*
 * wtrmm_serial.cpp — multifloats complex (complex64x2) triangular multiply,
 * single-thread core. Owns ALL the numerics shared by the serial and parallel
 * entries:
 *
 *   - scalar column "core" kernels for SIDE='L' (LLN/LUN + the conj-aware
 *     LLTC/LUTC) and the SIDE='R' cores (RLN/RUN + RLTC/RUTC),
 *   - the AVX2 4-wide SIMD diagonal kernels (SIDE='L' packed-SoA and SIDE='R'
 *     4-row chunks), under MBLAS_SIMD_DD,
 *   - the block-size policy and the blocked chunk workers for BOTH sides,
 *     whose trailing-matrix update routes through wgemm_serial (no nested
 *     OpenMP),
 *   - the per-slice workers wtrmm_L_slice / wtrmm_R_slice (declared in
 *     wtrmm_kernel.h) that the parallel entry fans across a team, plus the
 *     public `wtrmm_serial` entry.
 *
 * There is NO OpenMP on this path. Threading lives entirely in
 * wtrmm_parallel.cpp; both paths drive these workers, so a static partition
 * is bitwise-identical to the serial sweep.
 *
 * Unlike the real (mtrmm) twin, TRANSA is kept as 'N'/'T'/'C' DISTINCT — for
 * complex, the conjugate transpose differs from the plain transpose; the slice
 * workers take TRANS verbatim and map it (with UPLO) to the 6-way variant. The
 * conjugate is threaded through the T/C cores via a conj_flag + A_op().
 */

#include "wtrmm_kernel.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#include "wgemm_kernel.h"   /* wgemm_serial for the trailing update */
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include "mf_util.h"
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */

#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"   /* cmul, cadd primitives */
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;
namespace {

std::ptrdiff_t g_nb_trmm = 0;
std::ptrdiff_t trmm_nb(void) {
    if (g_nb_trmm == 0) g_nb_trmm = 64;
    return g_nb_trmm;
}

const TC zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
const TC one_cdd { R{1.0, 0.0}, R{0.0, 0.0} };


using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]

inline TC A_op(const TC *a, std::ptrdiff_t lda, std::ptrdiff_t row, std::ptrdiff_t col, bool conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

#ifdef MBLAS_SIMD_DD

/* AVX2+FMA under a possibly pre-Haswell baseline -march: these SIMD kernels and
 * their helpers are compiled with the feature enabled and reached only behind
 * mf_have_avx2_fma() at the call sites below. See mf_dispatch.h. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")

constexpr std::ptrdiff_t kSimdLane = simd_fast::NR;
constexpr std::ptrdiff_t kMaxBlockM = 128;

inline void pack_B_4col_cdd(std::ptrdiff_t m, const TC *b, std::ptrdiff_t ldb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                            double *rh, double *rl, double *ih, double *il)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        const TC *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            rh[i * kSimdLane + j] = col[i].re.limbs[0];
            rl[i * kSimdLane + j] = col[i].re.limbs[1];
            ih[i * kSimdLane + j] = col[i].im.limbs[0];
            il[i * kSimdLane + j] = col[i].im.limbs[1];
        }
    }
    for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j)
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            rh[i * kSimdLane + j] = 0.0; rl[i * kSimdLane + j] = 0.0;
            ih[i * kSimdLane + j] = 0.0; il[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_B_4col_cdd(std::ptrdiff_t m, TC *b, std::ptrdiff_t ldb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                              const double *rh, const double *rl,
                              const double *ih, const double *il)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        TC *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            col[i].re.limbs[0] = rh[i * kSimdLane + j];
            col[i].re.limbs[1] = rl[i * kSimdLane + j];
            col[i].im.limbs[0] = ih[i * kSimdLane + j];
            col[i].im.limbs[1] = il[i * kSimdLane + j];
        }
    }
}

/* Broadcast A(r,c) with optional conjugate (negate im if conj_flag). */
inline void broadcast_A_cdd(const TC *a, std::ptrdiff_t lda, std::ptrdiff_t r, std::ptrdiff_t c, bool conj_flag,
                            __m256d &rh, __m256d &rl, __m256d &ih, __m256d &il)
{
    rh = _mm256_set1_pd(A_(r, c).re.limbs[0]);
    rl = _mm256_set1_pd(A_(r, c).re.limbs[1]);
    if (conj_flag) {
        ih = _mm256_set1_pd(-A_(r, c).im.limbs[0]);
        il = _mm256_set1_pd(-A_(r, c).im.limbs[1]);
    } else {
        ih = _mm256_set1_pd(A_(r, c).im.limbs[0]);
        il = _mm256_set1_pd(A_(r, c).im.limbs[1]);
    }
}

inline void simd_wtrmm_lln(std::ptrdiff_t m, const TC *a, std::ptrdiff_t lda, TC alpha, bool nounit,
                           double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (std::ptrdiff_t k = m - 1; k >= 0; --k) {
        __m256d bkrh = _mm256_load_pd(&brh[k * kSimdLane]);
        __m256d bkrl = _mm256_load_pd(&brl[k * kSimdLane]);
        __m256d bkih = _mm256_load_pd(&bih[k * kSimdLane]);
        __m256d bkil = _mm256_load_pd(&bil[k * kSimdLane]);
        __m256d trh, trl, tih, til;
        simd_fast::cmul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                         trh, trl, tih, til);
        for (std::ptrdiff_t i = m - 1; i > k; --i) {
            __m256d akrh, akrl, akih, akil;
            broadcast_A_cdd(a, lda, i, k, 0, akrh, akrl, akih, akil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(trh, trl, tih, til, akrh, akrl, akih, akil,
                             prh, prl, pih, pil);
            __m256d birh = _mm256_load_pd(&brh[i * kSimdLane]);
            __m256d birl = _mm256_load_pd(&brl[i * kSimdLane]);
            __m256d biih = _mm256_load_pd(&bih[i * kSimdLane]);
            __m256d biil = _mm256_load_pd(&bil[i * kSimdLane]);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(birh, birl, biih, biil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            _mm256_store_pd(&brh[i * kSimdLane], nrh);
            _mm256_store_pd(&brl[i * kSimdLane], nrl);
            _mm256_store_pd(&bih[i * kSimdLane], nih);
            _mm256_store_pd(&bil[i * kSimdLane], nil_);
        }
        if (nounit) {
            __m256d akrh, akrl, akih, akil;
            broadcast_A_cdd(a, lda, k, k, 0, akrh, akrl, akih, akil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(trh, trl, tih, til, akrh, akrl, akih, akil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        _mm256_store_pd(&brh[k * kSimdLane], trh);
        _mm256_store_pd(&brl[k * kSimdLane], trl);
        _mm256_store_pd(&bih[k * kSimdLane], tih);
        _mm256_store_pd(&bil[k * kSimdLane], til);
    }
}

inline void simd_wtrmm_lun(std::ptrdiff_t m, const TC *a, std::ptrdiff_t lda, TC alpha, bool nounit,
                           double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (std::ptrdiff_t k = 0; k < m; ++k) {
        __m256d bkrh = _mm256_load_pd(&brh[k * kSimdLane]);
        __m256d bkrl = _mm256_load_pd(&brl[k * kSimdLane]);
        __m256d bkih = _mm256_load_pd(&bih[k * kSimdLane]);
        __m256d bkil = _mm256_load_pd(&bil[k * kSimdLane]);
        __m256d trh, trl, tih, til;
        simd_fast::cmul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                         trh, trl, tih, til);
        for (std::ptrdiff_t i = 0; i < k; ++i) {
            __m256d akrh, akrl, akih, akil;
            broadcast_A_cdd(a, lda, i, k, 0, akrh, akrl, akih, akil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(trh, trl, tih, til, akrh, akrl, akih, akil,
                             prh, prl, pih, pil);
            __m256d birh = _mm256_load_pd(&brh[i * kSimdLane]);
            __m256d birl = _mm256_load_pd(&brl[i * kSimdLane]);
            __m256d biih = _mm256_load_pd(&bih[i * kSimdLane]);
            __m256d biil = _mm256_load_pd(&bil[i * kSimdLane]);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(birh, birl, biih, biil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            _mm256_store_pd(&brh[i * kSimdLane], nrh);
            _mm256_store_pd(&brl[i * kSimdLane], nrl);
            _mm256_store_pd(&bih[i * kSimdLane], nih);
            _mm256_store_pd(&bil[i * kSimdLane], nil_);
        }
        if (nounit) {
            __m256d akrh, akrl, akih, akil;
            broadcast_A_cdd(a, lda, k, k, 0, akrh, akrl, akih, akil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(trh, trl, tih, til, akrh, akrl, akih, akil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        _mm256_store_pd(&brh[k * kSimdLane], trh);
        _mm256_store_pd(&brl[k * kSimdLane], trl);
        _mm256_store_pd(&bih[k * kSimdLane], tih);
        _mm256_store_pd(&bil[k * kSimdLane], til);
    }
}

/* LL T/C: for i = 0..M-1: t = B(i); if nounit: t *= A_op(i,i);
 * for k>i: t += A_op(k,i) · B(k); B(i) = alpha · t. */
inline void simd_wtrmm_llTC(std::ptrdiff_t m, const TC *a, std::ptrdiff_t lda, TC alpha, bool nounit,
                            bool conj_flag,
                            double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (std::ptrdiff_t i = 0; i < m; ++i) {
        __m256d trh = _mm256_load_pd(&brh[i * kSimdLane]);
        __m256d trl = _mm256_load_pd(&brl[i * kSimdLane]);
        __m256d tih = _mm256_load_pd(&bih[i * kSimdLane]);
        __m256d til = _mm256_load_pd(&bil[i * kSimdLane]);
        if (nounit) {
            __m256d aiih, aiil, aiiih, aiiil;
            broadcast_A_cdd(a, lda, i, i, conj_flag, aiih, aiil, aiiih, aiiil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(trh, trl, tih, til, aiih, aiil, aiiih, aiiil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        for (std::ptrdiff_t k = i + 1; k < m; ++k) {
            __m256d akrh, akrl, akih, akil;
            broadcast_A_cdd(a, lda, k, i, conj_flag, akrh, akrl, akih, akil);
            __m256d bkrh = _mm256_load_pd(&brh[k * kSimdLane]);
            __m256d bkrl = _mm256_load_pd(&brl[k * kSimdLane]);
            __m256d bkih = _mm256_load_pd(&bih[k * kSimdLane]);
            __m256d bkil = _mm256_load_pd(&bil[k * kSimdLane]);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(akrh, akrl, akih, akil, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(trh, trl, tih, til, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        __m256d nrh, nrl, nih, nil_;
        simd_fast::cmul(arh, arl, aih, ail, trh, trl, tih, til,
                         nrh, nrl, nih, nil_);
        _mm256_store_pd(&brh[i * kSimdLane], nrh);
        _mm256_store_pd(&brl[i * kSimdLane], nrl);
        _mm256_store_pd(&bih[i * kSimdLane], nih);
        _mm256_store_pd(&bil[i * kSimdLane], nil_);
    }
}

/* LU T/C: for i = M-1..0: t = B(i); if nounit: t *= A_op(i,i);
 * for k<i: t += A_op(k,i) · B(k); B(i) = alpha · t. */
inline void simd_wtrmm_luTC(std::ptrdiff_t m, const TC *a, std::ptrdiff_t lda, TC alpha, bool nounit,
                            bool conj_flag,
                            double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (std::ptrdiff_t i = m - 1; i >= 0; --i) {
        __m256d trh = _mm256_load_pd(&brh[i * kSimdLane]);
        __m256d trl = _mm256_load_pd(&brl[i * kSimdLane]);
        __m256d tih = _mm256_load_pd(&bih[i * kSimdLane]);
        __m256d til = _mm256_load_pd(&bil[i * kSimdLane]);
        if (nounit) {
            __m256d aiih, aiil, aiiih, aiiil;
            broadcast_A_cdd(a, lda, i, i, conj_flag, aiih, aiil, aiiih, aiiil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(trh, trl, tih, til, aiih, aiil, aiiih, aiiil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        for (std::ptrdiff_t k = 0; k < i; ++k) {
            __m256d akrh, akrl, akih, akil;
            broadcast_A_cdd(a, lda, k, i, conj_flag, akrh, akrl, akih, akil);
            __m256d bkrh = _mm256_load_pd(&brh[k * kSimdLane]);
            __m256d bkrl = _mm256_load_pd(&brl[k * kSimdLane]);
            __m256d bkih = _mm256_load_pd(&bih[k * kSimdLane]);
            __m256d bkil = _mm256_load_pd(&bil[k * kSimdLane]);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(akrh, akrl, akih, akil, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(trh, trl, tih, til, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        __m256d nrh, nrl, nih, nil_;
        simd_fast::cmul(arh, arl, aih, ail, trh, trl, tih, til,
                         nrh, nrl, nih, nil_);
        _mm256_store_pd(&brh[i * kSimdLane], nrh);
        _mm256_store_pd(&brl[i * kSimdLane], nrl);
        _mm256_store_pd(&bih[i * kSimdLane], nih);
        _mm256_store_pd(&bil[i * kSimdLane], nil_);
    }
}

enum trmm_simd_op_w { WSLLN, WSLUN, WSLLT, WSLUT, WSLLC, WSLUC };

inline void wtrmm_simd_diag(trmm_simd_op_w op, std::ptrdiff_t j_start, std::ptrdiff_t j_end,
                            std::ptrdiff_t m, TC alpha,
                            const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    alignas(32) double brh[kMaxBlockM * kSimdLane];
    alignas(32) double brl[kMaxBlockM * kSimdLane];
    alignas(32) double bih[kMaxBlockM * kSimdLane];
    alignas(32) double bil[kMaxBlockM * kSimdLane];
    for (std::ptrdiff_t j = j_start; j < j_end; j += kSimdLane) {
        const std::ptrdiff_t jc = (j_end - j < kSimdLane) ? (j_end - j) : kSimdLane;
        pack_B_4col_cdd(m, b, ldb, j, jc, brh, brl, bih, bil);
        switch (op) {
        case WSLLN: simd_wtrmm_lln(m, a, lda, alpha, nounit, brh, brl, bih, bil); break;
        case WSLUN: simd_wtrmm_lun(m, a, lda, alpha, nounit, brh, brl, bih, bil); break;
        case WSLLT: simd_wtrmm_llTC(m, a, lda, alpha, nounit, 0, brh, brl, bih, bil); break;
        case WSLUT: simd_wtrmm_luTC(m, a, lda, alpha, nounit, 0, brh, brl, bih, bil); break;
        case WSLLC: simd_wtrmm_llTC(m, a, lda, alpha, nounit, 1, brh, brl, bih, bil); break;
        case WSLUC: simd_wtrmm_luTC(m, a, lda, alpha, nounit, 1, brh, brl, bih, bil); break;
        }
        unpack_B_4col_cdd(m, b, ldb, j, jc, brh, brl, bih, bil);
    }
}

#pragma GCC pop_options

#endif  /* MBLAS_SIMD_DD */

inline void wtrmm_lln_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TC alpha,
                           const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t k = m - 1; k >= 0; --k) {
            if (!ceq0(B_(k, j))) {
                TC temp = cmul(alpha, B_(k, j));
                for (std::ptrdiff_t i = m - 1; i > k; --i)
                    B_(i, j) = cadd(B_(i, j), cmul(temp, A_(i, k)));
                if (nounit) temp = cmul(temp, A_(k, k));
                B_(k, j) = temp;
            }
        }
    }
}

inline void wtrmm_lun_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TC alpha,
                           const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t k = 0; k < m; ++k) {
            if (!ceq0(B_(k, j))) {
                TC temp = cmul(alpha, B_(k, j));
                for (std::ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(temp, A_(i, k)));
                if (nounit) temp = cmul(temp, A_(k, k));
                B_(k, j) = temp;
            }
        }
    }
}

inline void wtrmm_lltc_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TC alpha,
                            const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            TC t = B_(i, j);
            if (nounit) t = cmul(t, A_op(a, lda, i, i, conj_flag));
            for (std::ptrdiff_t k = i + 1; k < m; ++k)
                t = cadd(t, cmul(A_op(a, lda, k, i, conj_flag), B_(k, j)));
            B_(i, j) = cmul(alpha, t);
        }
    }
}

inline void wtrmm_lutc_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TC alpha,
                            const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t i = m - 1; i >= 0; --i) {
            TC t = B_(i, j);
            if (nounit) t = cmul(t, A_op(a, lda, i, i, conj_flag));
            for (std::ptrdiff_t k = 0; k < i; ++k)
                t = cadd(t, cmul(A_op(a, lda, k, i, conj_flag), B_(k, j)));
            B_(i, j) = cmul(alpha, t);
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

#ifdef MBLAS_SIMD_DD

/* Forward decls for scalar tails (defined below). These MUST be declared at the
 * baseline target — outside the push_options region — so their out-of-region
 * definitions do NOT inherit the avx2,fma target attribute (GCC binds the
 * attribute at a function's first declaration). */
inline void wtrmm_rln_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, TC, const TC*, std::ptrdiff_t, TC*, std::ptrdiff_t, bool);
inline void wtrmm_run_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, TC, const TC*, std::ptrdiff_t, TC*, std::ptrdiff_t, bool);
inline void wtrmm_rltc_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, TC, const TC*, std::ptrdiff_t, TC*, std::ptrdiff_t, bool, bool);
inline void wtrmm_rutc_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, TC, const TC*, std::ptrdiff_t, TC*, std::ptrdiff_t, bool, bool);

/* AVX2+FMA under a possibly pre-Haswell baseline -march; see mf_dispatch.h. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")

using simd_exact::cload4;
using simd_exact::cstore4;

using simd_exact::vbcast;

inline void simd_wtrmm_r4_rln(std::ptrdiff_t ib, std::ptrdiff_t n, TC alpha,
                              const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        TC *bj = b + static_cast<std::size_t>(j) * ldb;
        TC t = alpha;
        if (nounit) t = cmul(t, A_(j, j));
        __m256d brh, brl, bih, bil;
        cload4(bj + ib, brh, brl, bih, bil);
        if (!ceq1(t)) {
            __m256d trh, trl, tih, til;
            vbcast(t, trh, trl, tih, til);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(brh, brl, bih, bil, trh, trl, tih, til,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        for (std::ptrdiff_t k = j + 1; k < n; ++k) {
            const TC akj_v = A_(k, j);
            if (ceq0(akj_v)) continue;
            const TC akj = cmul(alpha, akj_v);
            __m256d arh, arl, aih, ail;
            vbcast(akj, arh, arl, aih, ail);
            const TC *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkrh, bkrl, bkih, bkil;
            cload4(bk + ib, bkrh, bkrl, bkih, bkil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(brh, brl, bih, bil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        cstore4(bj + ib, brh, brl, bih, bil);
    }
}

inline void simd_wtrmm_r4_run(std::ptrdiff_t ib, std::ptrdiff_t n, TC alpha,
                              const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
        TC *bj = b + static_cast<std::size_t>(j) * ldb;
        TC t = alpha;
        if (nounit) t = cmul(t, A_(j, j));
        __m256d brh, brl, bih, bil;
        cload4(bj + ib, brh, brl, bih, bil);
        if (!ceq1(t)) {
            __m256d trh, trl, tih, til;
            vbcast(t, trh, trl, tih, til);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(brh, brl, bih, bil, trh, trl, tih, til,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        for (std::ptrdiff_t k = 0; k < j; ++k) {
            const TC akj_v = A_(k, j);
            if (ceq0(akj_v)) continue;
            const TC akj = cmul(alpha, akj_v);
            __m256d arh, arl, aih, ail;
            vbcast(akj, arh, arl, aih, ail);
            const TC *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkrh, bkrl, bkih, bkil;
            cload4(bk + ib, bkrh, bkrl, bkih, bkil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(brh, brl, bih, bil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        cstore4(bj + ib, brh, brl, bih, bil);
    }
}

inline void simd_wtrmm_r4_rlTC(std::ptrdiff_t ib, std::ptrdiff_t n, TC alpha, bool conj_flag,
                               const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t k = n - 1; k >= 0; --k) {
        const TC *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkrh, bkrl, bkih, bkil;
        cload4(bk + ib, bkrh, bkrl, bkih, bkil);
        for (std::ptrdiff_t j = k + 1; j < n; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (ceq0(ajk)) continue;
            const TC scaled = cmul(alpha, ajk);
            __m256d arh, arl, aih, ail;
            vbcast(scaled, arh, arl, aih, ail);
            TC *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjrh, bjrl, bjih, bjil;
            cload4(bj + ib, bjrh, bjrl, bjih, bjil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(bjrh, bjrl, bjih, bjil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            cstore4(bj + ib, nrh, nrl, nih, nil_);
        }
        TC t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!ceq1(t)) {
            __m256d trh, trl, tih, til;
            vbcast(t, trh, trl, tih, til);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(bkrh, bkrl, bkih, bkil, trh, trl, tih, til,
                             nrh, nrl, nih, nil_);
            cstore4(const_cast<TC*>(bk) + ib, nrh, nrl, nih, nil_);
        }
    }
}

inline void simd_wtrmm_r4_ruTC(std::ptrdiff_t ib, std::ptrdiff_t n, TC alpha, bool conj_flag,
                               const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t k = 0; k < n; ++k) {
        const TC *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkrh, bkrl, bkih, bkil;
        cload4(bk + ib, bkrh, bkrl, bkih, bkil);
        for (std::ptrdiff_t j = 0; j < k; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (ceq0(ajk)) continue;
            const TC scaled = cmul(alpha, ajk);
            __m256d arh, arl, aih, ail;
            vbcast(scaled, arh, arl, aih, ail);
            TC *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjrh, bjrl, bjih, bjil;
            cload4(bj + ib, bjrh, bjrl, bjih, bjil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(bjrh, bjrl, bjih, bjil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            cstore4(bj + ib, nrh, nrl, nih, nil_);
        }
        TC t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!ceq1(t)) {
            __m256d trh, trl, tih, til;
            vbcast(t, trh, trl, tih, til);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(bkrh, bkrl, bkih, bkil, trh, trl, tih, til,
                             nrh, nrl, nih, nil_);
            cstore4(const_cast<TC*>(bk) + ib, nrh, nrl, nih, nil_);
        }
    }
}

enum wtrmm_r_op { WRLN_OP, WRUN_OP, WRLT_OP, WRUT_OP, WRLC_OP, WRUC_OP };

inline void wtrmm_simd_diag_R(wtrmm_r_op op, std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t n, TC alpha,
                              const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    const std::ptrdiff_t i4_end = i_start + ((i_end - i_start) & ~3);
    for (std::ptrdiff_t ib = i_start; ib < i4_end; ib += 4) {
        switch (op) {
        case WRLN_OP: simd_wtrmm_r4_rln(ib, n, alpha, a, lda, b, ldb, nounit); break;
        case WRUN_OP: simd_wtrmm_r4_run(ib, n, alpha, a, lda, b, ldb, nounit); break;
        case WRLT_OP: simd_wtrmm_r4_rlTC(ib, n, alpha, 0, a, lda, b, ldb, nounit); break;
        case WRUT_OP: simd_wtrmm_r4_ruTC(ib, n, alpha, 0, a, lda, b, ldb, nounit); break;
        case WRLC_OP: simd_wtrmm_r4_rlTC(ib, n, alpha, 1, a, lda, b, ldb, nounit); break;
        case WRUC_OP: simd_wtrmm_r4_ruTC(ib, n, alpha, 1, a, lda, b, ldb, nounit); break;
        }
    }
    /* Scalar tail rows */
    if (i4_end < i_end) {
        switch (op) {
        case WRLN_OP: wtrmm_rln_core(i4_end, i_end, n, alpha, a, lda, b, ldb, nounit); break;
        case WRUN_OP: wtrmm_run_core(i4_end, i_end, n, alpha, a, lda, b, ldb, nounit); break;
        case WRLT_OP: wtrmm_rltc_core(i4_end, i_end, n, alpha, a, lda, b, ldb, nounit, 0); break;
        case WRUT_OP: wtrmm_rutc_core(i4_end, i_end, n, alpha, a, lda, b, ldb, nounit, 0); break;
        case WRLC_OP: wtrmm_rltc_core(i4_end, i_end, n, alpha, a, lda, b, ldb, nounit, 1); break;
        case WRUC_OP: wtrmm_rutc_core(i4_end, i_end, n, alpha, a, lda, b, ldb, nounit, 1); break;
        }
    }
}

#pragma GCC pop_options

#endif  /* MBLAS_SIMD_DD */

inline void wtrmm_rln_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t n, TC alpha,
                           const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        TC t = alpha;
        if (nounit) t = cmul(t, A_(j, j));
        if (!ceq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) = cmul(B_(i, j), t);
        for (std::ptrdiff_t k = j + 1; k < n; ++k) {
            if (!ceq0(A_(k, j))) {
                const TC akj = cmul(alpha, A_(k, j));
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(akj, B_(i, k)));
            }
        }
    }
}

inline void wtrmm_run_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t n, TC alpha,
                           const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
        TC t = alpha;
        if (nounit) t = cmul(t, A_(j, j));
        if (!ceq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) = cmul(B_(i, j), t);
        for (std::ptrdiff_t k = 0; k < j; ++k) {
            if (!ceq0(A_(k, j))) {
                const TC akj = cmul(alpha, A_(k, j));
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(akj, B_(i, k)));
            }
        }
    }
}

inline void wtrmm_rltc_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t n, TC alpha,
                            const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t k = n - 1; k >= 0; --k) {
        for (std::ptrdiff_t j = k + 1; j < n; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (!ceq0(ajk)) {
                const TC scaled = cmul(alpha, ajk);
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(scaled, B_(i, k)));
            }
        }
        TC t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!ceq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) = cmul(B_(i, k), t);
    }
}

inline void wtrmm_rutc_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t n, TC alpha,
                            const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t k = 0; k < n; ++k) {
        for (std::ptrdiff_t j = 0; j < k; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (!ceq0(ajk)) {
                const TC scaled = cmul(alpha, ajk);
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(scaled, B_(i, k)));
            }
        }
        TC t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!ceq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) = cmul(B_(i, k), t);
    }
}

/* ── Blocked SIDE='L' chunk worker: serial blocked-TRMM over one column slice
 * [j_start, j_end). The wgemm trailing update routes through wgemm_serial. */

enum trmm_variant_L { WLLN, WLUN, WLLT, WLUT, WLLC, WLUC };

void blocked_chunk_L(trmm_variant_L V, std::ptrdiff_t j_start, std::ptrdiff_t j_end,
                     std::ptrdiff_t m, std::ptrdiff_t nb, TC alpha,
                     const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    const std::ptrdiff_t my_N = j_end - j_start;
    if (my_N <= 0) return;

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    TC *B_chunk = &B_(0, j_start);

    if (V == WLLN) {
        std::ptrdiff_t ic = ((m - 1) / nb) * nb;
        while (ic >= 0) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma() && ib <= kMaxBlockM) {
                wtrmm_simd_diag(WSLLN, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_lln_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                wgemm_serial(NN[0], NN[0], ib, my_N, ic, &alpha, &A_(ic, 0), lda, B_chunk, ldb, &one_cdd, &B_chunk[ic], ldb);
            }
            ic -= nb;
        }
    } else if (V == WLUN) {
        for (std::ptrdiff_t ic = 0; ic < m; ic += nb) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma() && ib <= kMaxBlockM) {
                wtrmm_simd_diag(WSLUN, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_lun_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const std::ptrdiff_t trailing = m - (ic + ib);
            if (trailing > 0) {
                const std::ptrdiff_t j0 = ic + ib;
                wgemm_serial(NN[0], NN[0], ib, my_N, trailing, &alpha, &A_(ic, j0), lda, &B_chunk[j0], ldb, &one_cdd, &B_chunk[ic], ldb);
            }
        }
    } else if (V == WLLT || V == WLLC) {
        const bool conj_flag = (V == WLLC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        for (std::ptrdiff_t ic = 0; ic < m; ic += nb) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma() && ib <= kMaxBlockM) {
                wtrmm_simd_diag(conj_flag ? WSLLC : WSLLT, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_lltc_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            const std::ptrdiff_t trailing = m - (ic + ib);
            if (trailing > 0) {
                const std::ptrdiff_t i0 = ic + ib;
                wgemm_serial(gemm_trans[0], NN[0], ib, my_N, trailing, &alpha, &A_(i0, ic), lda, &B_chunk[i0], ldb, &one_cdd, &B_chunk[ic], ldb);
            }
        }
    } else { /* WLUT or WLUC */
        const bool conj_flag = (V == WLUC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        std::ptrdiff_t ic = ((m - 1) / nb) * nb;
        while (ic >= 0) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma() && ib <= kMaxBlockM) {
                wtrmm_simd_diag(conj_flag ? WSLUC : WSLUT, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_lutc_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            if (ic > 0) {
                wgemm_serial(gemm_trans[0], NN[0], ib, my_N, ic, &alpha, &A_(0, ic), lda, B_chunk, ldb, &one_cdd, &B_chunk[ic], ldb);
            }
            ic -= nb;
        }
    }
}

/* ── Blocked SIDE='R' chunk worker: serial blocked-TRMM over one row slice
 * [i_start, i_end). The wgemm trailing update routes through wgemm_serial. */

enum trmm_variant_R { WRLN, WRUN, WRLT, WRUT, WRLC, WRUC };

void blocked_chunk_R(trmm_variant_R V, std::ptrdiff_t i_start, std::ptrdiff_t i_end,
                     std::ptrdiff_t n, std::ptrdiff_t nb, TC alpha,
                     const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    const std::ptrdiff_t my_M = i_end - i_start;
    if (my_M <= 0) return;
    TC *B_chunk = &B_(i_start, 0);

    if (V == WRLN) {
        for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
            const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma()) {
                wtrmm_simd_diag_R(WRLN_OP, i_start, i_end, jb, alpha,
                                  &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            } else
#endif
            wtrmm_rln_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            const std::ptrdiff_t trailing = n - (jc + jb);
            if (trailing > 0) {
                const std::ptrdiff_t k0 = jc + jb;
                wgemm_serial(NN[0], NN[0], my_M, jb, trailing, &alpha, &B_chunk[static_cast<std::size_t>(k0) * ldb], ldb, &A_(k0, jc), lda, &one_cdd, &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
        }
    } else if (V == WRUN) {
        std::ptrdiff_t jc = ((n - 1) / nb) * nb;
        while (jc >= 0) {
            const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma()) {
                wtrmm_simd_diag_R(WRUN_OP, i_start, i_end, jb, alpha,
                                  &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            } else
#endif
            wtrmm_run_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            if (jc > 0) {
                wgemm_serial(NN[0], NN[0], my_M, jb, jc, &alpha, B_chunk, ldb, &A_(0, jc), lda, &one_cdd, &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
            jc -= nb;
        }
    } else if (V == WRLT || V == WRLC) {
        const bool conj_flag = (V == WRLC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        std::ptrdiff_t jc = ((n - 1) / nb) * nb;
        while (jc >= 0) {
            const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma()) {
                wtrmm_simd_diag_R(conj_flag ? WRLC_OP : WRLT_OP, i_start, i_end, jb, alpha,
                                  &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            } else
#endif
            wtrmm_rltc_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
            if (jc > 0) {
                wgemm_serial(NN[0], gemm_trans[0], my_M, jb, jc, &alpha, B_chunk, ldb, &A_(jc, 0), lda, &one_cdd, &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
            jc -= nb;
        }
    } else { /* WRUT or WRUC */
        const bool conj_flag = (V == WRUC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        for (std::ptrdiff_t jc = 0; jc < n; jc += nb) {
            const std::ptrdiff_t jb = (n - jc < nb) ? (n - jc) : nb;
#ifdef MBLAS_SIMD_DD
            if (mf_have_avx2_fma()) {
                wtrmm_simd_diag_R(conj_flag ? WRUC_OP : WRUT_OP, i_start, i_end, jb, alpha,
                                  &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            } else
#endif
            wtrmm_rutc_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
            const std::ptrdiff_t trailing = n - (jc + jb);
            if (trailing > 0) {
                const std::ptrdiff_t k0 = jc + jb;
                wgemm_serial(NN[0], gemm_trans[0], my_M, jb, trailing, &alpha, &B_chunk[static_cast<std::size_t>(k0) * ldb], ldb, &A_(jc, k0), lda, &one_cdd, &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
        }
    }
}

/* Map (UPLO, TRANS) → blocked variant. TRANS ∈ {'N','T','C'} kept distinct. */
inline trmm_variant_L l_variant(char UPLO, char TRANS) {
    if (TRANS == 'N') return (UPLO == 'L') ? WLLN : WLUN;
    if (TRANS == 'T') return (UPLO == 'L') ? WLLT : WLUT;
    return (UPLO == 'L') ? WLLC : WLUC;
}
inline trmm_variant_R r_variant(char UPLO, char TRANS) {
    if (TRANS == 'N') return (UPLO == 'L') ? WRLN : WRUN;
    if (TRANS == 'T') return (UPLO == 'L') ? WRLT : WRUT;
    return (UPLO == 'L') ? WRLC : WRUC;
}

}  // namespace

/* ── Exposed surface (wtrmm_kernel.h). ─────────────────────────────────── */

std::ptrdiff_t wtrmm_block_nb(void) { return trmm_nb(); }

void wtrmm_zero_B(std::ptrdiff_t m, std::ptrdiff_t n, TC *b, std::ptrdiff_t ldb)
{
    for (std::ptrdiff_t j = 0; j < n; ++j)
        for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = zero_cdd;
}

void wtrmm_L_slice(char UPLO, char TRANS, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, std::ptrdiff_t nb, TC alpha,
                   const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
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
        trmm_simd_op_w op;
        if (TRANS == 'N')      op = (UPLO == 'L') ? WSLLN : WSLUN;
        else if (TRANS == 'T') op = (UPLO == 'L') ? WSLLT : WSLUT;
        else                op = (UPLO == 'L') ? WSLLC : WSLUC;
        wtrmm_simd_diag(op, j_start, j_end, m, alpha, a, lda, b, ldb, nounit);
        return;
    }
#endif
    switch (V) {
    case WLLN: wtrmm_lln_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit); break;
    case WLUN: wtrmm_lun_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit); break;
    case WLLT: wtrmm_lltc_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit, 0); break;
    case WLUT: wtrmm_lutc_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit, 0); break;
    case WLLC: wtrmm_lltc_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit, 1); break;
    case WLUC: wtrmm_lutc_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit, 1); break;
    }
}

void wtrmm_R_slice(char UPLO, char TRANS, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t row_lo, std::ptrdiff_t row_hi, std::ptrdiff_t n, std::ptrdiff_t nb, TC alpha,
                   const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    if (row_lo >= row_hi) return;
    const trmm_variant_R V = r_variant(UPLO, TRANS);
    if (use_blocked) {
        blocked_chunk_R(V, row_lo, row_hi, n, nb, alpha, a, lda, b, ldb, nounit);
        return;
    }
#ifdef MBLAS_SIMD_DD
    if (mf_have_avx2_fma()) {
        wtrmm_r_op op;
        if (TRANS == 'N')      op = (UPLO == 'L') ? WRLN_OP : WRUN_OP;
        else if (TRANS == 'T') op = (UPLO == 'L') ? WRLT_OP : WRUT_OP;
        else                op = (UPLO == 'L') ? WRLC_OP : WRUC_OP;
        wtrmm_simd_diag_R(op, row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit);
        return;
    }
#endif
    switch (V) {
    case WRLN: wtrmm_rln_core(row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit); break;
    case WRUN: wtrmm_run_core(row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit); break;
    case WRLT: wtrmm_rltc_core(row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit, 0); break;
    case WRUT: wtrmm_rutc_core(row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit, 0); break;
    case WRLC: wtrmm_rltc_core(row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit, 1); break;
    case WRUC: wtrmm_rutc_core(row_lo, row_hi, n, alpha, a, lda, b, ldb, nounit, 1); break;
    }
}

extern "C" void wtrmm_serial(
    char side, char uplo, char transa, char diag,
    std::ptrdiff_t m, std::ptrdiff_t n,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    TC *b, std::ptrdiff_t ldb)
{
    const TC alpha = *alpha_;
    using mf_util::up;  /* char flag uppercase — mf_util.h (2a-4) */
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);
    const char TRANS = up(&transa);   /* complex: N/T/C kept distinct */
    const bool nounit = (up(&diag) != 'U');

    if (m == 0 || n == 0) return;

    if (ceq0(alpha)) { wtrmm_zero_B(m, n, b, ldb); return; }

    const std::ptrdiff_t nb = trmm_nb();

    if (SIDE == 'L') {
        const std::ptrdiff_t use_blocked = (m >= 2 * nb);
        wtrmm_L_slice(UPLO, TRANS, use_blocked, 0, n, m, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        const std::ptrdiff_t use_blocked = (n >= 2 * nb);
        wtrmm_R_slice(UPLO, TRANS, use_blocked, 0, m, n, nb, alpha,
                      a, lda, b, ldb, nounit);
    }
}

#undef A_
#undef B_
