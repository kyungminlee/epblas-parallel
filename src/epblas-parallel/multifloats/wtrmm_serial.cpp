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
 * workers take TR verbatim and map it (with UPLO) to the 6-way variant. The
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

#ifdef MBLAS_SIMD_DD
#include "mf_simd_fast.h"   /* cmul, cadd primitives */
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;


/* zero/one predicates — see mf_pred.h (2a-4 unification) */
using mf_pred::ceq0;
using mf_pred::ceq1;
namespace {

std::ptrdiff_t g_nb_trmm = 0;
std::ptrdiff_t trmm_nb(void) {
    if (g_nb_trmm == 0) g_nb_trmm = 64;
    return g_nb_trmm;
}

const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
const T one_cdd { R{1.0, 0.0}, R{0.0, 0.0} };


using mf_kernels::cmul;
using mf_kernels::cadd;
using mf_kernels::cconj;

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]

inline T A_op(const T *a, std::ptrdiff_t lda, std::ptrdiff_t row, std::ptrdiff_t col, bool conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

#ifdef MBLAS_SIMD_DD

constexpr std::ptrdiff_t kSimdLane = simd_fast::NR;
constexpr std::ptrdiff_t kMaxBlockM = 128;

inline void pack_B_4col_cdd(std::ptrdiff_t M, const T *b, std::ptrdiff_t ldb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                            double *rh, double *rl, double *ih, double *il)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        const T *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (std::ptrdiff_t i = 0; i < M; ++i) {
            rh[i * kSimdLane + j] = col[i].re.limbs[0];
            rl[i * kSimdLane + j] = col[i].re.limbs[1];
            ih[i * kSimdLane + j] = col[i].im.limbs[0];
            il[i * kSimdLane + j] = col[i].im.limbs[1];
        }
    }
    for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j)
        for (std::ptrdiff_t i = 0; i < M; ++i) {
            rh[i * kSimdLane + j] = 0.0; rl[i * kSimdLane + j] = 0.0;
            ih[i * kSimdLane + j] = 0.0; il[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_B_4col_cdd(std::ptrdiff_t M, T *b, std::ptrdiff_t ldb, std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                              const double *rh, const double *rl,
                              const double *ih, const double *il)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        T *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (std::ptrdiff_t i = 0; i < M; ++i) {
            col[i].re.limbs[0] = rh[i * kSimdLane + j];
            col[i].re.limbs[1] = rl[i * kSimdLane + j];
            col[i].im.limbs[0] = ih[i * kSimdLane + j];
            col[i].im.limbs[1] = il[i * kSimdLane + j];
        }
    }
}

/* Broadcast A(r,c) with optional conjugate (negate im if conj_flag). */
inline void broadcast_A_cdd(const T *a, std::ptrdiff_t lda, std::ptrdiff_t r, std::ptrdiff_t c, bool conj_flag,
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

inline void simd_wtrmm_lln(std::ptrdiff_t M, const T *a, std::ptrdiff_t lda, T alpha, bool nounit,
                           double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (std::ptrdiff_t k = M - 1; k >= 0; --k) {
        __m256d bkrh = _mm256_load_pd(&brh[k * kSimdLane]);
        __m256d bkrl = _mm256_load_pd(&brl[k * kSimdLane]);
        __m256d bkih = _mm256_load_pd(&bih[k * kSimdLane]);
        __m256d bkil = _mm256_load_pd(&bil[k * kSimdLane]);
        __m256d trh, trl, tih, til;
        simd_fast::cmul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                         trh, trl, tih, til);
        for (std::ptrdiff_t i = M - 1; i > k; --i) {
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

inline void simd_wtrmm_lun(std::ptrdiff_t M, const T *a, std::ptrdiff_t lda, T alpha, bool nounit,
                           double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (std::ptrdiff_t k = 0; k < M; ++k) {
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
inline void simd_wtrmm_llTC(std::ptrdiff_t M, const T *a, std::ptrdiff_t lda, T alpha, bool nounit,
                            bool conj_flag,
                            double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (std::ptrdiff_t i = 0; i < M; ++i) {
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
        for (std::ptrdiff_t k = i + 1; k < M; ++k) {
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
inline void simd_wtrmm_luTC(std::ptrdiff_t M, const T *a, std::ptrdiff_t lda, T alpha, bool nounit,
                            bool conj_flag,
                            double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (std::ptrdiff_t i = M - 1; i >= 0; --i) {
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
                            std::ptrdiff_t M, T alpha,
                            const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    alignas(32) double brh[kMaxBlockM * kSimdLane];
    alignas(32) double brl[kMaxBlockM * kSimdLane];
    alignas(32) double bih[kMaxBlockM * kSimdLane];
    alignas(32) double bil[kMaxBlockM * kSimdLane];
    for (std::ptrdiff_t j = j_start; j < j_end; j += kSimdLane) {
        const std::ptrdiff_t jc = (j_end - j < kSimdLane) ? (j_end - j) : kSimdLane;
        pack_B_4col_cdd(M, b, ldb, j, jc, brh, brl, bih, bil);
        switch (op) {
        case WSLLN: simd_wtrmm_lln(M, a, lda, alpha, nounit, brh, brl, bih, bil); break;
        case WSLUN: simd_wtrmm_lun(M, a, lda, alpha, nounit, brh, brl, bih, bil); break;
        case WSLLT: simd_wtrmm_llTC(M, a, lda, alpha, nounit, 0, brh, brl, bih, bil); break;
        case WSLUT: simd_wtrmm_luTC(M, a, lda, alpha, nounit, 0, brh, brl, bih, bil); break;
        case WSLLC: simd_wtrmm_llTC(M, a, lda, alpha, nounit, 1, brh, brl, bih, bil); break;
        case WSLUC: simd_wtrmm_luTC(M, a, lda, alpha, nounit, 1, brh, brl, bih, bil); break;
        }
        unpack_B_4col_cdd(M, b, ldb, j, jc, brh, brl, bih, bil);
    }
}

#endif  /* MBLAS_SIMD_DD */

inline void wtrmm_lln_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t k = M - 1; k >= 0; --k) {
            if (!ceq0(B_(k, j))) {
                T temp = cmul(alpha, B_(k, j));
                for (std::ptrdiff_t i = M - 1; i > k; --i)
                    B_(i, j) = cadd(B_(i, j), cmul(temp, A_(i, k)));
                if (nounit) temp = cmul(temp, A_(k, k));
                B_(k, j) = temp;
            }
        }
    }
}

inline void wtrmm_lun_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t k = 0; k < M; ++k) {
            if (!ceq0(B_(k, j))) {
                T temp = cmul(alpha, B_(k, j));
                for (std::ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(temp, A_(i, k)));
                if (nounit) temp = cmul(temp, A_(k, k));
                B_(k, j) = temp;
            }
        }
    }
}

inline void wtrmm_llTC_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, T alpha,
                            const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t i = 0; i < M; ++i) {
            T t = B_(i, j);
            if (nounit) t = cmul(t, A_op(a, lda, i, i, conj_flag));
            for (std::ptrdiff_t k = i + 1; k < M; ++k)
                t = cadd(t, cmul(A_op(a, lda, k, i, conj_flag), B_(k, j)));
            B_(i, j) = cmul(alpha, t);
        }
    }
}

inline void wtrmm_luTC_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, T alpha,
                            const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t i = M - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t = cmul(t, A_op(a, lda, i, i, conj_flag));
            for (std::ptrdiff_t k = 0; k < i; ++k)
                t = cadd(t, cmul(A_op(a, lda, k, i, conj_flag), B_(k, j)));
            B_(i, j) = cmul(alpha, t);
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

#ifdef MBLAS_SIMD_DD

/* Forward decls for scalar tails (defined below). */
inline void wtrmm_rln_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, T, const T*, std::ptrdiff_t, T*, std::ptrdiff_t, bool);
inline void wtrmm_run_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, T, const T*, std::ptrdiff_t, T*, std::ptrdiff_t, bool);
inline void wtrmm_rlTC_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, T, const T*, std::ptrdiff_t, T*, std::ptrdiff_t, bool, bool);
inline void wtrmm_ruTC_core(std::ptrdiff_t, std::ptrdiff_t, std::ptrdiff_t, T, const T*, std::ptrdiff_t, T*, std::ptrdiff_t, bool, bool);

using simd_exact::cload4;
using simd_exact::cstore4;

using simd_exact::vbcast;

inline void simd_wtrmm_r4_rln(std::ptrdiff_t ib, std::ptrdiff_t N, T alpha,
                              const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = 0; j < N; ++j) {
        T *bj = b + static_cast<std::size_t>(j) * ldb;
        T t = alpha;
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
        for (std::ptrdiff_t k = j + 1; k < N; ++k) {
            const T akj_v = A_(k, j);
            if (ceq0(akj_v)) continue;
            const T akj = cmul(alpha, akj_v);
            __m256d arh, arl, aih, ail;
            vbcast(akj, arh, arl, aih, ail);
            const T *bk = b + static_cast<std::size_t>(k) * ldb;
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

inline void simd_wtrmm_r4_run(std::ptrdiff_t ib, std::ptrdiff_t N, T alpha,
                              const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = N - 1; j >= 0; --j) {
        T *bj = b + static_cast<std::size_t>(j) * ldb;
        T t = alpha;
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
            const T akj_v = A_(k, j);
            if (ceq0(akj_v)) continue;
            const T akj = cmul(alpha, akj_v);
            __m256d arh, arl, aih, ail;
            vbcast(akj, arh, arl, aih, ail);
            const T *bk = b + static_cast<std::size_t>(k) * ldb;
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

inline void simd_wtrmm_r4_rlTC(std::ptrdiff_t ib, std::ptrdiff_t N, T alpha, bool conj_flag,
                               const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t k = N - 1; k >= 0; --k) {
        const T *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkrh, bkrl, bkih, bkil;
        cload4(bk + ib, bkrh, bkrl, bkih, bkil);
        for (std::ptrdiff_t j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ceq0(ajk)) continue;
            const T scaled = cmul(alpha, ajk);
            __m256d arh, arl, aih, ail;
            vbcast(scaled, arh, arl, aih, ail);
            T *bj = b + static_cast<std::size_t>(j) * ldb;
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
        T t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!ceq1(t)) {
            __m256d trh, trl, tih, til;
            vbcast(t, trh, trl, tih, til);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(bkrh, bkrl, bkih, bkil, trh, trl, tih, til,
                             nrh, nrl, nih, nil_);
            cstore4(const_cast<T*>(bk) + ib, nrh, nrl, nih, nil_);
        }
    }
}

inline void simd_wtrmm_r4_ruTC(std::ptrdiff_t ib, std::ptrdiff_t N, T alpha, bool conj_flag,
                               const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t k = 0; k < N; ++k) {
        const T *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkrh, bkrl, bkih, bkil;
        cload4(bk + ib, bkrh, bkrl, bkih, bkil);
        for (std::ptrdiff_t j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ceq0(ajk)) continue;
            const T scaled = cmul(alpha, ajk);
            __m256d arh, arl, aih, ail;
            vbcast(scaled, arh, arl, aih, ail);
            T *bj = b + static_cast<std::size_t>(j) * ldb;
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
        T t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!ceq1(t)) {
            __m256d trh, trl, tih, til;
            vbcast(t, trh, trl, tih, til);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(bkrh, bkrl, bkih, bkil, trh, trl, tih, til,
                             nrh, nrl, nih, nil_);
            cstore4(const_cast<T*>(bk) + ib, nrh, nrl, nih, nil_);
        }
    }
}

enum wtrmm_r_op { WRLN_OP, WRUN_OP, WRLT_OP, WRUT_OP, WRLC_OP, WRUC_OP };

inline void wtrmm_simd_diag_R(wtrmm_r_op op, std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t N, T alpha,
                              const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    const std::ptrdiff_t i4_end = i_start + ((i_end - i_start) & ~3);
    for (std::ptrdiff_t ib = i_start; ib < i4_end; ib += 4) {
        switch (op) {
        case WRLN_OP: simd_wtrmm_r4_rln(ib, N, alpha, a, lda, b, ldb, nounit); break;
        case WRUN_OP: simd_wtrmm_r4_run(ib, N, alpha, a, lda, b, ldb, nounit); break;
        case WRLT_OP: simd_wtrmm_r4_rlTC(ib, N, alpha, 0, a, lda, b, ldb, nounit); break;
        case WRUT_OP: simd_wtrmm_r4_ruTC(ib, N, alpha, 0, a, lda, b, ldb, nounit); break;
        case WRLC_OP: simd_wtrmm_r4_rlTC(ib, N, alpha, 1, a, lda, b, ldb, nounit); break;
        case WRUC_OP: simd_wtrmm_r4_ruTC(ib, N, alpha, 1, a, lda, b, ldb, nounit); break;
        }
    }
    /* Scalar tail rows */
    if (i4_end < i_end) {
        switch (op) {
        case WRLN_OP: wtrmm_rln_core(i4_end, i_end, N, alpha, a, lda, b, ldb, nounit); break;
        case WRUN_OP: wtrmm_run_core(i4_end, i_end, N, alpha, a, lda, b, ldb, nounit); break;
        case WRLT_OP: wtrmm_rlTC_core(i4_end, i_end, N, alpha, a, lda, b, ldb, nounit, 0); break;
        case WRUT_OP: wtrmm_ruTC_core(i4_end, i_end, N, alpha, a, lda, b, ldb, nounit, 0); break;
        case WRLC_OP: wtrmm_rlTC_core(i4_end, i_end, N, alpha, a, lda, b, ldb, nounit, 1); break;
        case WRUC_OP: wtrmm_ruTC_core(i4_end, i_end, N, alpha, a, lda, b, ldb, nounit, 1); break;
        }
    }
}

#endif  /* MBLAS_SIMD_DD */

inline void wtrmm_rln_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t N, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = 0; j < N; ++j) {
        T t = alpha;
        if (nounit) t = cmul(t, A_(j, j));
        if (!ceq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) = cmul(B_(i, j), t);
        for (std::ptrdiff_t k = j + 1; k < N; ++k) {
            if (!ceq0(A_(k, j))) {
                const T akj = cmul(alpha, A_(k, j));
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(akj, B_(i, k)));
            }
        }
    }
}

inline void wtrmm_run_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t N, T alpha,
                           const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = N - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t = cmul(t, A_(j, j));
        if (!ceq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, j) = cmul(B_(i, j), t);
        for (std::ptrdiff_t k = 0; k < j; ++k) {
            if (!ceq0(A_(k, j))) {
                const T akj = cmul(alpha, A_(k, j));
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(akj, B_(i, k)));
            }
        }
    }
}

inline void wtrmm_rlTC_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t N, T alpha,
                            const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t k = N - 1; k >= 0; --k) {
        for (std::ptrdiff_t j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (!ceq0(ajk)) {
                const T scaled = cmul(alpha, ajk);
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(scaled, B_(i, k)));
            }
        }
        T t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!ceq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) = cmul(B_(i, k), t);
    }
}

inline void wtrmm_ruTC_core(std::ptrdiff_t i_start, std::ptrdiff_t i_end, std::ptrdiff_t N, T alpha,
                            const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t k = 0; k < N; ++k) {
        for (std::ptrdiff_t j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (!ceq0(ajk)) {
                const T scaled = cmul(alpha, ajk);
                for (std::ptrdiff_t i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(scaled, B_(i, k)));
            }
        }
        T t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!ceq1(t))
            for (std::ptrdiff_t i = i_start; i < i_end; ++i) B_(i, k) = cmul(B_(i, k), t);
    }
}

/* ── Blocked SIDE='L' chunk worker: serial blocked-TRMM over one column slice
 * [j_start, j_end). The wgemm trailing update routes through wgemm_serial. */

enum trmm_variant_L { WLLN, WLUN, WLLT, WLUT, WLLC, WLUC };

void blocked_chunk_L(trmm_variant_L V, std::ptrdiff_t j_start, std::ptrdiff_t j_end,
                     std::ptrdiff_t M, std::ptrdiff_t nb, T alpha,
                     const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    const std::ptrdiff_t my_N = j_end - j_start;
    if (my_N <= 0) return;

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    T *B_chunk = &B_(0, j_start);

    if (V == WLLN) {
        std::ptrdiff_t ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const std::ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
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
        for (std::ptrdiff_t ic = 0; ic < M; ic += nb) {
            const std::ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                wtrmm_simd_diag(WSLUN, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_lun_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const std::ptrdiff_t trailing = M - (ic + ib);
            if (trailing > 0) {
                const std::ptrdiff_t j0 = ic + ib;
                wgemm_serial(NN[0], NN[0], ib, my_N, trailing, &alpha, &A_(ic, j0), lda, &B_chunk[j0], ldb, &one_cdd, &B_chunk[ic], ldb);
            }
        }
    } else if (V == WLLT || V == WLLC) {
        const bool conj_flag = (V == WLLC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        for (std::ptrdiff_t ic = 0; ic < M; ic += nb) {
            const std::ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                wtrmm_simd_diag(conj_flag ? WSLLC : WSLLT, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_llTC_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            const std::ptrdiff_t trailing = M - (ic + ib);
            if (trailing > 0) {
                const std::ptrdiff_t i0 = ic + ib;
                wgemm_serial(gemm_trans[0], NN[0], ib, my_N, trailing, &alpha, &A_(i0, ic), lda, &B_chunk[i0], ldb, &one_cdd, &B_chunk[ic], ldb);
            }
        }
    } else { /* WLUT or WLUC */
        const bool conj_flag = (V == WLUC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        std::ptrdiff_t ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const std::ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                wtrmm_simd_diag(conj_flag ? WSLUC : WSLUT, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_luTC_core(j_start, j_end, ib, alpha,
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
                     std::ptrdiff_t N, std::ptrdiff_t nb, T alpha,
                     const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    const std::ptrdiff_t my_M = i_end - i_start;
    if (my_M <= 0) return;
    T *B_chunk = &B_(i_start, 0);

    if (V == WRLN) {
        for (std::ptrdiff_t jc = 0; jc < N; jc += nb) {
            const std::ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            wtrmm_simd_diag_R(WRLN_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            wtrmm_rln_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#endif
            const std::ptrdiff_t trailing = N - (jc + jb);
            if (trailing > 0) {
                const std::ptrdiff_t k0 = jc + jb;
                wgemm_serial(NN[0], NN[0], my_M, jb, trailing, &alpha, &B_chunk[static_cast<std::size_t>(k0) * ldb], ldb, &A_(k0, jc), lda, &one_cdd, &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
        }
    } else if (V == WRUN) {
        std::ptrdiff_t jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const std::ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            wtrmm_simd_diag_R(WRUN_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            wtrmm_run_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#endif
            if (jc > 0) {
                wgemm_serial(NN[0], NN[0], my_M, jb, jc, &alpha, B_chunk, ldb, &A_(0, jc), lda, &one_cdd, &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
            jc -= nb;
        }
    } else if (V == WRLT || V == WRLC) {
        const bool conj_flag = (V == WRLC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        std::ptrdiff_t jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const std::ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            wtrmm_simd_diag_R(conj_flag ? WRLC_OP : WRLT_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            wtrmm_rlTC_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
#endif
            if (jc > 0) {
                wgemm_serial(NN[0], gemm_trans[0], my_M, jb, jc, &alpha, B_chunk, ldb, &A_(jc, 0), lda, &one_cdd, &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
            jc -= nb;
        }
    } else { /* WRUT or WRUC */
        const bool conj_flag = (V == WRUC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        for (std::ptrdiff_t jc = 0; jc < N; jc += nb) {
            const std::ptrdiff_t jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            wtrmm_simd_diag_R(conj_flag ? WRUC_OP : WRUT_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            wtrmm_ruTC_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
#endif
            const std::ptrdiff_t trailing = N - (jc + jb);
            if (trailing > 0) {
                const std::ptrdiff_t k0 = jc + jb;
                wgemm_serial(NN[0], gemm_trans[0], my_M, jb, trailing, &alpha, &B_chunk[static_cast<std::size_t>(k0) * ldb], ldb, &A_(jc, k0), lda, &one_cdd, &B_chunk[static_cast<std::size_t>(jc) * ldb], ldb);
            }
        }
    }
}

/* Map (UPLO, TR) → blocked variant. TR ∈ {'N','T','C'} kept distinct. */
inline trmm_variant_L l_variant(char UPLO, char TR) {
    if (TR == 'N') return (UPLO == 'L') ? WLLN : WLUN;
    if (TR == 'T') return (UPLO == 'L') ? WLLT : WLUT;
    return (UPLO == 'L') ? WLLC : WLUC;
}
inline trmm_variant_R r_variant(char UPLO, char TR) {
    if (TR == 'N') return (UPLO == 'L') ? WRLN : WRUN;
    if (TR == 'T') return (UPLO == 'L') ? WRLT : WRUT;
    return (UPLO == 'L') ? WRLC : WRUC;
}

}  // namespace

/* ── Exposed surface (wtrmm_kernel.h). ─────────────────────────────────── */

std::ptrdiff_t wtrmm_block_nb(void) { return trmm_nb(); }

void wtrmm_zero_B(std::ptrdiff_t M, std::ptrdiff_t N, T *b, std::ptrdiff_t ldb)
{
    for (std::ptrdiff_t j = 0; j < N; ++j)
        for (std::ptrdiff_t i = 0; i < M; ++i) B_(i, j) = zero_cdd;
}

void wtrmm_L_slice(char UPLO, char TR, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t M, std::ptrdiff_t nb, T alpha,
                   const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    if (j_start >= j_end) return;
    const trmm_variant_L V = l_variant(UPLO, TR);
    if (use_blocked) {
        blocked_chunk_L(V, j_start, j_end, M, nb, alpha, a, lda, b, ldb, nounit);
        return;
    }
#ifdef MBLAS_SIMD_DD
    /* Small-M (single-block) regime: route the unblocked path through the SIMD
     * diag too — same kernel, over this slice's column range. */
    if (M <= kMaxBlockM) {
        trmm_simd_op_w op;
        if (TR == 'N')      op = (UPLO == 'L') ? WSLLN : WSLUN;
        else if (TR == 'T') op = (UPLO == 'L') ? WSLLT : WSLUT;
        else                op = (UPLO == 'L') ? WSLLC : WSLUC;
        wtrmm_simd_diag(op, j_start, j_end, M, alpha, a, lda, b, ldb, nounit);
        return;
    }
#endif
    switch (V) {
    case WLLN: wtrmm_lln_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit); break;
    case WLUN: wtrmm_lun_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit); break;
    case WLLT: wtrmm_llTC_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit, 0); break;
    case WLUT: wtrmm_luTC_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit, 0); break;
    case WLLC: wtrmm_llTC_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit, 1); break;
    case WLUC: wtrmm_luTC_core(j_start, j_end, M, alpha, a, lda, b, ldb, nounit, 1); break;
    }
}

void wtrmm_R_slice(char UPLO, char TR, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t row_lo, std::ptrdiff_t row_hi, std::ptrdiff_t N, std::ptrdiff_t nb, T alpha,
                   const T *a, std::ptrdiff_t lda, T *b, std::ptrdiff_t ldb, bool nounit)
{
    if (row_lo >= row_hi) return;
    const trmm_variant_R V = r_variant(UPLO, TR);
    if (use_blocked) {
        blocked_chunk_R(V, row_lo, row_hi, N, nb, alpha, a, lda, b, ldb, nounit);
        return;
    }
#ifdef MBLAS_SIMD_DD
    wtrmm_r_op op;
    if (TR == 'N')      op = (UPLO == 'L') ? WRLN_OP : WRUN_OP;
    else if (TR == 'T') op = (UPLO == 'L') ? WRLT_OP : WRUT_OP;
    else                op = (UPLO == 'L') ? WRLC_OP : WRUC_OP;
    wtrmm_simd_diag_R(op, row_lo, row_hi, N, alpha, a, lda, b, ldb, nounit);
#else
    switch (V) {
    case WRLN: wtrmm_rln_core(row_lo, row_hi, N, alpha, a, lda, b, ldb, nounit); break;
    case WRUN: wtrmm_run_core(row_lo, row_hi, N, alpha, a, lda, b, ldb, nounit); break;
    case WRLT: wtrmm_rlTC_core(row_lo, row_hi, N, alpha, a, lda, b, ldb, nounit, 0); break;
    case WRUT: wtrmm_ruTC_core(row_lo, row_hi, N, alpha, a, lda, b, ldb, nounit, 0); break;
    case WRLC: wtrmm_rlTC_core(row_lo, row_hi, N, alpha, a, lda, b, ldb, nounit, 1); break;
    case WRUC: wtrmm_ruTC_core(row_lo, row_hi, N, alpha, a, lda, b, ldb, nounit, 1); break;
    }
#endif
}

extern "C" void wtrmm_serial(
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
    const char TR = up(&transa);   /* complex: N/T/C kept distinct */
    const bool nounit = (up(&diag) != 'U');

    if (M == 0 || N == 0) return;

    if (ceq0(alpha)) { wtrmm_zero_B(M, N, b, ldb); return; }

    const std::ptrdiff_t nb = trmm_nb();

    if (SIDE == 'L') {
        const std::ptrdiff_t use_blocked = (M >= 2 * nb);
        wtrmm_L_slice(UPLO, TR, use_blocked, 0, N, M, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        const std::ptrdiff_t use_blocked = (N >= 2 * nb);
        wtrmm_R_slice(UPLO, TR, use_blocked, 0, M, N, nb, alpha,
                      a, lda, b, ldb, nounit);
    }
}

#undef A_
#undef B_
