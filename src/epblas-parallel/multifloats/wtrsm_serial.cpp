/*
 * wtrsm_serial.cpp — multifloats complex (complex64x2) triangular solve,
 * single-thread core. Owns ALL the numerics shared by the serial and parallel
 * entries:
 *
 *   - scalar column "core" kernels for SIDE='L' (LLN/LUN + LLT/LLC/LUT/LUC via
 *     a conj flag) and the SIDE='R' cores (RLN/RUN + RLT/RLC/RUT/RUC),
 *   - the AVX2 4-wide SIMD diagonal kernels (SIDE='L' packed-SoA and SIDE='R'
 *     4-row chunks), under WBLAS_SIMD_DD,
 *   - the block-size policy and the blocked SIDE='L' chunk worker, whose
 *     trailing-matrix update routes through wgemm_serial (no nested OpenMP),
 *   - the per-slice workers wtrsm_L_slice / wtrsm_R_slice (declared in
 *     wtrsm_kernel.h) that the parallel entry fans across a team, plus the
 *     public `wtrsm_serial` entry.
 *
 * There is NO OpenMP on this path. Threading lives entirely in
 * wtrsm_parallel.cpp; both paths drive these workers, so a static partition
 * is bitwise-identical to the serial sweep.
 *
 * Unlike the real (mtrsm) twin, TRANSA='C' (conjugate transpose) is kept
 * DISTINCT from 'T' — the conj flag gates conj() on the A reads.
 */

#include "wtrsm_kernel.h"
#include "mf_pred.h"
#include "mf_kernels.h"
#include "wgemm_kernel.h"   /* wgemm_serial for the trailing update */
#include <cstddef>
#include <cstdlib>
#include <cctype>
#include "mf_util.h"
#include "mf_dispatch.h"   /* MF_SIMD_TARGET + mf_have_avx2_fma() runtime gate */

#ifdef WBLAS_SIMD_DD
#include "mf_simd_fast.h"   /* mul, add, neg, cmul/add */
#include "mf_simd_exact.h"
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using TC = mf::complex64x2;


/* zero/one predicates — see mf_pred.h */
using mf_pred::ceq0;
using mf_pred::ceq1;
namespace {

/* Triangular-axis block size for the blocked paths — compile-time constant
 * (nothing writes it). */
constexpr std::ptrdiff_t g_nb_trsm = 64;
std::ptrdiff_t trsm_nb(void) { return g_nb_trsm; }

using mf_pred::zero_cdd;   /* shared DD constants — mf_pred.h */
using mf_pred::one_cdd;


/* Complex DD ops via header overloads. */
using mf_kernels::cmul;
using mf_kernels::csub;
using mf_kernels::cconj;
inline TC cdiv(TC const &a, TC const &b) {
    /* a / b = a · conj(b) / |b|² ; multifloats provides operator/ on
     * float64x2 so we just compute via the standard formula. */
    R denom = b.re * b.re + b.im * b.im;
    return TC{ (a.re * b.re + a.im * b.im) / denom,
              (a.im * b.re - a.re * b.im) / denom };
}

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]

/* ── Scalar column-range cores ──────────────────────────────────
 * The complex variants follow the same algorithm shape as the real
 * counterparts. For 'C' (conjugate transpose) we replace A[k,i]
 * with conj(A[k,i]) — the math is solve conj(A)ᵀ X = α B. */

inline void wtrsm_lln_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TC alpha,
                           const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        if (!ceq1(alpha)) for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = cmul(B_(i, j), alpha);
        for (std::ptrdiff_t k = 0; k < m; ++k) {
            if (!ceq0(B_(k, j))) {
                if (nounit) B_(k, j) = cdiv(B_(k, j), A_(k, k));
                const TC bk = B_(k, j);
                for (std::ptrdiff_t i = k + 1; i < m; ++i)
                    B_(i, j) = csub(B_(i, j), cmul(bk, A_(i, k)));
            }
        }
    }
}

inline void wtrsm_lun_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TC alpha,
                           const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        if (!ceq1(alpha)) for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = cmul(B_(i, j), alpha);
        for (std::ptrdiff_t k = m - 1; k >= 0; --k) {
            if (!ceq0(B_(k, j))) {
                if (nounit) B_(k, j) = cdiv(B_(k, j), A_(k, k));
                const TC bk = B_(k, j);
                for (std::ptrdiff_t i = 0; i < k; ++i)
                    B_(i, j) = csub(B_(i, j), cmul(bk, A_(i, k)));
            }
        }
    }
}

/* For TRANS='T' (transpose, no conj): use A[k,i] as written.
 * For TRANS='C' (conjugate transpose): use conj(A[k,i]). The conj flag
 * gates the conj on A reads inside the inner loop. */
inline TC A_op(const TC *a, std::ptrdiff_t lda, std::ptrdiff_t row, std::ptrdiff_t col, bool conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* (L, L, T) and (L, L, C): solve op(A)ᵀ X = α B where A is lower-tri.
 * Inner-product form: t = α B[i,j]; for k > i: t -= op(A)[k,i] B[k,j];
 *                     B[i,j] = t / op(A)[i,i] (or = t if unit). */
inline void wtrsm_lltc_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TC alpha,
                            const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t i = m - 1; i >= 0; --i) {
            TC t = cmul(alpha, B_(i, j));
            for (std::ptrdiff_t k = i + 1; k < m; ++k) {
                t = csub(t, cmul(A_op(a, lda, k, i, conj_flag), B_(k, j)));
            }
            if (nounit) t = cdiv(t, A_op(a, lda, i, i, conj_flag));
            B_(i, j) = t;
        }
    }
}

/* (L, U, T) and (L, U, C): solve op(A)ᵀ X = α B, A upper. */
inline void wtrsm_lutc_core(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TC alpha,
                            const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t j = j_start; j < j_end; ++j) {
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            TC t = cmul(alpha, B_(i, j));
            for (std::ptrdiff_t k = 0; k < i; ++k) {
                t = csub(t, cmul(A_op(a, lda, k, i, conj_flag), B_(k, j)));
            }
            if (nounit) t = cdiv(t, A_op(a, lda, i, i, conj_flag));
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R' cores: scalar full-N. */

inline void wtrsm_rln_core(std::ptrdiff_t m, std::ptrdiff_t n, TC alpha,
                           const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
        if (!ceq1(alpha)) for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = cmul(B_(i, j), alpha);
        for (std::ptrdiff_t k = j + 1; k < n; ++k) {
            if (!ceq0(A_(k, j))) {
                const TC akj = A_(k, j);
                for (std::ptrdiff_t i = 0; i < m; ++i)
                    B_(i, j) = csub(B_(i, j), cmul(akj, B_(i, k)));
            }
        }
        if (nounit) {
            const TC inv = cdiv(one_cdd, A_(j, j));
            for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = cmul(B_(i, j), inv);
        }
    }
}

inline void wtrsm_run_core(std::ptrdiff_t m, std::ptrdiff_t n, TC alpha,
                           const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        if (!ceq1(alpha)) for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = cmul(B_(i, j), alpha);
        for (std::ptrdiff_t k = 0; k < j; ++k) {
            if (!ceq0(A_(k, j))) {
                const TC akj = A_(k, j);
                for (std::ptrdiff_t i = 0; i < m; ++i)
                    B_(i, j) = csub(B_(i, j), cmul(akj, B_(i, k)));
            }
        }
        if (nounit) {
            const TC inv = cdiv(one_cdd, A_(j, j));
            for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = cmul(B_(i, j), inv);
        }
    }
}

inline void wtrsm_rltc_core(std::ptrdiff_t m, std::ptrdiff_t n, TC alpha,
                            const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t k = 0; k < n; ++k) {
        if (nounit) {
            const TC inv = cdiv(one_cdd, A_op(a, lda, k, k, conj_flag));
            for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, k) = cmul(B_(i, k), inv);
        }
        for (std::ptrdiff_t j = k + 1; j < n; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (!ceq0(ajk)) {
                for (std::ptrdiff_t i = 0; i < m; ++i)
                    B_(i, j) = csub(B_(i, j), cmul(ajk, B_(i, k)));
            }
        }
        if (!ceq1(alpha)) for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, k) = cmul(B_(i, k), alpha);
    }
}

inline void wtrsm_rutc_core(std::ptrdiff_t m, std::ptrdiff_t n, TC alpha,
                            const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb,
                            bool nounit, bool conj_flag)
{
    for (std::ptrdiff_t k = n - 1; k >= 0; --k) {
        if (nounit) {
            const TC inv = cdiv(one_cdd, A_op(a, lda, k, k, conj_flag));
            for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, k) = cmul(B_(i, k), inv);
        }
        for (std::ptrdiff_t j = 0; j < k; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (!ceq0(ajk)) {
                for (std::ptrdiff_t i = 0; i < m; ++i)
                    B_(i, j) = csub(B_(i, j), cmul(ajk, B_(i, k)));
            }
        }
        if (!ceq1(alpha)) for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, k) = cmul(B_(i, k), alpha);
    }
}

/* ── SIMD 4-wide diagonal kernels (complex). ─────────────────── */

#ifdef WBLAS_SIMD_DD

/* AVX2+FMA under a possibly pre-Haswell baseline -march: these SIMD kernels and
 * their helpers are compiled with the feature enabled and reached only behind
 * mf_have_avx2_fma() at the call sites below. See mf_dispatch.h. */
#pragma GCC push_options
#pragma GCC target("avx2,fma")

constexpr std::ptrdiff_t kSimdLane = simd_fast::NR;
constexpr std::ptrdiff_t kMaxBlockM = 256;

inline void pack_B_4col_complex(std::ptrdiff_t m, const TC *b, std::ptrdiff_t ldb,
                                std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                                double *brh, double *brl,
                                double *bih, double *bil)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        const TC *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            brh[i * kSimdLane + j] = col[i].re.limbs[0];
            brl[i * kSimdLane + j] = col[i].re.limbs[1];
            bih[i * kSimdLane + j] = col[i].im.limbs[0];
            bil[i * kSimdLane + j] = col[i].im.limbs[1];
        }
    }
    for (std::ptrdiff_t j = j_count; j < kSimdLane; ++j)
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            brh[i * kSimdLane + j] = 0.0;
            brl[i * kSimdLane + j] = 0.0;
            bih[i * kSimdLane + j] = 0.0;
            bil[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_B_4col_complex(std::ptrdiff_t m, TC *b, std::ptrdiff_t ldb,
                                  std::ptrdiff_t j_start, std::ptrdiff_t j_count,
                                  const double *brh, const double *brl,
                                  const double *bih, const double *bil)
{
    for (std::ptrdiff_t j = 0; j < j_count; ++j) {
        TC *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (std::ptrdiff_t i = 0; i < m; ++i) {
            col[i].re.limbs[0] = brh[i * kSimdLane + j];
            col[i].re.limbs[1] = brl[i * kSimdLane + j];
            col[i].im.limbs[0] = bih[i * kSimdLane + j];
            col[i].im.limbs[1] = bil[i * kSimdLane + j];
        }
    }
}

/* Helper: load row k into 4 ymm regs from SoA scratch. */
#define LOAD_ROW(idx, rh, rl, ih, il)                          \
    do {                                                       \
        rh = _mm256_loadu_pd(&brh[(idx) * kSimdLane]);         \
        rl = _mm256_loadu_pd(&brl[(idx) * kSimdLane]);         \
        ih = _mm256_loadu_pd(&bih[(idx) * kSimdLane]);         \
        il = _mm256_loadu_pd(&bil[(idx) * kSimdLane]);         \
    } while (0)

#define STORE_ROW(idx, rh, rl, ih, il)                         \
    do {                                                       \
        _mm256_storeu_pd(&brh[(idx) * kSimdLane], rh);         \
        _mm256_storeu_pd(&brl[(idx) * kSimdLane], rl);         \
        _mm256_storeu_pd(&bih[(idx) * kSimdLane], ih);         \
        _mm256_storeu_pd(&bil[(idx) * kSimdLane], il);         \
    } while (0)

/* SIMD broadcast of a complex DD scalar into 4 ymm regs. */
#define BCAST_T(x, rh, rl, ih, il)                                 \
    do {                                                           \
        rh = _mm256_set1_pd((x).re.limbs[0]);                      \
        rl = _mm256_set1_pd((x).re.limbs[1]);                      \
        ih = _mm256_set1_pd((x).im.limbs[0]);                      \
        il = _mm256_set1_pd((x).im.limbs[1]);                      \
    } while (0)

inline void simd_prescale_complex(std::ptrdiff_t m, TC alpha,
                                  double *brh, double *brl,
                                  double *bih, double *bil)
{
    if (ceq1(alpha)) return;
    if (ceq0(alpha)) {
        const __m256d z = _mm256_setzero_pd();
        for (std::ptrdiff_t k = 0; k < m; ++k) STORE_ROW(k, z, z, z, z);
        return;
    }
    __m256d arh, arl, aih, ail;
    BCAST_T(alpha, arh, arl, aih, ail);
    for (std::ptrdiff_t k = 0; k < m; ++k) {
        __m256d brh_v, brl_v, bih_v, bil_v;
        LOAD_ROW(k, brh_v, brl_v, bih_v, bil_v);
        __m256d nrh, nrl, nih, nil;
        simd_fast::cmul(arh, arl, aih, ail,
                         brh_v, brl_v, bih_v, bil_v,
                         nrh, nrl, nih, nil);
        STORE_ROW(k, nrh, nrl, nih, nil);
    }
}

inline void simd_neg_cdd(__m256d &rh, __m256d &rl, __m256d &ih, __m256d &il) {
    simd_fast::neg(rh, rl);
    simd_fast::neg(ih, il);
}

/* SIMD forward sub on (L, L, N): rank-1 form. */
inline void simd_fwd_sub_lln_cdd(std::ptrdiff_t m, const TC *a, std::ptrdiff_t lda, bool nounit,
                                 double *brh, double *brl,
                                 double *bih, double *bil)
{
    for (std::ptrdiff_t k = 0; k < m; ++k) {
        __m256d bkrh, bkrl, bkih, bkil;
        LOAD_ROW(k, bkrh, bkrl, bkih, bkil);
        if (nounit) {
            /* bk /= A[k,k] — compute scalar inverse once, broadcast,
             * SIMD-multiply (cheaper than vector inversion). */
            const TC inv = cdiv(one_cdd, A_(k, k));
            __m256d irh, irl, iih, iil;
            BCAST_T(inv, irh, irl, iih, iil);
            __m256d nrh, nrl, nih, nil;
            simd_fast::cmul(bkrh, bkrl, bkih, bkil,
                             irh, irl, iih, iil,
                             nrh, nrl, nih, nil);
            bkrh = nrh; bkrl = nrl; bkih = nih; bkil = nil;
            STORE_ROW(k, bkrh, bkrl, bkih, bkil);
        }
        for (std::ptrdiff_t i = k + 1; i < m; ++i) {
            __m256d arh, arl, aih, ail;
            BCAST_T(A_(i, k), arh, arl, aih, ail);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(arh, arl, aih, ail,
                             bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            simd_neg_cdd(prh, prl, pih, pil);
            __m256d birh, birl, biih, biil;
            LOAD_ROW(i, birh, birl, biih, biil);
            __m256d nrh, nrl, nih, nil;
            simd_fast::cadd(birh, birl, biih, biil,
                             prh, prl, pih, pil,
                             nrh, nrl, nih, nil);
            STORE_ROW(i, nrh, nrl, nih, nil);
        }
    }
}

/* (L, U, N): back sub. */
inline void simd_bwd_sub_lun_cdd(std::ptrdiff_t m, const TC *a, std::ptrdiff_t lda, bool nounit,
                                 double *brh, double *brl,
                                 double *bih, double *bil)
{
    for (std::ptrdiff_t k = m - 1; k >= 0; --k) {
        __m256d bkrh, bkrl, bkih, bkil;
        LOAD_ROW(k, bkrh, bkrl, bkih, bkil);
        if (nounit) {
            const TC inv = cdiv(one_cdd, A_(k, k));
            __m256d irh, irl, iih, iil;
            BCAST_T(inv, irh, irl, iih, iil);
            __m256d nrh, nrl, nih, nil;
            simd_fast::cmul(bkrh, bkrl, bkih, bkil,
                             irh, irl, iih, iil,
                             nrh, nrl, nih, nil);
            bkrh = nrh; bkrl = nrl; bkih = nih; bkil = nil;
            STORE_ROW(k, bkrh, bkrl, bkih, bkil);
        }
        for (std::ptrdiff_t i = 0; i < k; ++i) {
            __m256d arh, arl, aih, ail;
            BCAST_T(A_(i, k), arh, arl, aih, ail);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(arh, arl, aih, ail,
                             bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            simd_neg_cdd(prh, prl, pih, pil);
            __m256d birh, birl, biih, biil;
            LOAD_ROW(i, birh, birl, biih, biil);
            __m256d nrh, nrl, nih, nil;
            simd_fast::cadd(birh, birl, biih, biil,
                             prh, prl, pih, pil,
                             nrh, nrl, nih, nil);
            STORE_ROW(i, nrh, nrl, nih, nil);
        }
    }
}

/* (L, L, T) and (L, L, C): inner-product form on op(A)ᵀ. */
inline void simd_fwd_sub_lltc_cdd(std::ptrdiff_t m, const TC *a, std::ptrdiff_t lda, TC alpha,
                                  bool nounit, bool conj_flag,
                                  double *brh, double *brl,
                                  double *bih, double *bil)
{
    __m256d arh, arl, aih, ail;
    BCAST_T(alpha, arh, arl, aih, ail);
    for (std::ptrdiff_t i = m - 1; i >= 0; --i) {
        __m256d birh, birl, biih, biil;
        LOAD_ROW(i, birh, birl, biih, biil);
        __m256d trh, trl, tih, til;
        simd_fast::cmul(arh, arl, aih, ail,
                         birh, birl, biih, biil,
                         trh, trl, tih, til);
        for (std::ptrdiff_t k = i + 1; k < m; ++k) {
            const TC aki = conj_flag ? cconj(A_(k, i)) : A_(k, i);
            __m256d akrh, akrl, akih, akil;
            BCAST_T(aki, akrh, akrl, akih, akil);
            __m256d bkrh, bkrl, bkih, bkil;
            LOAD_ROW(k, bkrh, bkrl, bkih, bkil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(akrh, akrl, akih, akil,
                             bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            simd_neg_cdd(prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil;
            simd_fast::cadd(trh, trl, tih, til,
                             prh, prl, pih, pil,
                             nrh, nrl, nih, nil);
            trh = nrh; trl = nrl; tih = nih; til = nil;
        }
        if (nounit) {
            const TC aii = conj_flag ? cconj(A_(i, i)) : A_(i, i);
            const TC inv = cdiv(one_cdd, aii);
            __m256d irh, irl, iih, iil;
            BCAST_T(inv, irh, irl, iih, iil);
            __m256d nrh, nrl, nih, nil;
            simd_fast::cmul(trh, trl, tih, til,
                             irh, irl, iih, iil,
                             nrh, nrl, nih, nil);
            trh = nrh; trl = nrl; tih = nih; til = nil;
        }
        STORE_ROW(i, trh, trl, tih, til);
    }
}

inline void simd_bwd_sub_lutc_cdd(std::ptrdiff_t m, const TC *a, std::ptrdiff_t lda, TC alpha,
                                  bool nounit, bool conj_flag,
                                  double *brh, double *brl,
                                  double *bih, double *bil)
{
    __m256d arh, arl, aih, ail;
    BCAST_T(alpha, arh, arl, aih, ail);
    for (std::ptrdiff_t i = 0; i < m; ++i) {
        __m256d birh, birl, biih, biil;
        LOAD_ROW(i, birh, birl, biih, biil);
        __m256d trh, trl, tih, til;
        simd_fast::cmul(arh, arl, aih, ail,
                         birh, birl, biih, biil,
                         trh, trl, tih, til);
        for (std::ptrdiff_t k = 0; k < i; ++k) {
            const TC aki = conj_flag ? cconj(A_(k, i)) : A_(k, i);
            __m256d akrh, akrl, akih, akil;
            BCAST_T(aki, akrh, akrl, akih, akil);
            __m256d bkrh, bkrl, bkih, bkil;
            LOAD_ROW(k, bkrh, bkrl, bkih, bkil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(akrh, akrl, akih, akil,
                             bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            simd_neg_cdd(prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil;
            simd_fast::cadd(trh, trl, tih, til,
                             prh, prl, pih, pil,
                             nrh, nrl, nih, nil);
            trh = nrh; trl = nrl; tih = nih; til = nil;
        }
        if (nounit) {
            const TC aii = conj_flag ? cconj(A_(i, i)) : A_(i, i);
            const TC inv = cdiv(one_cdd, aii);
            __m256d irh, irl, iih, iil;
            BCAST_T(inv, irh, irl, iih, iil);
            __m256d nrh, nrl, nih, nil;
            simd_fast::cmul(trh, trl, tih, til,
                             irh, irl, iih, iil,
                             nrh, nrl, nih, nil);
            trh = nrh; trl = nrl; tih = nih; til = nil;
        }
        STORE_ROW(i, trh, trl, tih, til);
    }
}

#undef LOAD_ROW
#undef STORE_ROW
#undef BCAST_T

enum trsm_simd_cop { CSLLN, CSLUN, CSLLT, CSLUT, CSLLC, CSLUC };

inline void wtrsm_simd_diag(trsm_simd_cop op, std::ptrdiff_t j_start, std::ptrdiff_t j_end,
                            std::ptrdiff_t m, TC alpha,
                            const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    alignas(32) double brh[kMaxBlockM * kSimdLane];
    alignas(32) double brl[kMaxBlockM * kSimdLane];
    alignas(32) double bih[kMaxBlockM * kSimdLane];
    alignas(32) double bil[kMaxBlockM * kSimdLane];
    for (std::ptrdiff_t j = j_start; j < j_end; j += kSimdLane) {
        const std::ptrdiff_t jc = (j_end - j < kSimdLane) ? (j_end - j) : kSimdLane;
        pack_B_4col_complex(m, b, ldb, j, jc, brh, brl, bih, bil);
        switch (op) {
        case CSLLN:
            simd_prescale_complex(m, alpha, brh, brl, bih, bil);
            simd_fwd_sub_lln_cdd(m, a, lda, nounit, brh, brl, bih, bil);
            break;
        case CSLUN:
            simd_prescale_complex(m, alpha, brh, brl, bih, bil);
            simd_bwd_sub_lun_cdd(m, a, lda, nounit, brh, brl, bih, bil);
            break;
        case CSLLT:
            simd_fwd_sub_lltc_cdd(m, a, lda, alpha, nounit, 0, brh, brl, bih, bil);
            break;
        case CSLUT:
            simd_bwd_sub_lutc_cdd(m, a, lda, alpha, nounit, 0, brh, brl, bih, bil);
            break;
        case CSLLC:
            simd_fwd_sub_lltc_cdd(m, a, lda, alpha, nounit, 1, brh, brl, bih, bil);
            break;
        case CSLUC:
            simd_bwd_sub_lutc_cdd(m, a, lda, alpha, nounit, 1, brh, brl, bih, bil);
            break;
        }
        unpack_B_4col_complex(m, b, ldb, j, jc, brh, brl, bih, bil);
    }
}

/* ── SIDE='R' SIMD: 4-row chunks of B; column-walk trsm. ─────── */

using simd_exact::cload4;
using simd_exact::cstore4;

using simd_exact::vbcast;

/* RLN, RUN: column-walk j with α-scale then off-diag subtract then divide. */
inline void simd_wtrsm_r4_rln(std::ptrdiff_t ib, std::ptrdiff_t n, TC alpha,
                              const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    __m256d arh, arl, aih, ail;
    vbcast(alpha, arh, arl, aih, ail);
    const bool alpha_nontriv = !ceq1(alpha);
    for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
        TC *bj = b + static_cast<std::size_t>(j) * ldb;
        __m256d brh, brl, bih, bil;
        cload4(bj + ib, brh, brl, bih, bil);
        if (alpha_nontriv) {
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(brh, brl, bih, bil, arh, arl, aih, ail,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        for (std::ptrdiff_t k = j + 1; k < n; ++k) {
            const TC akj = A_(k, j);
            if (ceq0(akj)) continue;
            __m256d akrh, akrl, akih, akil;
            vbcast(akj, akrh, akrl, akih, akil);
            const TC *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkrh, bkrl, bkih, bkil;
            cload4(bk + ib, bkrh, bkrl, bkih, bkil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(akrh, akrl, akih, akil, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            simd_fast::neg(prh, prl);
            simd_fast::neg(pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(brh, brl, bih, bil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        if (nounit) {
            const TC inv = cdiv(one_cdd, A_(j, j));
            __m256d irh, irl, iih, iil;
            vbcast(inv, irh, irl, iih, iil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(brh, brl, bih, bil, irh, irl, iih, iil,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        cstore4(bj + ib, brh, brl, bih, bil);
    }
}

inline void simd_wtrsm_r4_run(std::ptrdiff_t ib, std::ptrdiff_t n, TC alpha,
                              const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    __m256d arh, arl, aih, ail;
    vbcast(alpha, arh, arl, aih, ail);
    const bool alpha_nontriv = !ceq1(alpha);
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        TC *bj = b + static_cast<std::size_t>(j) * ldb;
        __m256d brh, brl, bih, bil;
        cload4(bj + ib, brh, brl, bih, bil);
        if (alpha_nontriv) {
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(brh, brl, bih, bil, arh, arl, aih, ail,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        for (std::ptrdiff_t k = 0; k < j; ++k) {
            const TC akj = A_(k, j);
            if (ceq0(akj)) continue;
            __m256d akrh, akrl, akih, akil;
            vbcast(akj, akrh, akrl, akih, akil);
            const TC *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkrh, bkrl, bkih, bkil;
            cload4(bk + ib, bkrh, bkrl, bkih, bkil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(akrh, akrl, akih, akil, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            simd_fast::neg(prh, prl);
            simd_fast::neg(pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(brh, brl, bih, bil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        if (nounit) {
            const TC inv = cdiv(one_cdd, A_(j, j));
            __m256d irh, irl, iih, iil;
            vbcast(inv, irh, irl, iih, iil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(brh, brl, bih, bil, irh, irl, iih, iil,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        cstore4(bj + ib, brh, brl, bih, bil);
    }
}

/* RLT/RLC, RUT/RUC: column-walk k; divide-first then subtract from j != k. */
inline void simd_wtrsm_r4_rlTC(std::ptrdiff_t ib, std::ptrdiff_t n, TC alpha, bool conj_flag,
                               const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    __m256d arh, arl, aih, ail;
    vbcast(alpha, arh, arl, aih, ail);
    const bool alpha_nontriv = !ceq1(alpha);
    for (std::ptrdiff_t k = 0; k < n; ++k) {
        TC *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkrh, bkrl, bkih, bkil;
        cload4(bk + ib, bkrh, bkrl, bkih, bkil);
        if (nounit) {
            const TC inv = cdiv(one_cdd, A_op(a, lda, k, k, conj_flag));
            __m256d irh, irl, iih, iil;
            vbcast(inv, irh, irl, iih, iil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(bkrh, bkrl, bkih, bkil, irh, irl, iih, iil,
                             nrh, nrl, nih, nil_);
            bkrh = nrh; bkrl = nrl; bkih = nih; bkil = nil_;
            cstore4(bk + ib, bkrh, bkrl, bkih, bkil);
        }
        for (std::ptrdiff_t j = k + 1; j < n; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (ceq0(ajk)) continue;
            __m256d ajrh, ajrl, ajih, ajil;
            vbcast(ajk, ajrh, ajrl, ajih, ajil);
            TC *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjrh, bjrl, bjih, bjil;
            cload4(bj + ib, bjrh, bjrl, bjih, bjil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(ajrh, ajrl, ajih, ajil, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            simd_fast::neg(prh, prl);
            simd_fast::neg(pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(bjrh, bjrl, bjih, bjil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            cstore4(bj + ib, nrh, nrl, nih, nil_);
        }
        if (alpha_nontriv) {
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(bkrh, bkrl, bkih, bkil, arh, arl, aih, ail,
                             nrh, nrl, nih, nil_);
            cstore4(bk + ib, nrh, nrl, nih, nil_);
        }
    }
}

inline void simd_wtrsm_r4_ruTC(std::ptrdiff_t ib, std::ptrdiff_t n, TC alpha, bool conj_flag,
                               const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    __m256d arh, arl, aih, ail;
    vbcast(alpha, arh, arl, aih, ail);
    const bool alpha_nontriv = !ceq1(alpha);
    for (std::ptrdiff_t k = n - 1; k >= 0; --k) {
        TC *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkrh, bkrl, bkih, bkil;
        cload4(bk + ib, bkrh, bkrl, bkih, bkil);
        if (nounit) {
            const TC inv = cdiv(one_cdd, A_op(a, lda, k, k, conj_flag));
            __m256d irh, irl, iih, iil;
            vbcast(inv, irh, irl, iih, iil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(bkrh, bkrl, bkih, bkil, irh, irl, iih, iil,
                             nrh, nrl, nih, nil_);
            bkrh = nrh; bkrl = nrl; bkih = nih; bkil = nil_;
            cstore4(bk + ib, bkrh, bkrl, bkih, bkil);
        }
        for (std::ptrdiff_t j = 0; j < k; ++j) {
            const TC ajk = A_op(a, lda, j, k, conj_flag);
            if (ceq0(ajk)) continue;
            __m256d ajrh, ajrl, ajih, ajil;
            vbcast(ajk, ajrh, ajrl, ajih, ajil);
            TC *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjrh, bjrl, bjih, bjil;
            cload4(bj + ib, bjrh, bjrl, bjih, bjil);
            __m256d prh, prl, pih, pil;
            simd_fast::cmul(ajrh, ajrl, ajih, ajil, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            simd_fast::neg(prh, prl);
            simd_fast::neg(pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cadd(bjrh, bjrl, bjih, bjil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            cstore4(bj + ib, nrh, nrl, nih, nil_);
        }
        if (alpha_nontriv) {
            __m256d nrh, nrl, nih, nil_;
            simd_fast::cmul(bkrh, bkrl, bkih, bkil, arh, arl, aih, ail,
                             nrh, nrl, nih, nil_);
            cstore4(bk + ib, nrh, nrl, nih, nil_);
        }
    }
}

enum wtrsm_r_op { WTR_RLN, WTR_RUN, WTR_RLT, WTR_RUT, WTR_RLC, WTR_RUC };

inline void wtrsm_simd_diag_R(wtrsm_r_op op, std::ptrdiff_t m, std::ptrdiff_t n, TC alpha,
                              const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    const std::ptrdiff_t M4 = m & ~3;
    for (std::ptrdiff_t ib = 0; ib < M4; ib += 4) {
        switch (op) {
        case WTR_RLN: simd_wtrsm_r4_rln(ib, n, alpha, a, lda, b, ldb, nounit); break;
        case WTR_RUN: simd_wtrsm_r4_run(ib, n, alpha, a, lda, b, ldb, nounit); break;
        case WTR_RLT: simd_wtrsm_r4_rlTC(ib, n, alpha, 0, a, lda, b, ldb, nounit); break;
        case WTR_RUT: simd_wtrsm_r4_ruTC(ib, n, alpha, 0, a, lda, b, ldb, nounit); break;
        case WTR_RLC: simd_wtrsm_r4_rlTC(ib, n, alpha, 1, a, lda, b, ldb, nounit); break;
        case WTR_RUC: simd_wtrsm_r4_ruTC(ib, n, alpha, 1, a, lda, b, ldb, nounit); break;
        }
    }
    /* Scalar tail rows */
    if (M4 < m) {
        const std::ptrdiff_t Mt = m;
        auto subtract_col = [&](std::ptrdiff_t j, std::ptrdiff_t k, const TC &ajk_v) {
            if (ceq0(ajk_v)) return;
            for (std::ptrdiff_t i = M4; i < Mt; ++i)
                B_(i, j) = csub(B_(i, j), cmul(ajk_v, B_(i, k)));
        };
        switch (op) {
        case WTR_RLN:
            for (std::ptrdiff_t j = n - 1; j >= 0; --j) {
                if (!ceq1(alpha)) for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = cmul(B_(i, j), alpha);
                for (std::ptrdiff_t k = j + 1; k < n; ++k) subtract_col(j, k, A_(k, j));
                if (nounit) { const TC inv = cdiv(one_cdd, A_(j, j));
                    for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = cmul(B_(i, j), inv); }
            }
            break;
        case WTR_RUN:
            for (std::ptrdiff_t j = 0; j < n; ++j) {
                if (!ceq1(alpha)) for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = cmul(B_(i, j), alpha);
                for (std::ptrdiff_t k = 0; k < j; ++k) subtract_col(j, k, A_(k, j));
                if (nounit) { const TC inv = cdiv(one_cdd, A_(j, j));
                    for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, j) = cmul(B_(i, j), inv); }
            }
            break;
        case WTR_RLT: case WTR_RLC: {
            const std::ptrdiff_t cf = (op == WTR_RLC) ? 1 : 0;
            for (std::ptrdiff_t k = 0; k < n; ++k) {
                if (nounit) { const TC inv = cdiv(one_cdd, A_op(a, lda, k, k, cf));
                    for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, k) = cmul(B_(i, k), inv); }
                for (std::ptrdiff_t j = k + 1; j < n; ++j) subtract_col(j, k, A_op(a, lda, j, k, cf));
                if (!ceq1(alpha)) for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, k) = cmul(B_(i, k), alpha);
            }
        } break;
        case WTR_RUT: case WTR_RUC: {
            const std::ptrdiff_t cf = (op == WTR_RUC) ? 1 : 0;
            for (std::ptrdiff_t k = n - 1; k >= 0; --k) {
                if (nounit) { const TC inv = cdiv(one_cdd, A_op(a, lda, k, k, cf));
                    for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, k) = cmul(B_(i, k), inv); }
                for (std::ptrdiff_t j = 0; j < k; ++j) subtract_col(j, k, A_op(a, lda, j, k, cf));
                if (!ceq1(alpha)) for (std::ptrdiff_t i = M4; i < Mt; ++i) B_(i, k) = cmul(B_(i, k), alpha);
            }
        } break;
        }
    }
}

#pragma GCC pop_options

#endif  /* WBLAS_SIMD_DD */

/* ── Blocked SIDE='L' chunk worker: serial blocked-TRSM over one column slice
 * [j_start, j_end). The wgemm trailing update routes through wgemm_serial. */

inline void prescale_chunk(std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, TC alpha,
                           TC *b, std::ptrdiff_t ldb)
{
    if (ceq1(alpha)) return;
    if (ceq0(alpha)) {
        for (std::ptrdiff_t j = j_start; j < j_end; ++j)
            for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = zero_cdd;
        return;
    }
    for (std::ptrdiff_t j = j_start; j < j_end; ++j)
        for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = cmul(B_(i, j), alpha);
}

enum wtrsm_variant { WLLN, WLUN, WLLT, WLUT, WLLC, WLUC };

void blocked_chunk(wtrsm_variant V, std::ptrdiff_t j_start, std::ptrdiff_t j_end,
                   std::ptrdiff_t m, std::ptrdiff_t nb, TC alpha,
                   const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    const std::ptrdiff_t my_N = j_end - j_start;
    if (my_N <= 0) return;
    prescale_chunk(j_start, j_end, m, alpha, b, ldb);

    const TC m_one = TC{ R{-1.0, 0.0}, R{0.0, 0.0} };
    const TC one   = one_cdd;
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    TC *B_chunk = &B_(0, j_start);

    /* Diagonal-solve helper macros (alpha=1 since we prescaled). AVX2/FMA SIMD
     * path at runtime on Haswell+, scalar otherwise; both arms compiled. */
#ifdef WBLAS_SIMD_DD
#define DIAG_C(op_simd, scalar_core, ib_arg)                              \
    do {                                                                   \
        if (mf_have_avx2_fma() && (ib_arg) <= kMaxBlockM)                  \
            wtrsm_simd_diag(op_simd, j_start, j_end, (ib_arg), one,        \
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);    \
        else                                                               \
            scalar_core;                                                    \
    } while (0)
#else
#define DIAG_C(op_simd, scalar_core, ib_arg) do { scalar_core; } while (0)
#endif

    if (V == WLLN) {
        for (std::ptrdiff_t ic = 0; ic < m; ic += nb) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            if (ic > 0) {
                wgemm_serial(NN[0], NN[0], ib, my_N, ic, &m_one, &A_(ic, 0), lda, B_chunk, ldb, &one, &B_chunk[ic], ldb);
            }
            DIAG_C(CSLLN,
                wtrsm_lln_core(j_start, j_end, ib, one,
                               &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit),
                ib);
        }
    } else if (V == WLUN) {
        std::ptrdiff_t ic = ((m - 1) / nb) * nb;
        while (ic >= 0) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            const std::ptrdiff_t trailing = m - (ic + ib);
            if (trailing > 0) {
                const std::ptrdiff_t j0 = ic + ib;
                wgemm_serial(NN[0], NN[0], ib, my_N, trailing, &m_one, &A_(ic, j0), lda, &B_chunk[j0], ldb, &one, &B_chunk[ic], ldb);
            }
            DIAG_C(CSLUN,
                wtrsm_lun_core(j_start, j_end, ib, one,
                               &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit),
                ib);
            ic -= nb;
        }
    } else if (V == WLLT || V == WLLC) {
        const bool conj_flag = (V == WLLC) ? 1 : 0;
        const char *trans_gemm = conj_flag ? CN : TN;
        std::ptrdiff_t ic = ((m - 1) / nb) * nb;
        while (ic >= 0) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            const std::ptrdiff_t trailing = m - (ic + ib);
            if (trailing > 0) {
                const std::ptrdiff_t i0 = ic + ib;
                wgemm_serial(trans_gemm[0], NN[0], ib, my_N, trailing, &m_one, &A_(i0, ic), lda, &B_chunk[i0], ldb, &one, &B_chunk[ic], ldb);
            }
            DIAG_C((conj_flag ? CSLLC : CSLLT),
                wtrsm_lltc_core(j_start, j_end, ib, one,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb,
                                nounit, conj_flag),
                ib);
            ic -= nb;
        }
    } else { /* WLUT or WLUC */
        const bool conj_flag = (V == WLUC) ? 1 : 0;
        const char *trans_gemm = conj_flag ? CN : TN;
        for (std::ptrdiff_t ic = 0; ic < m; ic += nb) {
            const std::ptrdiff_t ib = (m - ic < nb) ? (m - ic) : nb;
            if (ic > 0) {
                wgemm_serial(trans_gemm[0], NN[0], ib, my_N, ic, &m_one, &A_(0, ic), lda, B_chunk, ldb, &one, &B_chunk[ic], ldb);
            }
            DIAG_C((conj_flag ? CSLUC : CSLUT),
                wtrsm_lutc_core(j_start, j_end, ib, one,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb,
                                nounit, conj_flag),
                ib);
        }
    }
#undef DIAG_C
}

/* Map (UPLO, TRANS) → blocked variant. */
inline wtrsm_variant w_variant(char UPLO, char TRANS) {
    if (TRANS == 'N') return (UPLO == 'L') ? WLLN : WLUN;
    if (TRANS == 'C') return (UPLO == 'L') ? WLLC : WLUC;
    return (UPLO == 'L') ? WLLT : WLUT;
}

}  // namespace

/* ── Exposed surface (wtrsm_kernel.h). ─────────────────────────────────── */

std::ptrdiff_t wtrsm_block_nb(void) { return trsm_nb(); }

void wtrsm_zero_B(std::ptrdiff_t m, std::ptrdiff_t n, TC *b, std::ptrdiff_t ldb)
{
    for (std::ptrdiff_t j = 0; j < n; ++j)
        for (std::ptrdiff_t i = 0; i < m; ++i) B_(i, j) = zero_cdd;
}

void wtrsm_L_slice(char UPLO, char TRANS, std::ptrdiff_t use_blocked,
                   std::ptrdiff_t j_start, std::ptrdiff_t j_end, std::ptrdiff_t m, std::ptrdiff_t nb, TC alpha,
                   const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    if (j_start >= j_end) return;
    const wtrsm_variant V = w_variant(UPLO, TRANS);
    if (use_blocked) {
        blocked_chunk(V, j_start, j_end, m, nb, alpha, a, lda, b, ldb, nounit);
        return;
    }
#ifdef WBLAS_SIMD_DD
    /* Unblocked but still SIMD: the 4-column packed-SoA diagonal kernels handle
     * any M <= kMaxBlockM, so the small-M path (below 2*nb, where blocking would
     * not pay) runs vectorized instead of the scalar cores. Same kernels the
     * blocked path already drives -> bitwise-identical to the serial sweep. */
    if (mf_have_avx2_fma() && m <= kMaxBlockM) {
        trsm_simd_cop cop;
        switch (V) {
        case WLLN: cop = CSLLN; break;
        case WLUN: cop = CSLUN; break;
        case WLLT: cop = CSLLT; break;
        case WLUT: cop = CSLUT; break;
        case WLLC: cop = CSLLC; break;
        default:   cop = CSLUC; break;  /* WLUC */
        }
        wtrsm_simd_diag(cop, j_start, j_end, m, alpha, a, lda, b, ldb, nounit);
        return;
    }
#endif
    switch (V) {
    case WLLN: wtrsm_lln_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit); break;
    case WLUN: wtrsm_lun_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit); break;
    case WLLT: wtrsm_lltc_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit, 0); break;
    case WLUT: wtrsm_lutc_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit, 0); break;
    case WLLC: wtrsm_lltc_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit, 1); break;
    case WLUC: wtrsm_lutc_core(j_start, j_end, m, alpha, a, lda, b, ldb, nounit, 1); break;
    }
}

void wtrsm_R_slice(char UPLO, char TRANS, std::ptrdiff_t row_lo, std::ptrdiff_t row_hi,
                   std::ptrdiff_t n, TC alpha,
                   const TC *a, std::ptrdiff_t lda, TC *b, std::ptrdiff_t ldb, bool nounit)
{
    const std::ptrdiff_t Mslice = row_hi - row_lo;
    if (Mslice <= 0) return;
    TC *b_slice = b + row_lo;
#ifdef WBLAS_SIMD_DD
    if (mf_have_avx2_fma()) {
        wtrsm_r_op op;
        if (TRANS == 'N')      op = (UPLO == 'L') ? WTR_RLN : WTR_RUN;
        else if (TRANS == 'C') op = (UPLO == 'L') ? WTR_RLC : WTR_RUC;
        else                op = (UPLO == 'L') ? WTR_RLT : WTR_RUT;
        wtrsm_simd_diag_R(op, Mslice, n, alpha, a, lda, b_slice, ldb, nounit);
        return;
    }
#endif
    if (TRANS == 'N') {
        if (UPLO == 'L') wtrsm_rln_core(Mslice, n, alpha, a, lda, b_slice, ldb, nounit);
        else             wtrsm_run_core(Mslice, n, alpha, a, lda, b_slice, ldb, nounit);
    } else if (TRANS == 'C') {
        if (UPLO == 'L') wtrsm_rltc_core(Mslice, n, alpha, a, lda, b_slice, ldb, nounit, 1);
        else             wtrsm_rutc_core(Mslice, n, alpha, a, lda, b_slice, ldb, nounit, 1);
    } else {
        if (UPLO == 'L') wtrsm_rltc_core(Mslice, n, alpha, a, lda, b_slice, ldb, nounit, 0);
        else             wtrsm_rutc_core(Mslice, n, alpha, a, lda, b_slice, ldb, nounit, 0);
    }
}

extern "C" void wtrsm_serial(
    char side, char uplo, char transa, char diag,
    std::ptrdiff_t m, std::ptrdiff_t n,
    const TC *alpha_,
    const TC *a, std::ptrdiff_t lda,
    TC *b, std::ptrdiff_t ldb)
{
    const TC alpha = *alpha_;
    using mf_util::up;  /* char flag uppercase — mf_util.h */
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);
    const char TRANS = up(&transa);
    const bool nounit = (up(&diag) != 'U');

    if (m == 0 || n == 0) return;

    if (ceq0(alpha)) { wtrsm_zero_B(m, n, b, ldb); return; }

    if (SIDE == 'L') {
        const std::ptrdiff_t nb = trsm_nb();
        const std::ptrdiff_t use_blocked = (m >= 2 * nb);
        wtrsm_L_slice(UPLO, TRANS, use_blocked, 0, n, m, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        wtrsm_R_slice(UPLO, TRANS, 0, m, n, alpha, a, lda, b, ldb, nounit);
    }
}

#undef A_
#undef B_
