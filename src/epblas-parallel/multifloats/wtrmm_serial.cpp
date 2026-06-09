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
#include "wgemm_kernel.h"   /* wgemm_serial for the trailing update */
#include <cstddef>
#include <cstdlib>
#include <cctype>

#ifdef MBLAS_SIMD_DD
#include "mgemm_simd_kernel.h"   /* cdd_mul, cdd_add primitives */
#include <immintrin.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {

int g_nb_trmm = 0;
int trmm_nb(void) {
    if (g_nb_trmm == 0) g_nb_trmm = 64;
    return g_nb_trmm;
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
inline T cconj(T const &a) { return T{ a.re, -a.im }; }

#define A_(i, j)  a[static_cast<std::size_t>(j) * lda + (i)]
#define B_(i, j)  b[static_cast<std::size_t>(j) * ldb + (i)]

inline T A_op(const T *a, int lda, int row, int col, int conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

#ifdef MBLAS_SIMD_DD

constexpr int kSimdLane = simd_dd::NR;
constexpr int kMaxBlockM = 128;

inline void pack_B_4col_cdd(int M, const T *b, int ldb, int j_start, int j_count,
                            double *rh, double *rl, double *ih, double *il)
{
    for (int j = 0; j < j_count; ++j) {
        const T *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (int i = 0; i < M; ++i) {
            rh[i * kSimdLane + j] = col[i].re.limbs[0];
            rl[i * kSimdLane + j] = col[i].re.limbs[1];
            ih[i * kSimdLane + j] = col[i].im.limbs[0];
            il[i * kSimdLane + j] = col[i].im.limbs[1];
        }
    }
    for (int j = j_count; j < kSimdLane; ++j)
        for (int i = 0; i < M; ++i) {
            rh[i * kSimdLane + j] = 0.0; rl[i * kSimdLane + j] = 0.0;
            ih[i * kSimdLane + j] = 0.0; il[i * kSimdLane + j] = 0.0;
        }
}

inline void unpack_B_4col_cdd(int M, T *b, int ldb, int j_start, int j_count,
                              const double *rh, const double *rl,
                              const double *ih, const double *il)
{
    for (int j = 0; j < j_count; ++j) {
        T *col = &b[static_cast<std::size_t>(j_start + j) * ldb];
        for (int i = 0; i < M; ++i) {
            col[i].re.limbs[0] = rh[i * kSimdLane + j];
            col[i].re.limbs[1] = rl[i * kSimdLane + j];
            col[i].im.limbs[0] = ih[i * kSimdLane + j];
            col[i].im.limbs[1] = il[i * kSimdLane + j];
        }
    }
}

/* Broadcast A(r,c) with optional conjugate (negate im if conj_flag). */
inline void broadcast_A_cdd(const T *a, int lda, int r, int c, int conj_flag,
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

inline void simd_wtrmm_lln(int M, const T *a, int lda, T alpha, int nounit,
                           double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (int k = M - 1; k >= 0; --k) {
        __m256d bkrh = _mm256_load_pd(&brh[k * kSimdLane]);
        __m256d bkrl = _mm256_load_pd(&brl[k * kSimdLane]);
        __m256d bkih = _mm256_load_pd(&bih[k * kSimdLane]);
        __m256d bkil = _mm256_load_pd(&bil[k * kSimdLane]);
        __m256d trh, trl, tih, til;
        simd_dd::cdd_mul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                         trh, trl, tih, til);
        for (int i = M - 1; i > k; --i) {
            __m256d akrh, akrl, akih, akil;
            broadcast_A_cdd(a, lda, i, k, 0, akrh, akrl, akih, akil);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(trh, trl, tih, til, akrh, akrl, akih, akil,
                             prh, prl, pih, pil);
            __m256d birh = _mm256_load_pd(&brh[i * kSimdLane]);
            __m256d birl = _mm256_load_pd(&brl[i * kSimdLane]);
            __m256d biih = _mm256_load_pd(&bih[i * kSimdLane]);
            __m256d biil = _mm256_load_pd(&bil[i * kSimdLane]);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(birh, birl, biih, biil, prh, prl, pih, pil,
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
            simd_dd::cdd_mul(trh, trl, tih, til, akrh, akrl, akih, akil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        _mm256_store_pd(&brh[k * kSimdLane], trh);
        _mm256_store_pd(&brl[k * kSimdLane], trl);
        _mm256_store_pd(&bih[k * kSimdLane], tih);
        _mm256_store_pd(&bil[k * kSimdLane], til);
    }
}

inline void simd_wtrmm_lun(int M, const T *a, int lda, T alpha, int nounit,
                           double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (int k = 0; k < M; ++k) {
        __m256d bkrh = _mm256_load_pd(&brh[k * kSimdLane]);
        __m256d bkrl = _mm256_load_pd(&brl[k * kSimdLane]);
        __m256d bkih = _mm256_load_pd(&bih[k * kSimdLane]);
        __m256d bkil = _mm256_load_pd(&bil[k * kSimdLane]);
        __m256d trh, trl, tih, til;
        simd_dd::cdd_mul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                         trh, trl, tih, til);
        for (int i = 0; i < k; ++i) {
            __m256d akrh, akrl, akih, akil;
            broadcast_A_cdd(a, lda, i, k, 0, akrh, akrl, akih, akil);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(trh, trl, tih, til, akrh, akrl, akih, akil,
                             prh, prl, pih, pil);
            __m256d birh = _mm256_load_pd(&brh[i * kSimdLane]);
            __m256d birl = _mm256_load_pd(&brl[i * kSimdLane]);
            __m256d biih = _mm256_load_pd(&bih[i * kSimdLane]);
            __m256d biil = _mm256_load_pd(&bil[i * kSimdLane]);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(birh, birl, biih, biil, prh, prl, pih, pil,
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
            simd_dd::cdd_mul(trh, trl, tih, til, akrh, akrl, akih, akil,
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
inline void simd_wtrmm_llTC(int M, const T *a, int lda, T alpha, int nounit,
                            int conj_flag,
                            double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (int i = 0; i < M; ++i) {
        __m256d trh = _mm256_load_pd(&brh[i * kSimdLane]);
        __m256d trl = _mm256_load_pd(&brl[i * kSimdLane]);
        __m256d tih = _mm256_load_pd(&bih[i * kSimdLane]);
        __m256d til = _mm256_load_pd(&bil[i * kSimdLane]);
        if (nounit) {
            __m256d aiih, aiil, aiiih, aiiil;
            broadcast_A_cdd(a, lda, i, i, conj_flag, aiih, aiil, aiiih, aiiil);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_mul(trh, trl, tih, til, aiih, aiil, aiiih, aiiil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        for (int k = i + 1; k < M; ++k) {
            __m256d akrh, akrl, akih, akil;
            broadcast_A_cdd(a, lda, k, i, conj_flag, akrh, akrl, akih, akil);
            __m256d bkrh = _mm256_load_pd(&brh[k * kSimdLane]);
            __m256d bkrl = _mm256_load_pd(&brl[k * kSimdLane]);
            __m256d bkih = _mm256_load_pd(&bih[k * kSimdLane]);
            __m256d bkil = _mm256_load_pd(&bil[k * kSimdLane]);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(akrh, akrl, akih, akil, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(trh, trl, tih, til, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        __m256d nrh, nrl, nih, nil_;
        simd_dd::cdd_mul(arh, arl, aih, ail, trh, trl, tih, til,
                         nrh, nrl, nih, nil_);
        _mm256_store_pd(&brh[i * kSimdLane], nrh);
        _mm256_store_pd(&brl[i * kSimdLane], nrl);
        _mm256_store_pd(&bih[i * kSimdLane], nih);
        _mm256_store_pd(&bil[i * kSimdLane], nil_);
    }
}

/* LU T/C: for i = M-1..0: t = B(i); if nounit: t *= A_op(i,i);
 * for k<i: t += A_op(k,i) · B(k); B(i) = alpha · t. */
inline void simd_wtrmm_luTC(int M, const T *a, int lda, T alpha, int nounit,
                            int conj_flag,
                            double *brh, double *brl, double *bih, double *bil)
{
    __m256d arh = _mm256_set1_pd(alpha.re.limbs[0]);
    __m256d arl = _mm256_set1_pd(alpha.re.limbs[1]);
    __m256d aih = _mm256_set1_pd(alpha.im.limbs[0]);
    __m256d ail = _mm256_set1_pd(alpha.im.limbs[1]);
    for (int i = M - 1; i >= 0; --i) {
        __m256d trh = _mm256_load_pd(&brh[i * kSimdLane]);
        __m256d trl = _mm256_load_pd(&brl[i * kSimdLane]);
        __m256d tih = _mm256_load_pd(&bih[i * kSimdLane]);
        __m256d til = _mm256_load_pd(&bil[i * kSimdLane]);
        if (nounit) {
            __m256d aiih, aiil, aiiih, aiiil;
            broadcast_A_cdd(a, lda, i, i, conj_flag, aiih, aiil, aiiih, aiiil);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_mul(trh, trl, tih, til, aiih, aiil, aiiih, aiiil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        for (int k = 0; k < i; ++k) {
            __m256d akrh, akrl, akih, akil;
            broadcast_A_cdd(a, lda, k, i, conj_flag, akrh, akrl, akih, akil);
            __m256d bkrh = _mm256_load_pd(&brh[k * kSimdLane]);
            __m256d bkrl = _mm256_load_pd(&brl[k * kSimdLane]);
            __m256d bkih = _mm256_load_pd(&bih[k * kSimdLane]);
            __m256d bkil = _mm256_load_pd(&bil[k * kSimdLane]);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(akrh, akrl, akih, akil, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(trh, trl, tih, til, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            trh = nrh; trl = nrl; tih = nih; til = nil_;
        }
        __m256d nrh, nrl, nih, nil_;
        simd_dd::cdd_mul(arh, arl, aih, ail, trh, trl, tih, til,
                         nrh, nrl, nih, nil_);
        _mm256_store_pd(&brh[i * kSimdLane], nrh);
        _mm256_store_pd(&brl[i * kSimdLane], nrl);
        _mm256_store_pd(&bih[i * kSimdLane], nih);
        _mm256_store_pd(&bil[i * kSimdLane], nil_);
    }
}

enum trmm_simd_op_w { WSLLN, WSLUN, WSLLT, WSLUT, WSLLC, WSLUC };

inline void wtrmm_simd_diag(trmm_simd_op_w op, int j_start, int j_end,
                            int M, T alpha,
                            const T *a, int lda, T *b, int ldb, int nounit)
{
    alignas(32) double brh[kMaxBlockM * kSimdLane];
    alignas(32) double brl[kMaxBlockM * kSimdLane];
    alignas(32) double bih[kMaxBlockM * kSimdLane];
    alignas(32) double bil[kMaxBlockM * kSimdLane];
    for (int j = j_start; j < j_end; j += kSimdLane) {
        const int jc = (j_end - j < kSimdLane) ? (j_end - j) : kSimdLane;
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

inline void wtrmm_lln_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = M - 1; k >= 0; --k) {
            if (!cdd_iszero(B_(k, j))) {
                T temp = cmul(alpha, B_(k, j));
                for (int i = M - 1; i > k; --i)
                    B_(i, j) = cadd(B_(i, j), cmul(temp, A_(i, k)));
                if (nounit) temp = cmul(temp, A_(k, k));
                B_(k, j) = temp;
            }
        }
    }
}

inline void wtrmm_lun_core(int j_start, int j_end, int M, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = 0; k < M; ++k) {
            if (!cdd_iszero(B_(k, j))) {
                T temp = cmul(alpha, B_(k, j));
                for (int i = 0; i < k; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(temp, A_(i, k)));
                if (nounit) temp = cmul(temp, A_(k, k));
                B_(k, j) = temp;
            }
        }
    }
}

inline void wtrmm_llTC_core(int j_start, int j_end, int M, T alpha,
                            const T *a, int lda, T *b, int ldb,
                            int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = B_(i, j);
            if (nounit) t = cmul(t, A_op(a, lda, i, i, conj_flag));
            for (int k = i + 1; k < M; ++k)
                t = cadd(t, cmul(A_op(a, lda, k, i, conj_flag), B_(k, j)));
            B_(i, j) = cmul(alpha, t);
        }
    }
}

inline void wtrmm_luTC_core(int j_start, int j_end, int M, T alpha,
                            const T *a, int lda, T *b, int ldb,
                            int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t = cmul(t, A_op(a, lda, i, i, conj_flag));
            for (int k = 0; k < i; ++k)
                t = cadd(t, cmul(A_op(a, lda, k, i, conj_flag), B_(k, j)));
            B_(i, j) = cmul(alpha, t);
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

#ifdef MBLAS_SIMD_DD

/* Forward decls for scalar tails (defined below). */
inline void wtrmm_rln_core(int, int, int, T, const T*, int, T*, int, int);
inline void wtrmm_run_core(int, int, int, T, const T*, int, T*, int, int);
inline void wtrmm_rlTC_core(int, int, int, T, const T*, int, T*, int, int, int);
inline void wtrmm_ruTC_core(int, int, int, T, const T*, int, T*, int, int, int);

inline void load_4cell_csoa(const T *col, int ofs,
                            __m256d &rh, __m256d &rl,
                            __m256d &ih, __m256d &il)
{
    __m256d v0 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs]));
    __m256d v1 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs + 1]));
    __m256d v2 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs + 2]));
    __m256d v3 = _mm256_loadu_pd(reinterpret_cast<const double*>(&col[ofs + 3]));
    __m256d t0 = _mm256_unpacklo_pd(v0, v1);
    __m256d t1 = _mm256_unpackhi_pd(v0, v1);
    __m256d t2 = _mm256_unpacklo_pd(v2, v3);
    __m256d t3 = _mm256_unpackhi_pd(v2, v3);
    rh = _mm256_permute2f128_pd(t0, t2, 0x20);
    rl = _mm256_permute2f128_pd(t1, t3, 0x20);
    ih = _mm256_permute2f128_pd(t0, t2, 0x31);
    il = _mm256_permute2f128_pd(t1, t3, 0x31);
}

inline void store_4cell_csoa(T *col, int ofs,
                             __m256d rh, __m256d rl,
                             __m256d ih, __m256d il)
{
    __m256d t0 = _mm256_unpacklo_pd(rh, rl);
    __m256d t1 = _mm256_unpackhi_pd(rh, rl);
    __m256d t2 = _mm256_unpacklo_pd(ih, il);
    __m256d t3 = _mm256_unpackhi_pd(ih, il);
    __m256d v0 = _mm256_permute2f128_pd(t0, t2, 0x20);
    __m256d v1 = _mm256_permute2f128_pd(t1, t3, 0x20);
    __m256d v2 = _mm256_permute2f128_pd(t0, t2, 0x31);
    __m256d v3 = _mm256_permute2f128_pd(t1, t3, 0x31);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs]),     v0);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs + 1]), v1);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs + 2]), v2);
    _mm256_storeu_pd(reinterpret_cast<double*>(&col[ofs + 3]), v3);
}

inline void broadcast_c4(const T &v, __m256d &rh, __m256d &rl, __m256d &ih, __m256d &il)
{
    rh = _mm256_set1_pd(v.re.limbs[0]); rl = _mm256_set1_pd(v.re.limbs[1]);
    ih = _mm256_set1_pd(v.im.limbs[0]); il = _mm256_set1_pd(v.im.limbs[1]);
}

inline void simd_wtrmm_r4_rln(int ib, int N, T alpha,
                              const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        T *bj = b + static_cast<std::size_t>(j) * ldb;
        T t = alpha;
        if (nounit) t = cmul(t, A_(j, j));
        __m256d brh, brl, bih, bil;
        load_4cell_csoa(bj, ib, brh, brl, bih, bil);
        if (!cdd_isone(t)) {
            __m256d trh, trl, tih, til;
            broadcast_c4(t, trh, trl, tih, til);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_mul(brh, brl, bih, bil, trh, trl, tih, til,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        for (int k = j + 1; k < N; ++k) {
            const T akj_v = A_(k, j);
            if (cdd_iszero(akj_v)) continue;
            const T akj = cmul(alpha, akj_v);
            __m256d arh, arl, aih, ail;
            broadcast_c4(akj, arh, arl, aih, ail);
            const T *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkrh, bkrl, bkih, bkil;
            load_4cell_csoa(bk, ib, bkrh, bkrl, bkih, bkil);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(brh, brl, bih, bil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        store_4cell_csoa(bj, ib, brh, brl, bih, bil);
    }
}

inline void simd_wtrmm_r4_run(int ib, int N, T alpha,
                              const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        T *bj = b + static_cast<std::size_t>(j) * ldb;
        T t = alpha;
        if (nounit) t = cmul(t, A_(j, j));
        __m256d brh, brl, bih, bil;
        load_4cell_csoa(bj, ib, brh, brl, bih, bil);
        if (!cdd_isone(t)) {
            __m256d trh, trl, tih, til;
            broadcast_c4(t, trh, trl, tih, til);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_mul(brh, brl, bih, bil, trh, trl, tih, til,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        for (int k = 0; k < j; ++k) {
            const T akj_v = A_(k, j);
            if (cdd_iszero(akj_v)) continue;
            const T akj = cmul(alpha, akj_v);
            __m256d arh, arl, aih, ail;
            broadcast_c4(akj, arh, arl, aih, ail);
            const T *bk = b + static_cast<std::size_t>(k) * ldb;
            __m256d bkrh, bkrl, bkih, bkil;
            load_4cell_csoa(bk, ib, bkrh, bkrl, bkih, bkil);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(brh, brl, bih, bil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            brh = nrh; brl = nrl; bih = nih; bil = nil_;
        }
        store_4cell_csoa(bj, ib, brh, brl, bih, bil);
    }
}

inline void simd_wtrmm_r4_rlTC(int ib, int N, T alpha, int conj_flag,
                               const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = N - 1; k >= 0; --k) {
        const T *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkrh, bkrl, bkih, bkil;
        load_4cell_csoa(bk, ib, bkrh, bkrl, bkih, bkil);
        for (int j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (cdd_iszero(ajk)) continue;
            const T scaled = cmul(alpha, ajk);
            __m256d arh, arl, aih, ail;
            broadcast_c4(scaled, arh, arl, aih, ail);
            T *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjrh, bjrl, bjih, bjil;
            load_4cell_csoa(bj, ib, bjrh, bjrl, bjih, bjil);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(bjrh, bjrl, bjih, bjil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            store_4cell_csoa(bj, ib, nrh, nrl, nih, nil_);
        }
        T t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!cdd_isone(t)) {
            __m256d trh, trl, tih, til;
            broadcast_c4(t, trh, trl, tih, til);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_mul(bkrh, bkrl, bkih, bkil, trh, trl, tih, til,
                             nrh, nrl, nih, nil_);
            store_4cell_csoa(const_cast<T*>(bk), ib, nrh, nrl, nih, nil_);
        }
    }
}

inline void simd_wtrmm_r4_ruTC(int ib, int N, T alpha, int conj_flag,
                               const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = 0; k < N; ++k) {
        const T *bk = b + static_cast<std::size_t>(k) * ldb;
        __m256d bkrh, bkrl, bkih, bkil;
        load_4cell_csoa(bk, ib, bkrh, bkrl, bkih, bkil);
        for (int j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (cdd_iszero(ajk)) continue;
            const T scaled = cmul(alpha, ajk);
            __m256d arh, arl, aih, ail;
            broadcast_c4(scaled, arh, arl, aih, ail);
            T *bj = b + static_cast<std::size_t>(j) * ldb;
            __m256d bjrh, bjrl, bjih, bjil;
            load_4cell_csoa(bj, ib, bjrh, bjrl, bjih, bjil);
            __m256d prh, prl, pih, pil;
            simd_dd::cdd_mul(arh, arl, aih, ail, bkrh, bkrl, bkih, bkil,
                             prh, prl, pih, pil);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_add(bjrh, bjrl, bjih, bjil, prh, prl, pih, pil,
                             nrh, nrl, nih, nil_);
            store_4cell_csoa(bj, ib, nrh, nrl, nih, nil_);
        }
        T t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!cdd_isone(t)) {
            __m256d trh, trl, tih, til;
            broadcast_c4(t, trh, trl, tih, til);
            __m256d nrh, nrl, nih, nil_;
            simd_dd::cdd_mul(bkrh, bkrl, bkih, bkil, trh, trl, tih, til,
                             nrh, nrl, nih, nil_);
            store_4cell_csoa(const_cast<T*>(bk), ib, nrh, nrl, nih, nil_);
        }
    }
}

enum wtrmm_r_op { WRLN_OP, WRUN_OP, WRLT_OP, WRUT_OP, WRLC_OP, WRUC_OP };

inline void wtrmm_simd_diag_R(wtrmm_r_op op, int i_start, int i_end, int N, T alpha,
                              const T *a, int lda, T *b, int ldb, int nounit)
{
    const int i4_end = i_start + ((i_end - i_start) & ~3);
    for (int ib = i_start; ib < i4_end; ib += 4) {
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

inline void wtrmm_rln_core(int i_start, int i_end, int N, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        T t = alpha;
        if (nounit) t = cmul(t, A_(j, j));
        if (!cdd_isone(t))
            for (int i = i_start; i < i_end; ++i) B_(i, j) = cmul(B_(i, j), t);
        for (int k = j + 1; k < N; ++k) {
            if (!cdd_iszero(A_(k, j))) {
                const T akj = cmul(alpha, A_(k, j));
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(akj, B_(i, k)));
            }
        }
    }
}

inline void wtrmm_run_core(int i_start, int i_end, int N, T alpha,
                           const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t = cmul(t, A_(j, j));
        if (!cdd_isone(t))
            for (int i = i_start; i < i_end; ++i) B_(i, j) = cmul(B_(i, j), t);
        for (int k = 0; k < j; ++k) {
            if (!cdd_iszero(A_(k, j))) {
                const T akj = cmul(alpha, A_(k, j));
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(akj, B_(i, k)));
            }
        }
    }
}

inline void wtrmm_rlTC_core(int i_start, int i_end, int N, T alpha,
                            const T *a, int lda, T *b, int ldb,
                            int nounit, int conj_flag)
{
    for (int k = N - 1; k >= 0; --k) {
        for (int j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (!cdd_iszero(ajk)) {
                const T scaled = cmul(alpha, ajk);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(scaled, B_(i, k)));
            }
        }
        T t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!cdd_isone(t))
            for (int i = i_start; i < i_end; ++i) B_(i, k) = cmul(B_(i, k), t);
    }
}

inline void wtrmm_ruTC_core(int i_start, int i_end, int N, T alpha,
                            const T *a, int lda, T *b, int ldb,
                            int nounit, int conj_flag)
{
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (!cdd_iszero(ajk)) {
                const T scaled = cmul(alpha, ajk);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) = cadd(B_(i, j), cmul(scaled, B_(i, k)));
            }
        }
        T t = alpha;
        if (nounit) t = cmul(t, A_op(a, lda, k, k, conj_flag));
        if (!cdd_isone(t))
            for (int i = i_start; i < i_end; ++i) B_(i, k) = cmul(B_(i, k), t);
    }
}

/* ── Blocked SIDE='L' chunk worker: serial blocked-TRMM over one column slice
 * [j_start, j_end). The wgemm trailing update routes through wgemm_serial. */

enum trmm_variant_L { WLLN, WLUN, WLLT, WLUT, WLLC, WLUC };

void blocked_chunk_L(trmm_variant_L V, int j_start, int j_end,
                     int M, int nb, T alpha,
                     const T *a, int lda, T *b, int ldb, int nounit)
{
    const int my_N = j_end - j_start;
    if (my_N <= 0) return;

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    T *B_chunk = &B_(0, j_start);

    if (V == WLLN) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                wtrmm_simd_diag(WSLLN, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_lln_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                wgemm_serial(NN, NN, &ib, &my_N, &ic, &alpha,
                             &A_(ic, 0), &lda,
                             B_chunk, &ldb, &one_cdd,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    } else if (V == WLUN) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                wtrmm_simd_diag(WSLUN, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_lun_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int j0 = ic + ib;
                wgemm_serial(NN, NN, &ib, &my_N, &trailing, &alpha,
                             &A_(ic, j0), &lda,
                             &B_chunk[j0], &ldb, &one_cdd,
                             &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else if (V == WLLT || V == WLLC) {
        const int conj_flag = (V == WLLC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                wtrmm_simd_diag(conj_flag ? WSLLC : WSLLT, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_llTC_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int i0 = ic + ib;
                wgemm_serial(gemm_trans, NN, &ib, &my_N, &trailing, &alpha,
                             &A_(i0, ic), &lda,
                             &B_chunk[i0], &ldb, &one_cdd,
                             &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else { /* WLUT or WLUC */
        const int conj_flag = (V == WLUC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
#ifdef MBLAS_SIMD_DD
            if (ib <= kMaxBlockM) {
                wtrmm_simd_diag(conj_flag ? WSLUC : WSLUT, j_start, j_end, ib, alpha,
                                &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            } else
#endif
            wtrmm_luTC_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            if (ic > 0) {
                wgemm_serial(gemm_trans, NN, &ib, &my_N, &ic, &alpha,
                             &A_(0, ic), &lda,
                             B_chunk, &ldb, &one_cdd,
                             &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    }
}

/* ── Blocked SIDE='R' chunk worker: serial blocked-TRMM over one row slice
 * [i_start, i_end). The wgemm trailing update routes through wgemm_serial. */

enum trmm_variant_R { WRLN, WRUN, WRLT, WRUT, WRLC, WRUC };

void blocked_chunk_R(trmm_variant_R V, int i_start, int i_end,
                     int N, int nb, T alpha,
                     const T *a, int lda, T *b, int ldb, int nounit)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    const int my_M = i_end - i_start;
    if (my_M <= 0) return;
    T *B_chunk = &B_(i_start, 0);

    if (V == WRLN) {
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            wtrmm_simd_diag_R(WRLN_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            wtrmm_rln_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#endif
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                wgemm_serial(NN, NN, &my_M, &jb, &trailing, &alpha,
                             &B_chunk[static_cast<std::size_t>(k0) * ldb], &ldb,
                             &A_(k0, jc), &lda, &one_cdd,
                             &B_chunk[static_cast<std::size_t>(jc) * ldb], &ldb, 1, 1);
            }
        }
    } else if (V == WRUN) {
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            wtrmm_simd_diag_R(WRUN_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            wtrmm_run_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#endif
            if (jc > 0) {
                wgemm_serial(NN, NN, &my_M, &jb, &jc, &alpha,
                             B_chunk, &ldb,
                             &A_(0, jc), &lda, &one_cdd,
                             &B_chunk[static_cast<std::size_t>(jc) * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else if (V == WRLT || V == WRLC) {
        const int conj_flag = (V == WRLC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            wtrmm_simd_diag_R(conj_flag ? WRLC_OP : WRLT_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            wtrmm_rlTC_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
#endif
            if (jc > 0) {
                wgemm_serial(NN, gemm_trans, &my_M, &jb, &jc, &alpha,
                             B_chunk, &ldb,
                             &A_(jc, 0), &lda, &one_cdd,
                             &B_chunk[static_cast<std::size_t>(jc) * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else { /* WRUT or WRUC */
        const int conj_flag = (V == WRUC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
#ifdef MBLAS_SIMD_DD
            wtrmm_simd_diag_R(conj_flag ? WRUC_OP : WRUT_OP, i_start, i_end, jb, alpha,
                              &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
#else
            wtrmm_ruTC_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
#endif
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                wgemm_serial(NN, gemm_trans, &my_M, &jb, &trailing, &alpha,
                             &B_chunk[static_cast<std::size_t>(k0) * ldb], &ldb,
                             &A_(jc, k0), &lda, &one_cdd,
                             &B_chunk[static_cast<std::size_t>(jc) * ldb], &ldb, 1, 1);
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

int wtrmm_block_nb(void) { return trmm_nb(); }

void wtrmm_zero_B(int M, int N, T *b, int ldb)
{
    for (int j = 0; j < N; ++j)
        for (int i = 0; i < M; ++i) B_(i, j) = zero_cdd;
}

void wtrmm_L_slice(char UPLO, char TR, int use_blocked,
                   int j_start, int j_end, int M, int nb, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
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

void wtrmm_R_slice(char UPLO, char TR, int use_blocked,
                   int row_lo, int row_hi, int N, int nb, T alpha,
                   const T *a, int lda, T *b, int ldb, int nounit)
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
    const char TR = up(transa);   /* complex: N/T/C kept distinct */
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (cdd_iszero(alpha)) { wtrmm_zero_B(M, N, b, ldb); return; }

    const int nb = trmm_nb();

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * nb);
        wtrmm_L_slice(UPLO, TR, use_blocked, 0, N, M, nb, alpha,
                      a, lda, b, ldb, nounit);
    } else {
        const int use_blocked = (N >= 2 * nb);
        wtrmm_R_slice(UPLO, TR, use_blocked, 0, M, N, nb, alpha,
                      a, lda, b, ldb, nounit);
    }
}

#undef A_
#undef B_
