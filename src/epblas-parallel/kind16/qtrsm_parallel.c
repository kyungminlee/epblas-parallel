/*
 * qtrsm_ / qtrsm_blocked_ — kind16 (REAL(KIND=16) / __float128) triangular
 * solve, public Fortran entries. THREADING ORCHESTRATION ONLY: all numerics
 * (uplo decode + the eight solve cores) live in qtrsm_serial.c, shared
 * through qtrsm_kernel.h.
 *
 * Solves one of:
 *   op(A) · X = alpha · B          (SIDE='L')
 *   X · op(A) = alpha · B          (SIDE='R')
 *
 * where op(A) ∈ {A, Aᵀ}; for real types Aᴴ ≡ Aᵀ. B is overwritten with X.
 *
 * Three forms of parallelism:
 *
 *   - Column-parallel unblocked (qtrsm_, default): partitions B columns
 *     across threads (SIDE='L') or rows (SIDE='R'). One fork-join total.
 *     Defensive omp_in_parallel guard avoids nested parallelism.
 *
 *   - qtrsv-loop fast path: at SIDE='L', stride-1 B, M large enough that
 *     qtrsv routes into its block-parallel variant, and nrhs small
 *     enough that column-parallel xtrsm can't fill the thread pool,
 *     decomposes the call into nrhs serial qtrsv solves. Each qtrsv
 *     internally parallelizes via its trailing qgemv.
 *
 *   - Block-parallel qtrsm_blocked_: SIDE='L' only. LAPACK-blocked
 *     algorithm wrapped in a SINGLE `#pragma omp parallel` region.
 *     Threads partition B columns once and stay on their slice through
 *     pre-scale, every diagonal sub-solve, and every trailing qgemm.
 *     No barriers needed (each operation is local to the thread's
 *     column slice). Trailing updates route through qgemm_serial.
 *
 * Each public entry is a thin EPBLAS_FACADE_TRMM pair (name_ / name_64_)
 * over a by-value `*_core`; the cores take ptrdiff_t sizes so the LP64 and
 * ILP64 facades share one body.
 */

#include "qtrsm_kernel.h"
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdlib.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* Column-parallel gate: lowered from 32 to 2 once the qtrsv-loop fast
 * path took over for tiny nrhs. At nrhs ≥ 2 the per-column work is many
 * ms of libquadmath, vastly exceeding ~5 µs OpenMP fork-join cost. */
#define QTRSM_OMP_MIN 2

/* qtrsv-loop fast path thresholds (see qtrsm_qtrsv_loop_max below). */
#define QTRSM_QTRSV_LOOP_M_MIN       128
#define QTRSM_QTRSV_LOOP_NB_HINT     64

typedef qtrsm_T T;

extern void qtrsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx);

extern void qgemm_serial(
    char transa, char transb,
    ptrdiff_t M, ptrdiff_t N, ptrdiff_t K,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    const T *b, ptrdiff_t ldb,
    const T *beta_,
    T *c, ptrdiff_t ldc);

/* Maximum nrhs at which the qtrsv-loop fast path beats column-parallel
 * qtrsm. Derived from qtrsv_blocked's effective scaling:
 *   scaling = min(nthreads, M / nb)
 * Crossover where fast path stops winning vs col-parallel is at
 * nrhs ≈ scaling — hence MAX = scaling - 1 (last nrhs where fast path
 * is still preferred). */
static ptrdiff_t qtrsm_qtrsv_loop_max(ptrdiff_t M) {
#ifdef _OPENMP
    const ptrdiff_t max_nt     = blas_omp_max_threads() - 1;
#else
    const ptrdiff_t max_nt     = 1 - 1;
#endif
    const ptrdiff_t max_amdahl = M / QTRSM_QTRSV_LOOP_NB_HINT;
    ptrdiff_t v = (max_nt < max_amdahl) ? max_nt : max_amdahl;
    if (v < 1) v = 1;
    return v;
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── OMP wrappers: one parallel region per call, manual partition. ──
 * The cores are the externals from qtrsm_kernel.h. */

#ifdef _OPENMP
#define QTRSM_OMP_WRAP_L(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     int nounit) {                                          \
        if (N >= QTRSM_OMP_MIN && blas_omp_max_threads() > 1                \
                              && !omp_in_parallel()) {                      \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                            \
                int nt  = omp_get_num_threads();                           \
                ptrdiff_t js  = blas_part_bound(N, tid, nt);               \
                ptrdiff_t je  = blas_part_bound(N, tid + 1, nt);           \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);            \
            }                                                              \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }           \
    }
#define QTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     int nounit) {                                          \
        if (M >= QTRSM_OMP_MIN && blas_omp_max_threads() > 1                \
                              && !omp_in_parallel()) {                      \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                            \
                int nt  = omp_get_num_threads();                           \
                ptrdiff_t is  = blas_part_bound(M, tid, nt);               \
                ptrdiff_t ie  = blas_part_bound(M, tid + 1, nt);           \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);            \
            }                                                              \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }           \
    }
#else
#define QTRSM_OMP_WRAP_L(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     int nounit) {                                          \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#define QTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     int nounit) {                                          \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                       \
    }
#endif

QTRSM_OMP_WRAP_L(qtrsm_lln, qtrsm_lln_core)
QTRSM_OMP_WRAP_L(qtrsm_lun, qtrsm_lun_core)
QTRSM_OMP_WRAP_L(qtrsm_llt, qtrsm_llt_core)
QTRSM_OMP_WRAP_L(qtrsm_lut, qtrsm_lut_core)
QTRSM_OMP_WRAP_R(qtrsm_rln, qtrsm_rln_core)
QTRSM_OMP_WRAP_R(qtrsm_run, qtrsm_run_core)
QTRSM_OMP_WRAP_R(qtrsm_rlt, qtrsm_rlt_core)
QTRSM_OMP_WRAP_R(qtrsm_rut, qtrsm_rut_core)

/* ── Entry point ──────────────────────────────────────────────── */

static void qtrsm_core(
    char side, char uplo, char transa, char diag,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
    const T alpha = *alpha_;
    const char SIDE   = qtrsm_uplo(side);
    const char UPLO   = qtrsm_uplo(uplo);
    char TR           = qtrsm_uplo(transa);
    if (TR == 'C') TR = 'T';   /* real type: conj-trans ≡ trans */
    const int nounit = (qtrsm_uplo(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == 0.0Q) {
        for (ptrdiff_t j = 0; j < N; ++j)
            for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) = 0.0Q;
        return;
    }

    /* qtrsv-loop fast path. SIDE='L' is the only form where qtrsm
     * decomposes column-wise into qtrsv solves (b_j ← inv(op(A)) · b_j).
     * The M-threshold mirrors qtrsv's own block-parallel gate. */
    {
#ifdef _OPENMP
        const int xv_in_par = omp_in_parallel();
#else
        const int xv_in_par = 0;
#endif
        const ptrdiff_t xv_max = qtrsm_qtrsv_loop_max(M);
        if (SIDE == 'L' && N >= 1 && N <= xv_max && M >= QTRSM_QTRSV_LOOP_M_MIN
            && !xv_in_par) {
            if (alpha != 1.0Q) {
                for (ptrdiff_t j = 0; j < N; ++j)
                    for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) *= alpha;
            }
            for (ptrdiff_t j = 0; j < N; ++j) {
                qtrsv_core(uplo, transa, diag, M, a, lda, &B_(0, j), 1);
            }
            return;
        }
    }

    if (SIDE == 'L') {
        if (TR == 'N') {
            if (UPLO == 'L') qtrsm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            else             qtrsm_lun(M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') qtrsm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            else             qtrsm_lut(M, N, alpha, a, lda, b, ldb, nounit);
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') qtrsm_rln(M, N, alpha, a, lda, b, ldb, nounit);
            else             qtrsm_run(M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') qtrsm_rlt(M, N, alpha, a, lda, b, ldb, nounit);
            else             qtrsm_rut(M, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}

EPBLAS_FACADE_TRMM(qtrsm, T)

/* ── Block-parallel SIDE='L' variant ─────────────────────────────────
 *
 * LAPACK-blocked algorithm wrapped in a SINGLE `#pragma omp parallel`
 * region. Threads partition the column range of B once and stay on
 * their slice through pre-scale, every diagonal sub-solve, and every
 * trailing qgemm. No barriers needed: each operation reads and writes
 * only the thread's own column slice.
 *
 * SIDE='R' and small-M (M < 2·NB) fall back to qtrsm_core, which has its
 * own omp_in_parallel guard so the fallback is safe even when called
 * from inside another parallel region.
 */

#define QTRSM_BLOCKED_NB_DEFAULT 64

static int qtrsm_blocked_nb(void) {
    return QTRSM_BLOCKED_NB_DEFAULT;
}

static void qtrsm_blocked_core(
    char side, char uplo, char transa, char diag,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
    const T alpha = *alpha_;
    const ptrdiff_t nb = qtrsm_blocked_nb();
    const char SIDE = qtrsm_uplo(side);
    const char UPLO = qtrsm_uplo(uplo);
    char TR = qtrsm_uplo(transa);
    if (TR == 'C') TR = 'T';
    const int nounit = (qtrsm_uplo(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (SIDE != 'L' || M < 2 * nb) {
        qtrsm_core(side, uplo, transa, diag, M, N, alpha_, a, lda, b, ldb);
        return;
    }

    if (alpha == 0.0Q) {
        for (ptrdiff_t j = 0; j < N; ++j) for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) = 0.0Q;
        return;
    }

    const T neg_one = -1.0Q;
    const T one_v = 1.0Q;

#ifdef _OPENMP
    const int use_omp = (N >= 2 && blas_omp_max_threads() > 1 && !omp_in_parallel());
#else
    const int use_omp = 0;
#endif

#ifdef _OPENMP
    #pragma omp parallel if(use_omp)
#endif
    {
        int tid = 0, nt = 1;
#ifdef _OPENMP
        if (use_omp) { tid = omp_get_thread_num(); nt = omp_get_num_threads(); }
#endif
        const ptrdiff_t j_lo = blas_part_bound(N, tid, nt);
        const ptrdiff_t j_hi = blas_part_bound(N, tid + 1, nt);
        const ptrdiff_t n_slice = j_hi - j_lo;

        if (n_slice > 0) {
            /* Pre-scale this thread's column slice of B by alpha. */
            if (alpha != 1.0Q) {
                for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
                    for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) *= alpha;
                }
            }

            if (TR == 'N' && UPLO == 'L') {
                for (ptrdiff_t ic = 0; ic < M; ic += nb) {
                    ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
                    qtrsm_lln_core(j_lo, j_hi, ib, 1.0Q,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    ptrdiff_t mt = M - ic - ib;
                    if (mt > 0) {
                        ptrdiff_t i0 = ic + ib;
                        qgemm_serial('N', 'N', mt, n_slice, ib, &neg_one,
                                     &A_(i0, ic), lda,
                                     &B_(ic, j_lo), ldb, &one_v,
                                     &B_(i0, j_lo), ldb);
                    }
                }
            } else if (TR == 'N' && UPLO == 'U') {
                ptrdiff_t ic = ((M - 1) / nb) * nb;
                while (ic >= 0) {
                    ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
                    qtrsm_lun_core(j_lo, j_hi, ib, 1.0Q,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    if (ic > 0) {
                        qgemm_serial('N', 'N', ic, n_slice, ib, &neg_one,
                                     &A_(0, ic), lda,
                                     &B_(ic, j_lo), ldb, &one_v,
                                     &B_(0, j_lo), ldb);
                    }
                    ic -= nb;
                }
            } else if (UPLO == 'L') {
                /* L,L,T: bottom-up walk. */
                ptrdiff_t ic = ((M - 1) / nb) * nb;
                while (ic >= 0) {
                    ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
                    qtrsm_llt_core(j_lo, j_hi, ib, 1.0Q,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    if (ic > 0) {
                        qgemm_serial('T', 'N', ic, n_slice, ib, &neg_one,
                                     &A_(ic, 0), lda,
                                     &B_(ic, j_lo), ldb, &one_v,
                                     &B_(0, j_lo), ldb);
                    }
                    ic -= nb;
                }
            } else {
                /* L,U,T: top-down walk. */
                for (ptrdiff_t ic = 0; ic < M; ic += nb) {
                    ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
                    qtrsm_lut_core(j_lo, j_hi, ib, 1.0Q,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    ptrdiff_t mt = M - ic - ib;
                    if (mt > 0) {
                        ptrdiff_t i0 = ic + ib;
                        qgemm_serial('T', 'N', mt, n_slice, ib, &neg_one,
                                     &A_(ic, i0), lda,
                                     &B_(ic, j_lo), ldb, &one_v,
                                     &B_(i0, j_lo), ldb);
                    }
                }
            }
        }
    }
}

EPBLAS_FACADE_TRMM(qtrsm_blocked, T)

#undef A_
#undef B_
