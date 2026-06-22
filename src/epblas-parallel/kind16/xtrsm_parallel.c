/*
 * xtrsm_ / xtrsm_blocked_ — kind16 complex (COMPLEX(KIND=16) /
 * __complex128) triangular solve, public Fortran entries. THREADING
 * ORCHESTRATION ONLY: all numerics (uplo decode, conjugate / A_op helpers,
 * the eight solve cores) live in xtrsm_serial.c, shared through
 * xtrsm_kernel.h.
 *
 * Unblocked Netlib reference algorithm with OpenMP coarse-grain
 * parallelism. SIDE='L' partitions columns of B across threads;
 * SIDE='R' partitions rows of B. Both partition axes carry no
 * cross-thread dependence.
 *
 * No blocking / no xgemm trailing update in the default entry: at kind16,
 * every op lowers to a libquadmath call so blocking adds dispatch overhead
 * without accelerating the arithmetic. See doc/design.md §10.
 *
 * TRANSA='C' is handled as a distinct case from 'T' (conjugate vs
 * plain transpose).
 *
 * xtrsm_blocked_ adds a LAPACK-blocked SIDE='L' path wrapped in a SINGLE
 * `#pragma omp parallel` region; trailing updates route through
 * xgemm_serial.
 */

#include "xtrsm_kernel.h"
#include "xgemm_kernel.h"
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdlib.h>
#include <quadmath.h>
#include "../common/blas_omp.h"
#ifdef _OPENMP
#include <omp.h>
#endif

/* Column-parallel gate: when N (= nrhs for SIDE='L') reaches this many
 * columns and OpenMP is available, the per-core xtrsm dispatchers fan
 * out the column range across threads. Lowered from 32 to 2 once the
 * xtrsv-loop fast path took over for tiny nrhs — at nrhs ≥ 2, the
 * per-column work is many ms of libquadmath, vastly exceeding the
 * ~5 µs OpenMP fork-join cost. */
#define XTRSM_OMP_MIN 2

/* xtrsv-loop fast path: at SIDE='L', stride-1, M large enough that
 * xtrsv_ routes into its block-parallel variant, and nrhs small enough
 * that column-parallel xtrsm can't fill the thread pool, we decompose
 * xtrsm into nrhs serial xtrsv calls and let each call's internal
 * parallel xgemv carry the speedup. Crossover sits at nrhs ≈ the
 * xtrsv_blocked scaling factor — see xtrsm_xtrsv_loop_max() below. */
#define XTRSM_XTRSV_LOOP_M_MIN       128

/* xtrsv_blocked default nb (sub-block diagonal). Kept in sync with
 * XTRSV_BLOCKED_NB_DEFAULT in xtrsv.c — used here only to estimate
 * xtrsv_blocked's Amdahl ceiling = M/nb for the dispatch heuristic. */
#define XTRSM_XTRSV_LOOP_NB_HINT     64

typedef xtrsm_T T;

extern void xtrsv_core(
    char uplo, char trans, char diag,
    ptrdiff_t N,
    const T *restrict a, ptrdiff_t lda,
    T *restrict x, ptrdiff_t incx);

/* Maximum nrhs at which the xtrsv-loop fast path beats column-parallel
 * xtrsm. Derived from xtrsv_blocked's effective scaling factor:
 *
 *   scaling = min(nthreads, M / nb)
 *
 * xtrsv_blocked's serial sub-solve is nb / M of the total work, giving
 * an Amdahl ceiling of M/nb. At nthreads below that ceiling, scaling
 * is limited by thread count; above it, by the serial sub-solve. The
 * crossover where fast path stops winning vs col-parallel is at
 * nrhs = scaling — hence MAX = scaling - 1 (last nrhs where fast path
 * is still preferred). */
static ptrdiff_t xtrsm_xtrsv_loop_max(ptrdiff_t M) {
#ifdef _OPENMP
    const ptrdiff_t max_nt     = blas_omp_max_threads() - 1;
#else
    const ptrdiff_t max_nt     = 1 - 1;
#endif
    const ptrdiff_t max_amdahl = M / XTRSM_XTRSV_LOOP_NB_HINT;
    ptrdiff_t v = (max_nt < max_amdahl) ? max_nt : max_amdahl;
    if (v < 1) v = 1;
    return v;
}

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── OMP wrappers ──────────────────────────────────────────────────
 * The cores are the externals from xtrsm_kernel.h. */

#ifdef _OPENMP
#define XTRSM_OMP_WRAP_L(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     bool nounit) {                                          \
        if (N >= XTRSM_OMP_MIN && blas_omp_max_threads() > 1                 \
                              && !omp_in_parallel()) {                       \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t js  = blas_part_bound(N, tid, nth);                   \
                ptrdiff_t je  = blas_part_bound(N, tid + 1, nth);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }            \
    }
#define XTRSM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     bool nounit) {                                          \
        if (N >= XTRSM_OMP_MIN && blas_omp_max_threads() > 1                 \
                              && !omp_in_parallel()) {                       \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t js  = blas_part_bound(N, tid, nth);                   \
                ptrdiff_t je  = blas_part_bound(N, tid + 1, nth);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#define XTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     bool nounit) {                                          \
        if (M >= XTRSM_OMP_MIN && blas_omp_max_threads() > 1                 \
                              && !omp_in_parallel()) {                       \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t is  = blas_part_bound(M, tid, nth);                   \
                ptrdiff_t ie  = blas_part_bound(M, tid + 1, nth);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }            \
    }
#define XTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     bool nounit) {                                          \
        if (M >= XTRSM_OMP_MIN && blas_omp_max_threads() > 1                 \
                              && !omp_in_parallel()) {                       \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t is  = blas_part_bound(M, tid, nth);                   \
                ptrdiff_t ie  = blas_part_bound(M, tid + 1, nth);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#else
#define XTRSM_OMP_WRAP_L(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     bool nounit) {                                          \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#define XTRSM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     bool nounit) {                                          \
        core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#define XTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     bool nounit) {                                          \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                       \
    }
#define XTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                     \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb,        \
                     bool nounit) {                                          \
        core(0, M, N, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#endif

XTRSM_OMP_WRAP_L   (xtrsm_lln, xtrsm_lln_core)
XTRSM_OMP_WRAP_L   (xtrsm_lun, xtrsm_lun_core)
XTRSM_OMP_WRAP_L_TC(xtrsm_llt, xtrsm_llTC_core, 0)
XTRSM_OMP_WRAP_L_TC(xtrsm_lut, xtrsm_luTC_core, 0)
XTRSM_OMP_WRAP_L_TC(xtrsm_llc, xtrsm_llTC_core, 1)
XTRSM_OMP_WRAP_L_TC(xtrsm_luc, xtrsm_luTC_core, 1)
XTRSM_OMP_WRAP_R   (xtrsm_rln, xtrsm_rln_core)
XTRSM_OMP_WRAP_R   (xtrsm_run, xtrsm_run_core)
XTRSM_OMP_WRAP_R_TC(xtrsm_rlt, xtrsm_rlTC_core, 0)
XTRSM_OMP_WRAP_R_TC(xtrsm_rut, xtrsm_ruTC_core, 0)
XTRSM_OMP_WRAP_R_TC(xtrsm_rlc, xtrsm_rlTC_core, 1)
XTRSM_OMP_WRAP_R_TC(xtrsm_ruc, xtrsm_ruTC_core, 1)

/* ── Entry point ──────────────────────────────────────────────── */

static void xtrsm_core(
    char side, char uplo, char transa, char diag,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
    const T alpha = *alpha_;
    const char SIDE = xtrsm_uplo(side);
    const char UPLO = xtrsm_uplo(uplo);
    const char TR = xtrsm_uplo(transa);
    const bool nounit = (xtrsm_uplo(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        for (ptrdiff_t j = 0; j < N; ++j)
            for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }

    /* xtrsv-loop fast path. SIDE='L' is the only form where xtrsm
     * decomposes column-wise into xtrsv solves (b_j ← inv(op(A)) · b_j).
     * The M-threshold mirrors xtrsv's own block-parallel gate so we
     * only enter when each per-column call will actually parallelize. */
    {
#ifdef _OPENMP
        const bool xv_in_par = omp_in_parallel();
#else
        const bool xv_in_par = 0;
#endif
        const ptrdiff_t xv_max = xtrsm_xtrsv_loop_max(M);
        if (SIDE == 'L' && N >= 1 && N <= xv_max && M >= XTRSM_XTRSV_LOOP_M_MIN
            && !xv_in_par) {
            if (alpha != ONE) {
                for (ptrdiff_t j = 0; j < N; ++j)
                    for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) *= alpha;
            }
            for (ptrdiff_t j = 0; j < N; ++j) {
                xtrsv_core(uplo, transa, diag, M, a, lda, &B_(0, j), 1);
            }
            return;
        }
    }

    if (SIDE == 'L') {
        if (TR == 'N') {
            if (UPLO == 'L') xtrsm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_lun(M, N, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') xtrsm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_lut(M, N, alpha, a, lda, b, ldb, nounit);
        } else { /* 'C' */
            if (UPLO == 'L') xtrsm_llc(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_luc(M, N, alpha, a, lda, b, ldb, nounit);
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') xtrsm_rln(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_run(M, N, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') xtrsm_rlt(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_rut(M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') xtrsm_rlc(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrsm_ruc(M, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}

EPBLAS_FACADE_TRMM(xtrsm, T)

/* ── Block-parallel SIDE='L' variant ─────────────────────────────────
 *
 * LAPACK-blocked algorithm: walk the diagonal of A in NB×NB blocks;
 * for each block, solve the small diagonal sub-problem via the core,
 * then issue a trailing xgemm for the matrix update. The whole walk
 * lives inside ONE `#pragma omp parallel` region. Threads partition the
 * column range of B once and stay on their slice through pre-scale,
 * every diagonal sub-solve, and every trailing xgemm. No barriers
 * needed: each operation reads and writes only the thread's own column
 * slice, so the work is race-free even with no inter-thread sync.
 *
 * Inner work routes through xgemm_serial and the xtrsm_*_core helpers
 * (which have no OMP), so we never call an OMP-using function from
 * inside an OMP region.
 *
 * SIDE='R' and small-M (M < 2·NB) fall back to xtrsm_core, which has its
 * own omp_in_parallel guard so the fallback is safe even when called
 * from inside another parallel region.
 */

#define XTRSM_BLOCKED_NB_DEFAULT 64

static ptrdiff_t xtrsm_blocked_nb(void) {
    return XTRSM_BLOCKED_NB_DEFAULT;
}

static void xtrsm_blocked_core(
    char side, char uplo, char transa, char diag,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
    const T alpha = *alpha_;
    const ptrdiff_t nb = xtrsm_blocked_nb();
    const char SIDE = xtrsm_uplo(side);
    const char UPLO = xtrsm_uplo(uplo);
    const char TR   = xtrsm_uplo(transa);
    const bool nounit = (xtrsm_uplo(diag) != 'U');
    const bool cflag = (TR == 'C') ? 1 : 0;

    if (M == 0 || N == 0) return;

    if (SIDE != 'L' || M < 2 * nb) {
        xtrsm_core(side, uplo, transa, diag, M, N, alpha_, a, lda, b, ldb);
        return;
    }

    if (alpha == ZERO) {
        for (ptrdiff_t j = 0; j < N; ++j) for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }

    const T neg_one = -1.0Q + 0.0Qi;
    const char TTc = (TR == 'C') ? 'C' : 'T';
    const T one_v = ONE;

    const bool use_omp = (N >= 2 && blas_omp_should_thread());

#ifdef _OPENMP
    #pragma omp parallel if(use_omp)
#endif
    {
        ptrdiff_t tid = 0, nth = 1;
#ifdef _OPENMP
        if (use_omp) { tid = omp_get_thread_num(); nth = omp_get_num_threads(); }
#endif
        const ptrdiff_t j_lo = blas_part_bound(N, tid, nth);
        const ptrdiff_t j_hi = blas_part_bound(N, tid + 1, nth);
        const ptrdiff_t n_slice = j_hi - j_lo;

        if (n_slice > 0) {
            /* Pre-scale this thread's column slice of B by alpha. */
            if (alpha != ONE) {
                for (ptrdiff_t j = j_lo; j < j_hi; ++j) {
                    for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) *= alpha;
                }
            }

            if (TR == 'N' && UPLO == 'L') {
                /* LLN forward. */
                for (ptrdiff_t ic = 0; ic < M; ic += nb) {
                    ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
                    xtrsm_lln_core(j_lo, j_hi, ib, ONE,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    ptrdiff_t mt = M - ic - ib;
                    if (mt > 0) {
                        ptrdiff_t i0 = ic + ib;
                        xgemm_serial('N', 'N', mt, n_slice, ib, &neg_one,
                                     &A_(i0, ic), lda,
                                     &B_(ic, j_lo), ldb, &one_v,
                                     &B_(i0, j_lo), ldb);
                    }
                }
            } else if (TR == 'N' && UPLO == 'U') {
                /* LUN backward. */
                ptrdiff_t ic = ((M - 1) / nb) * nb;
                while (ic >= 0) {
                    ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
                    xtrsm_lun_core(j_lo, j_hi, ib, ONE,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    if (ic > 0) {
                        xgemm_serial('N', 'N', ic, n_slice, ib, &neg_one,
                                     &A_(0, ic), lda,
                                     &B_(ic, j_lo), ldb, &one_v,
                                     &B_(0, j_lo), ldb);
                    }
                    ic -= nb;
                }
            } else if (UPLO == 'L') {
                /* L,L,T/C: bottom-up walk. */
                ptrdiff_t ic = ((M - 1) / nb) * nb;
                while (ic >= 0) {
                    ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
                    xtrsm_llTC_core(j_lo, j_hi, ib, ONE,
                                    &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, cflag);
                    if (ic > 0) {
                        xgemm_serial(TTc, 'N', ic, n_slice, ib, &neg_one,
                                     &A_(ic, 0), lda,
                                     &B_(ic, j_lo), ldb, &one_v,
                                     &B_(0, j_lo), ldb);
                    }
                    ic -= nb;
                }
            } else {
                /* L,U,T/C: top-down walk. */
                for (ptrdiff_t ic = 0; ic < M; ic += nb) {
                    ptrdiff_t ib = (M - ic < nb) ? (M - ic) : nb;
                    xtrsm_luTC_core(j_lo, j_hi, ib, ONE,
                                    &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, cflag);
                    ptrdiff_t mt = M - ic - ib;
                    if (mt > 0) {
                        ptrdiff_t i0 = ic + ib;
                        xgemm_serial(TTc, 'N', mt, n_slice, ib, &neg_one,
                                     &A_(ic, i0), lda,
                                     &B_(ic, j_lo), ldb, &one_v,
                                     &B_(i0, j_lo), ldb);
                    }
                }
            }
        }
    }
}

EPBLAS_FACADE_TRMM(xtrsm_blocked, T)

#undef A_
#undef B_
