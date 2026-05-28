/*
 * xtrsm — kind16 complex (COMPLEX(KIND=16) / `__complex128`)
 *         triangular solve.
 *
 * Unblocked Netlib reference algorithm with OpenMP coarse-grain
 * parallelism. SIDE='L' partitions columns of B across threads;
 * SIDE='R' partitions rows of B. Both partition axes carry no
 * cross-thread dependence.
 *
 * No blocking / no xgemm trailing update: at kind16, every op lowers
 * to a libquadmath call so blocking adds dispatch overhead without
 * accelerating the arithmetic. See doc/design.md §10.
 *
 * TRANSA='C' is handled as a distinct case from 'T' (conjugate vs
 * plain transpose).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
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

typedef __complex128 T;

extern void xtrsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n,
    const T *a, const int *lda,
    T *x, const int *incx,
    size_t uplo_len, size_t trans_len, size_t diag_len);

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
 * is still preferred).
 *
 * XTRSM_XTRSV_LOOP_MAX env overrides the heuristic. */
static int xtrsm_xtrsv_loop_max(int M) {
    static int env_set = 0;
    static int env_val = -1;
    if (!__atomic_load_n(&env_set, __ATOMIC_RELAXED)) {
        const char *s = getenv("XTRSM_XTRSV_LOOP_MAX");
        if (s && *s) {
            int v = atoi(s);
            if (v >= 0) env_val = v;
        }
        __atomic_store_n(&env_set, 1, __ATOMIC_RELAXED);
    }
    if (env_val >= 0) return env_val;

    const int max_nt     = blas_omp_max_threads() - 1;
    const int max_amdahl = M / XTRSM_XTRSV_LOOP_NB_HINT;
    int v = (max_nt < max_amdahl) ? max_nt : max_amdahl;
    if (v < 1) v = 1;
    return v;
}

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

static inline T cconj(T a) { return conjq(a); }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline T A_op(const T *a, int lda, int row, int col, int conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

static inline void xtrsm_lln_core(int j_start, int j_end, int M, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = k + 1; i < M; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

static inline void xtrsm_lun_core(int j_start, int j_end, int M, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != ONE) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != ZERO) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

static inline void xtrsm_llTC_core(int j_start, int j_end, int M, T alpha,
                                   const T *a, int lda, T *b, int ldb,
                                   int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (int k = i + 1; k < M; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

static inline void xtrsm_luTC_core(int j_start, int j_end, int M, T alpha,
                                   const T *a, int lda, T *b, int ldb,
                                   int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = alpha * B_(i, j);
            for (int k = 0; k < i; ++k) t -= A_op(a, lda, k, i, conj_flag) * B_(k, j);
            if (nounit) t /= A_op(a, lda, i, i, conj_flag);
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

static inline void xtrsm_rln_core(int i_start, int i_end, int N, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        if (alpha != ONE) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = ONE / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

static inline void xtrsm_run_core(int i_start, int i_end, int N, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        if (alpha != ONE) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = ONE / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

static inline void xtrsm_rlTC_core(int i_start, int i_end, int N, T alpha,
                                   const T *a, int lda, T *b, int ldb,
                                   int nounit, int conj_flag)
{
    for (int k = 0; k < N; ++k) {
        if (nounit) {
            const T inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != ONE) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

static inline void xtrsm_ruTC_core(int i_start, int i_end, int N, T alpha,
                                   const T *a, int lda, T *b, int ldb,
                                   int nounit, int conj_flag)
{
    for (int k = N - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = ONE / A_op(a, lda, k, k, conj_flag);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != ONE) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

/* ── OMP wrappers ────────────────────────────────────────────────── */

#ifdef _OPENMP
#define XTRSM_OMP_WRAP_L(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (N >= XTRSM_OMP_MIN && blas_omp_max_threads() > 1                 \
                              && !omp_in_parallel()) {                       \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = (int)((long long)N * tid / nt);                   \
                int je  = (int)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }            \
    }
#define XTRSM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (N >= XTRSM_OMP_MIN && blas_omp_max_threads() > 1                 \
                              && !omp_in_parallel()) {                       \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = (int)((long long)N * tid / nt);                   \
                int je  = (int)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#define XTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= XTRSM_OMP_MIN && blas_omp_max_threads() > 1                 \
                              && !omp_in_parallel()) {                       \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int is  = (int)((long long)M * tid / nt);                   \
                int ie  = (int)((long long)M * (tid + 1) / nt);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }            \
    }
#define XTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= XTRSM_OMP_MIN && blas_omp_max_threads() > 1                 \
                              && !omp_in_parallel()) {                       \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int is  = (int)((long long)M * tid / nt);                   \
                int ie  = (int)((long long)M * (tid + 1) / nt);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#else
#define XTRSM_OMP_WRAP_L(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#define XTRSM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#define XTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                       \
    }
#define XTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
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

void xtrsm_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)
{
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;
    const char SIDE = up(side);
    const char UPLO = up(uplo);
    const char TR = up(transa);
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }

    /* xtrsv-loop fast path. SIDE='L' is the only form where xtrsm
     * decomposes column-wise into xtrsv solves (b_j ← inv(op(A)) · b_j).
     * The M-threshold mirrors xtrsv_'s own block-parallel gate so we
     * only enter when each per-column call will actually parallelize. */
    {
#ifdef _OPENMP
        const int xv_in_par = omp_in_parallel();
#else
        const int xv_in_par = 0;
#endif
        const int xv_max = xtrsm_xtrsv_loop_max(M);
        if (SIDE == 'L' && N >= 1 && N <= xv_max && M >= XTRSM_XTRSV_LOOP_M_MIN
            && !xv_in_par) {
            if (alpha != ONE) {
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
            }
            const int incx_one = 1;
            for (int j = 0; j < N; ++j) {
                xtrsv_(uplo, transa, diag, m_, a, lda_,
                       &B_(0, j), &incx_one, 1, 1, 1);
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

/* ── Block-parallel SIDE='L' variant ─────────────────────────────────
 *
 * LAPACK-blocked algorithm: walk the diagonal of A in NB×NB blocks;
 * for each block, call the unblocked xtrsm_ on the small diagonal
 * sub-problem (this re-enters with M=ib and parallelizes over B's
 * columns internally), then issue a parallel xgemm for the trailing
 * matrix update.
 *
 * At kind16 every scalar op is a libquadmath call (~100 ns). The
 * existing xtrsm_ column-parallel scheme scales with N (cols of B);
 * the blocked variant adds parallelism across M (rows of B) through
 * the trailing xgemm, useful when N is small relative to thread count
 * or M dominates. Pre-scales B by alpha once (manually parallel),
 * then runs the block loop with alpha=1.
 *
 * SIDE='R' is left to the unblocked xtrsm_ — same structural reason
 * as SIDE='L' but the matrix-multiply trailing update needs xgemm
 * with the right transpose, and SIDE='R' is rarely the bottleneck.
 */

extern void xgemm_serial_(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

#define XTRSM_BLOCKED_NB_DEFAULT 64

static int xtrsm_blocked_nb(void) {
    static int cached = 0;
    if (cached == 0) {
        const char *s = getenv("XTRSM_NB");
        int v = (s && *s) ? atoi(s) : 0;
        cached = (v > 0) ? v : XTRSM_BLOCKED_NB_DEFAULT;
    }
    return cached;
}

/* Single-parallel-region xtrsm_blocked. SIDE='L', stride-1 on B.
 *
 * The whole walk lives inside ONE `#pragma omp parallel` region.
 * Threads partition the column range of B once and stay on their
 * slice through pre-scale, every diagonal sub-solve, and every
 * trailing xgemm. No barriers needed: each operation reads and
 * writes only the thread's own column slice, so the work is
 * race-free even with no inter-thread synchronization.
 *
 * Replaces the previous "prescale parallel-for + N/nb separate
 * xgemm fork-joins" shape with a single fork-join. Aligns with the
 * rule that we don't call OMP-using functions from inside an OMP
 * region: inner work routes through xgemm_serial_ and the static
 * xtrsm_*_core helpers (which have no OMP).
 *
 * SIDE='R' and small-M (M < 2·NB) fall back to xtrsm_, which has
 * its own omp_in_parallel guard so the fallback is safe even when
 * xtrsm_blocked_ is itself called from inside another parallel
 * region.
 */
void xtrsm_blocked_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)
{
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;
    const int nb = xtrsm_blocked_nb();
    const char SIDE = up(side);
    const char UPLO = up(uplo);
    const char TR   = up(transa);
    const int nounit = (up(diag) != 'U');
    const int cflag = (TR == 'C') ? 1 : 0;

    if (M == 0 || N == 0) return;

    if (SIDE != 'L' || M < 2 * nb) {
        xtrsm_(side, uplo, transa, diag, m_, n_, alpha_, a, lda_, b, ldb_,
               side_len, uplo_len, transa_len, diag_len);
        return;
    }

    if (alpha == ZERO) {
        for (int j = 0; j < N; ++j) for (int i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }

    const T neg_one = -1.0Q + 0.0Qi;
    const char NN[1] = {'N'};
    const char TT[1] = {(TR == 'C') ? 'C' : 'T'};
    const T one_v = ONE;

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
        const int j_lo = (int)((long long)N * tid / nt);
        const int j_hi = (int)((long long)N * (tid + 1) / nt);
        const int n_slice = j_hi - j_lo;

        if (n_slice > 0) {
            /* Pre-scale this thread's column slice of B by alpha. */
            if (alpha != ONE) {
                for (int j = j_lo; j < j_hi; ++j) {
                    for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
                }
            }

            if (TR == 'N' && UPLO == 'L') {
                /* LLN forward. */
                for (int ic = 0; ic < M; ic += nb) {
                    int ib = (M - ic < nb) ? (M - ic) : nb;
                    xtrsm_lln_core(j_lo, j_hi, ib, ONE,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    int mt = M - ic - ib;
                    if (mt > 0) {
                        int i0 = ic + ib;
                        xgemm_serial_(NN, NN, &mt, &n_slice, &ib, &neg_one,
                                      &A_(i0, ic), lda_,
                                      &B_(ic, j_lo), ldb_, &one_v,
                                      &B_(i0, j_lo), ldb_, 1, 1);
                    }
                }
            } else if (TR == 'N' && UPLO == 'U') {
                /* LUN backward. */
                int ic = ((M - 1) / nb) * nb;
                while (ic >= 0) {
                    int ib = (M - ic < nb) ? (M - ic) : nb;
                    xtrsm_lun_core(j_lo, j_hi, ib, ONE,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    if (ic > 0) {
                        xgemm_serial_(NN, NN, &ic, &n_slice, &ib, &neg_one,
                                      &A_(0, ic), lda_,
                                      &B_(ic, j_lo), ldb_, &one_v,
                                      &B_(0, j_lo), ldb_, 1, 1);
                    }
                    ic -= nb;
                }
            } else if (UPLO == 'L') {
                /* L,L,T/C: bottom-up walk. */
                int ic = ((M - 1) / nb) * nb;
                while (ic >= 0) {
                    int ib = (M - ic < nb) ? (M - ic) : nb;
                    xtrsm_llTC_core(j_lo, j_hi, ib, ONE,
                                    &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, cflag);
                    if (ic > 0) {
                        xgemm_serial_(TT, NN, &ic, &n_slice, &ib, &neg_one,
                                      &A_(ic, 0), lda_,
                                      &B_(ic, j_lo), ldb_, &one_v,
                                      &B_(0, j_lo), ldb_, 1, 1);
                    }
                    ic -= nb;
                }
            } else {
                /* L,U,T/C: top-down walk. */
                for (int ic = 0; ic < M; ic += nb) {
                    int ib = (M - ic < nb) ? (M - ic) : nb;
                    xtrsm_luTC_core(j_lo, j_hi, ib, ONE,
                                    &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, cflag);
                    int mt = M - ic - ib;
                    if (mt > 0) {
                        int i0 = ic + ib;
                        xgemm_serial_(TT, NN, &mt, &n_slice, &ib, &neg_one,
                                      &A_(ic, i0), lda_,
                                      &B_(ic, j_lo), ldb_, &one_v,
                                      &B_(i0, j_lo), ldb_, 1, 1);
                    }
                }
            }
        }
    }
}

#undef A_
#undef B_
