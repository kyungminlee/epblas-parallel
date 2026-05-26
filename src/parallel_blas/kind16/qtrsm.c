/*
 * qtrsm — kind16 (REAL(KIND=16) / `__float128`) triangular solve.
 *
 * Solves one of:
 *   op(A) · X = alpha · B          (SIDE='L')
 *   X · op(A) = alpha · B          (SIDE='R')
 *
 * where op(A) ∈ {A, Aᵀ}; for real types Aᴴ ≡ Aᵀ. A is M×M (or N×N)
 * triangular (upper or lower; optionally unit-diagonal). B is overwritten
 * with the solution X.
 *
 * Three forms of parallelism:
 *
 *   - Column-parallel unblocked (qtrsm_, default): partitions B columns
 *     across threads (SIDE='L') or rows (SIDE='R'). One fork-join total.
 *     Defensive omp_in_parallel guard avoids nested parallelism.
 *
 *   - qtrsv-loop fast path: at SIDE='L', stride-1 B, M large enough that
 *     qtrsv_ routes into its block-parallel variant, and nrhs small
 *     enough that column-parallel xtrsm can't fill the thread pool,
 *     decomposes the call into nrhs serial qtrsv solves. Each qtrsv
 *     internally parallelizes via its trailing qgemv.
 *
 *   - Block-parallel qtrsm_blocked_: SIDE='L' only. LAPACK-blocked
 *     algorithm wrapped in a SINGLE `#pragma omp parallel` region.
 *     Threads partition B columns once and stay on their slice through
 *     pre-scale, every diagonal sub-solve, and every trailing qgemm.
 *     No barriers needed (each operation is local to the thread's
 *     column slice).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* Column-parallel gate: lowered from 32 to 2 once the qtrsv-loop fast
 * path took over for tiny nrhs. At nrhs ≥ 2 the per-column work is many
 * ms of libquadmath, vastly exceeding ~5 µs OpenMP fork-join cost. */
#define QTRSM_OMP_MIN 2

/* qtrsv-loop fast path thresholds (see qtrsm_xtrsv_loop_max below). */
#define QTRSM_QTRSV_LOOP_M_MIN       128
#define QTRSM_QTRSV_LOOP_NB_HINT     64

typedef __float128 T;

extern void qtrsv_(
    const char *uplo, const char *trans, const char *diag,
    const int *n,
    const T *a, const int *lda,
    T *x, const int *incx,
    size_t uplo_len, size_t trans_len, size_t diag_len);

/* Maximum nrhs at which the qtrsv-loop fast path beats column-parallel
 * qtrsm. Derived from qtrsv_blocked's effective scaling:
 *   scaling = min(nthreads, M / nb)
 * Crossover where fast path stops winning vs col-parallel is at
 * nrhs ≈ scaling — hence MAX = scaling - 1 (last nrhs where fast path
 * is still preferred). QTRSM_QTRSV_LOOP_MAX env overrides the heuristic. */
static int qtrsm_qtrsv_loop_max(int M) {
    static int env_set = 0;
    static int env_val = -1;
    if (!__atomic_load_n(&env_set, __ATOMIC_RELAXED)) {
        const char *s = getenv("QTRSM_QTRSV_LOOP_MAX");
        if (s && *s) {
            int v = atoi(s);
            if (v >= 0) env_val = v;
        }
        __atomic_store_n(&env_set, 1, __ATOMIC_RELAXED);
    }
    if (env_val >= 0) return env_val;

    const int max_nt     = blas_omp_max_threads() - 1;
    const int max_amdahl = M / QTRSM_QTRSV_LOOP_NB_HINT;
    int v = (max_nt < max_amdahl) ? max_nt : max_amdahl;
    if (v < 1) v = 1;
    return v;
}

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

static inline void qtrsm_lln_core(int j_start, int j_end, int M, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != 1.0Q) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != 0.0Q) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = k + 1; i < M; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

static inline void qtrsm_lun_core(int j_start, int j_end, int M, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != 1.0Q) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != 0.0Q) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

static inline void qtrsm_llt_core(int j_start, int j_end, int M, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = alpha * B_(i, j);
            for (int k = i + 1; k < M; ++k) t -= A_(k, i) * B_(k, j);
            if (nounit) t /= A_(i, i);
            B_(i, j) = t;
        }
    }
}

static inline void qtrsm_lut_core(int j_start, int j_end, int M, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = alpha * B_(i, j);
            for (int k = 0; k < i; ++k) t -= A_(k, i) * B_(k, j);
            if (nounit) t /= A_(i, i);
            B_(i, j) = t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

static inline void qtrsm_rln_core(int i_start, int i_end, int N, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        if (alpha != 1.0Q) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = 1.0Q / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

static inline void qtrsm_run_core(int i_start, int i_end, int N, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        if (alpha != 1.0Q) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = 1.0Q / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

static inline void qtrsm_rlt_core(int i_start, int i_end, int N, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = 0; k < N; ++k) {
        if (nounit) {
            const T inv = 1.0Q / A_(k, k);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = k + 1; j < N; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = A_(j, k);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != 1.0Q) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

static inline void qtrsm_rut_core(int i_start, int i_end, int N, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = N - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = 1.0Q / A_(k, k);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = 0; j < k; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = A_(j, k);
                for (int i = i_start; i < i_end; ++i) B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != 1.0Q) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

/* ── OMP wrappers: one parallel region per call, manual partition. ── */

#ifdef _OPENMP
#define QTRSM_OMP_WRAP_L(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (N >= QTRSM_OMP_MIN && blas_omp_max_threads() > 1                \
                              && !omp_in_parallel()) {                      \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = (int)((long long)N * tid / nt);                   \
                int je  = (int)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }            \
    }
#define QTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= QTRSM_OMP_MIN && blas_omp_max_threads() > 1                \
                              && !omp_in_parallel()) {                      \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int is  = (int)((long long)M * tid / nt);                   \
                int ie  = (int)((long long)M * (tid + 1) / nt);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }            \
    }
#else
#define QTRSM_OMP_WRAP_L(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#define QTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
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

void qtrsm_(
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
    const char SIDE   = up(side);
    const char UPLO   = up(uplo);
    char TR           = up(transa);
    if (TR == 'C') TR = 'T';   /* real type: conj-trans ≡ trans */
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == 0.0Q) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = 0.0Q;
        return;
    }

    /* qtrsv-loop fast path. SIDE='L' is the only form where qtrsm
     * decomposes column-wise into qtrsv solves (b_j ← inv(op(A)) · b_j).
     * The M-threshold mirrors qtrsv_'s own block-parallel gate. */
    {
#ifdef _OPENMP
        const int xv_in_par = omp_in_parallel();
#else
        const int xv_in_par = 0;
#endif
        const int xv_max = qtrsm_qtrsv_loop_max(M);
        if (SIDE == 'L' && N >= 1 && N <= xv_max && M >= QTRSM_QTRSV_LOOP_M_MIN
            && !xv_in_par) {
            if (alpha != 1.0Q) {
                for (int j = 0; j < N; ++j)
                    for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
            }
            const int incx_one = 1;
            for (int j = 0; j < N; ++j) {
                qtrsv_(uplo, transa, diag, m_, a, lda_,
                       &B_(0, j), &incx_one, 1, 1, 1);
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

/* ── Block-parallel SIDE='L' variant ─────────────────────────────────
 *
 * LAPACK-blocked algorithm wrapped in a SINGLE `#pragma omp parallel`
 * region. Threads partition the column range of B once and stay on
 * their slice through pre-scale, every diagonal sub-solve, and every
 * trailing qgemm. No barriers needed: each operation reads and writes
 * only the thread's own column slice.
 *
 * SIDE='R' and small-M (M < 2·NB) fall back to qtrsm_, which has its
 * own omp_in_parallel guard so the fallback is safe even when called
 * from inside another parallel region.
 */

extern void qgemm_serial_(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

#define QTRSM_BLOCKED_NB_DEFAULT 64

static int qtrsm_blocked_nb(void) {
    static int cached = 0;
    if (cached == 0) {
        const char *s = getenv("QTRSM_NB");
        int v = (s && *s) ? atoi(s) : 0;
        cached = (v > 0) ? v : QTRSM_BLOCKED_NB_DEFAULT;
    }
    return cached;
}

void qtrsm_blocked_(
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
    const int nb = qtrsm_blocked_nb();
    const char SIDE = up(side);
    const char UPLO = up(uplo);
    char TR = up(transa);
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (SIDE != 'L' || M < 2 * nb) {
        qtrsm_(side, uplo, transa, diag, m_, n_, alpha_, a, lda_, b, ldb_,
               side_len, uplo_len, transa_len, diag_len);
        return;
    }

    if (alpha == 0.0Q) {
        for (int j = 0; j < N; ++j) for (int i = 0; i < M; ++i) B_(i, j) = 0.0Q;
        return;
    }

    const T neg_one = -1.0Q;
    const char NN[1] = {'N'};
    const char TT[1] = {'T'};
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
        const int j_lo = (int)((long long)N * tid / nt);
        const int j_hi = (int)((long long)N * (tid + 1) / nt);
        const int n_slice = j_hi - j_lo;

        if (n_slice > 0) {
            /* Pre-scale this thread's column slice of B by alpha. */
            if (alpha != 1.0Q) {
                for (int j = j_lo; j < j_hi; ++j) {
                    for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
                }
            }

            if (TR == 'N' && UPLO == 'L') {
                for (int ic = 0; ic < M; ic += nb) {
                    int ib = (M - ic < nb) ? (M - ic) : nb;
                    qtrsm_lln_core(j_lo, j_hi, ib, 1.0Q,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    int mt = M - ic - ib;
                    if (mt > 0) {
                        int i0 = ic + ib;
                        qgemm_serial_(NN, NN, &mt, &n_slice, &ib, &neg_one,
                                      &A_(i0, ic), lda_,
                                      &B_(ic, j_lo), ldb_, &one_v,
                                      &B_(i0, j_lo), ldb_, 1, 1);
                    }
                }
            } else if (TR == 'N' && UPLO == 'U') {
                int ic = ((M - 1) / nb) * nb;
                while (ic >= 0) {
                    int ib = (M - ic < nb) ? (M - ic) : nb;
                    qtrsm_lun_core(j_lo, j_hi, ib, 1.0Q,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    if (ic > 0) {
                        qgemm_serial_(NN, NN, &ic, &n_slice, &ib, &neg_one,
                                      &A_(0, ic), lda_,
                                      &B_(ic, j_lo), ldb_, &one_v,
                                      &B_(0, j_lo), ldb_, 1, 1);
                    }
                    ic -= nb;
                }
            } else if (UPLO == 'L') {
                /* L,L,T: bottom-up walk. */
                int ic = ((M - 1) / nb) * nb;
                while (ic >= 0) {
                    int ib = (M - ic < nb) ? (M - ic) : nb;
                    qtrsm_llt_core(j_lo, j_hi, ib, 1.0Q,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    if (ic > 0) {
                        qgemm_serial_(TT, NN, &ic, &n_slice, &ib, &neg_one,
                                      &A_(ic, 0), lda_,
                                      &B_(ic, j_lo), ldb_, &one_v,
                                      &B_(0, j_lo), ldb_, 1, 1);
                    }
                    ic -= nb;
                }
            } else {
                /* L,U,T: top-down walk. */
                for (int ic = 0; ic < M; ic += nb) {
                    int ib = (M - ic < nb) ? (M - ic) : nb;
                    qtrsm_lut_core(j_lo, j_hi, ib, 1.0Q,
                                   &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
                    int mt = M - ic - ib;
                    if (mt > 0) {
                        int i0 = ic + ib;
                        qgemm_serial_(TT, NN, &mt, &n_slice, &ib, &neg_one,
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
