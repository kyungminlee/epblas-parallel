/*
 * etrsm — kind10 (REAL(KIND=10) / `long double`) triangular solve.
 *
 * Solves one of:
 *   op(A) · X = alpha · B          (SIDE='L')
 *   X · op(A) = alpha · B          (SIDE='R')
 *
 * where op(A) ∈ {A, Aᵀ, Aᴴ}; for real types Aᴴ ≡ Aᵀ. A is M×M (or
 * N×N) triangular (upper or lower; optionally unit-diagonal). B is
 * overwritten with the solution X.
 *
 * Stage 1 implementation: scalar unblocked, all 16 distinct algorithm
 * variants. Same rank-1-update loop structure as the upstream Netlib
 * reference DTRSM so the numerical behavior matches the migrated
 * archive to a tight tolerance.
 *
 * A blocked version (with `egemm` trailing updates) lands in a follow-
 * up commit. For unblocked the kernel is O(M²) per RHS column; the
 * blocked form gets the GEMM acceleration.
 *
 * Fortran ABI: same shape as the migrated entry point. Character args
 * have hidden trailing size_t length values appended; we ignore the
 * lengths and look at the first byte (upper-cased).
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

/* Threshold below which OMP parallel-for on the column axis isn't
 * worth the parallel-region setup. */
#define ETRSM_OMP_N_MIN 32

typedef long double T;

/* Block size for the SIDE='L' blocked path. Env-tunable; default
 * picked empirically to match egemm's natural panel sizing. */
static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}
static int g_nb_trsm = 0;
static int trsm_nb(void) {
    if (g_nb_trsm == 0) g_nb_trsm = env_int("ETRSM_NB", 64);
    return g_nb_trsm;
}

/* Local egemm declaration (the overlay's own egemm_ — symbol is in
 * our static archive). We call it for trailing-matrix updates. */
extern void egemm_(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

/* Convenience accessors over column-major A and B. */
#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE = 'L': solve op(A) X = α B, A is M×M, B is M×N ───────── */

/* Column-range "core" kernels: serial work over columns j in
 * [j_start, j_end). Used both standalone (wrapped in their own
 * parallel region) and inside the blocked path (where the outer
 * parallel region has already partitioned the column range). */

static inline void trsm_lln_core(int j_start, int j_end, int M, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != 1.0L) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != 0.0L) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = k + 1; i < M; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

static inline void trsm_lun_core(int j_start, int j_end, int M, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        if (alpha != 1.0L) for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != 0.0L) {
                if (nounit) B_(k, j) /= A_(k, k);
                const T bk = B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) -= bk * A_(i, k);
            }
        }
    }
}

static inline void trsm_llt_core(int j_start, int j_end, int M, T alpha,
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

static inline void trsm_lut_core(int j_start, int j_end, int M, T alpha,
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

/* Standalone unblocked entries: wrap in own parallel region if N is
 * big enough. Called when M < 2·nb (blocked path doesn't kick in). */
#ifdef _OPENMP
#define TRSM_OMP_WRAPPER(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit)       \
    {                                                                      \
        if (N >= ETRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {           \
            _Pragma("omp parallel")                                        \
            {                                                              \
                int tid = omp_get_thread_num();                            \
                int nt  = omp_get_num_threads();                           \
                int js  = ((long long)N * tid) / nt;                       \
                int je  = ((long long)N * (tid + 1)) / nt;                 \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);            \
            }                                                              \
        } else {                                                           \
            core(0, N, M, alpha, a, lda, b, ldb, nounit);                  \
        }                                                                  \
    }
#else
#define TRSM_OMP_WRAPPER(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit)       \
    {                                                                      \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                      \
    }
#endif

TRSM_OMP_WRAPPER(trsm_lln, trsm_lln_core)
TRSM_OMP_WRAPPER(trsm_lun, trsm_lun_core)
TRSM_OMP_WRAPPER(trsm_llt, trsm_llt_core)
TRSM_OMP_WRAPPER(trsm_lut, trsm_lut_core)

/* ── Blocked SIDE='L' variants: coarse-grain parallelism across N.
 *
 * One outer `#pragma omp parallel` partitions columns of B across
 * threads; each thread runs serial blocked-TRSM on its own column
 * chunk. egemm calls inside each thread automatically run with a
 * 1-thread inner team (OMP nesting disabled by default), so the
 * trailing-update is single-thread per outer thread — exactly the
 * pattern we want: 4 cores each doing serial gemm on their chunk
 * of B, near-perfect parallel scaling and one OMP setup for the
 * whole TRSM (not 16+ as before).
 *
 * The unblocked diagonal-block kernels are called via their _core
 * helpers (serial column-range version) directly inside the thread's
 * loop — the `#pragma omp for/parallel for` they used to spawn is
 * replaced by the outer team's manual partition.
 */

/* Apply alpha to a column slice [j_start, j_end) of B in place. */
static inline void prescale_chunk(int j_start, int j_end, int M, T alpha,
                                  T *b, int ldb)
{
    if (alpha == 1.0L) return;
    if (alpha == 0.0L) {
        for (int j = j_start; j < j_end; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = 0.0L;
        return;
    }
    for (int j = j_start; j < j_end; ++j)
        for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
}

/* One blocked-TRSM iteration body, shared across UPLO/TRANSA variants.
 * Caller passes a callback selector via enum + iteration direction. */

enum trsm_variant { LLN, LUN, LLT, LUT };

/* Per-thread serial blocked-TRSM on a column slice [j_start, j_end) of B. */
static void blocked_chunk(enum trsm_variant V, int j_start, int j_end,
                          int M, int nb, T alpha,
                          const T *a, int lda, T *b, int ldb, int nounit)
{
    const int my_N = j_end - j_start;
    if (my_N <= 0) return;
    prescale_chunk(j_start, j_end, M, alpha, b, ldb);

    const T m_one = -1.0L, one = 1.0L;
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    /* The egemm call sees only this thread's column slice — operates
     * on B starting at column j_start, my_N columns wide. */
    T *B_chunk = &B_(0, j_start);

    if (V == LLN) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                /* B[ic..ic+ib, j_start..j_end] -= A[ic..ic+ib, 0..ic] · B[0..ic, j_start..j_end] */
                egemm_(NN, NN, &ib, &my_N, &ic, &m_one,
                       &A_(ic, 0), &lda,
                       B_chunk, &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            trsm_lln_core(j_start, j_end, ib, 1.0L,
                          &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
        }
    } else if (V == LUN) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int j0 = ic + ib;
                egemm_(NN, NN, &ib, &my_N, &trailing, &m_one,
                       &A_(ic, j0), &lda,
                       &B_chunk[j0], &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            trsm_lun_core(j_start, j_end, ib, 1.0L,
                          &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            ic -= nb;
        }
    } else if (V == LLT) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int i0 = ic + ib;
                egemm_(TN, NN, &ib, &my_N, &trailing, &m_one,
                       &A_(i0, ic), &lda,
                       &B_chunk[i0], &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            trsm_llt_core(j_start, j_end, ib, 1.0L,
                          &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            ic -= nb;
        }
    } else { /* LUT */
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                egemm_(TN, NN, &ib, &my_N, &ic, &m_one,
                       &A_(0, ic), &lda,
                       B_chunk, &ldb, &one,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            trsm_lut_core(j_start, j_end, ib, 1.0L,
                          &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
        }
    }
}

static void blocked_dispatch(enum trsm_variant V, int M, int N, T alpha,
                             const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = trsm_nb();
#ifdef _OPENMP
    if (N >= ETRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int js  = ((long long)N * tid) / nt;
            int je  = ((long long)N * (tid + 1)) / nt;
            blocked_chunk(V, js, je, M, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    blocked_chunk(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
}

static void blocked_lln(int M, int N, T alpha,
                        const T *a, int lda, T *b, int ldb, int nounit) {
    blocked_dispatch(LLN, M, N, alpha, a, lda, b, ldb, nounit);
}
static void blocked_lun(int M, int N, T alpha,
                        const T *a, int lda, T *b, int ldb, int nounit) {
    blocked_dispatch(LUN, M, N, alpha, a, lda, b, ldb, nounit);
}
static void blocked_llt(int M, int N, T alpha,
                        const T *a, int lda, T *b, int ldb, int nounit) {
    blocked_dispatch(LLT, M, N, alpha, a, lda, b, ldb, nounit);
}
static void blocked_lut(int M, int N, T alpha,
                        const T *a, int lda, T *b, int ldb, int nounit) {
    blocked_dispatch(LUT, M, N, alpha, a, lda, b, ldb, nounit);
}

/* ── SIDE = 'R': solve X op(A) = α B, A is N×N, B is M×N ─────────
 *
 * R-side cores partition the M (row) axis: the j (column) loop walks
 * the diagonal serially (each B[:,j] depends on prior B[:,k]), but
 * within each step every row of B is processed identically. Each
 * thread owns a disjoint row slice [i_start, i_end) of B and reads
 * shared A read-only — race-free, no barriers needed. */

/* (R, L, N): solve X · A = α B, A lower triangular.
 * Equivalent to back-substitution on columns of B from j = N-1 down. */
static inline void trsm_rln_core(int i_start, int i_end, int N, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        if (alpha != 1.0L) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != 0.0L) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = 1.0L / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

/* (R, U, N): solve X · A = α B, A upper triangular.
 * Forward-sub on columns of B from j = 0 up. */
static inline void trsm_run_core(int i_start, int i_end, int N, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        if (alpha != 1.0L) for (int i = i_start; i < i_end; ++i) B_(i, j) *= alpha;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != 0.0L) {
                const T akj = A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) -= akj * B_(i, k);
            }
        }
        if (nounit) {
            const T inv = 1.0L / A_(j, j);
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= inv;
        }
    }
}

/* (R, L, T): solve X · Aᵀ = α B, A lower. */
static inline void trsm_rlt_core(int i_start, int i_end, int N, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = 0; k < N; ++k) {
        if (nounit) {
            const T inv = 1.0L / A_(k, k);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = k + 1; j < N; ++j) {
            if (A_(j, k) != 0.0L) {
                const T ajk = A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != 1.0L) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

/* (R, U, T): solve X · Aᵀ = α B, A upper. */
static inline void trsm_rut_core(int i_start, int i_end, int N, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = N - 1; k >= 0; --k) {
        if (nounit) {
            const T inv = 1.0L / A_(k, k);
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= inv;
        }
        for (int j = 0; j < k; ++j) {
            if (A_(j, k) != 0.0L) {
                const T ajk = A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) -= ajk * B_(i, k);
            }
        }
        if (alpha != 1.0L) for (int i = i_start; i < i_end; ++i) B_(i, k) *= alpha;
    }
}

/* OMP wrapper for R-side cores: one parallel region partitions the M
 * axis. Gates on M (the partition axis) >= ETRSM_OMP_N_MIN. */
#ifdef _OPENMP
#define ETRSM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= ETRSM_OMP_N_MIN && blas_omp_max_threads() > 1              \
                                && !omp_in_parallel()) {                    \
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
#define ETRSM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                       \
    }
#endif

ETRSM_OMP_WRAP_R(trsm_rln, trsm_rln_core)
ETRSM_OMP_WRAP_R(trsm_run, trsm_run_core)
ETRSM_OMP_WRAP_R(trsm_rlt, trsm_rlt_core)
ETRSM_OMP_WRAP_R(trsm_rut, trsm_rut_core)

/* ── Entry point ──────────────────────────────────────────────── */

void etrsm_(
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

    /* alpha == 0 quick return: B becomes all zeros. */
    if (alpha == 0.0L) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = 0.0L;
        return;
    }

    if (SIDE == 'L') {
        /* Blocked path when M is large enough to amortize the egemm
         * call overhead. Threshold is twice the block size — below
         * that, blocking only adds noise. */
        const int use_blocked = (M >= 2 * trsm_nb());
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_lln(M, N, alpha, a, lda, b, ldb, nounit);
                else             trsm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_lun(M, N, alpha, a, lda, b, ldb, nounit);
                else             trsm_lun(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else {
            if (UPLO == 'L') {
                if (use_blocked) blocked_llt(M, N, alpha, a, lda, b, ldb, nounit);
                else             trsm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_lut(M, N, alpha, a, lda, b, ldb, nounit);
                else             trsm_lut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') trsm_rln(M, N, alpha, a, lda, b, ldb, nounit);
            else             trsm_run(M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') trsm_rlt(M, N, alpha, a, lda, b, ldb, nounit);
            else             trsm_rut(M, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}

#undef A_
#undef B_
