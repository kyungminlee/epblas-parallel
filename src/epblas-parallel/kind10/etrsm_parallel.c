/*
 * etrsm_ — kind10 (REAL(KIND=10) / long double) triangular solve, the
 * public Fortran entry and threading-orchestration half of the etrsm
 * overlay (see etrsm_kernel.h for the split rationale; all the math
 * lives in etrsm_serial.c).
 *
 * Two parallel shapes, both partitioning a single axis across one team:
 *   SIDE='L'  — partition B's columns; each thread runs the serial
 *               blocked (or unblocked) solve on its own column slice.
 *   SIDE='R'  — partition B's rows; the diagonal walk stays serial per
 *               thread, rows are independent.
 *
 * One `omp parallel` per solve (not one per block), and egemm trailing
 * updates run single-thread (egemm_serial) inside each outer thread.
 *
 * Nesting guard: when etrsm_ is itself called from inside another
 * routine's parallel region, it delegates to etrsm_serial and opens no
 * region of its own — opening a nested team here trips the libgomp
 * barrier wedge (memory project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "etrsm_kernel.h"

typedef etrsm_T T;

/* Threshold below which OMP parallel-for on the partition axis isn't
 * worth the parallel-region setup. */
#define ETRSM_OMP_N_MIN 32

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE='L' standalone unblocked wrappers: wrap a core in this solve's
 *    own parallel region if N is big enough. Called when M < 2·nb (the
 *    blocked path doesn't kick in). ─────────────────────────────── */
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

TRSM_OMP_WRAPPER(trsm_lln, etrsm_lln_core)
TRSM_OMP_WRAPPER(trsm_lun, etrsm_lun_core)
TRSM_OMP_WRAPPER(trsm_llt, etrsm_llt_core)
TRSM_OMP_WRAPPER(trsm_lut, etrsm_lut_core)

/* ── Blocked SIDE='L': one outer `omp parallel` partitions B's columns
 *    across threads; each thread runs the serial blocked worker on its
 *    own column chunk. ─────────────────────────────────────────── */
static void blocked_dispatch(enum etrsm_variant V, int M, int N, T alpha,
                             const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = etrsm_nb();
#ifdef _OPENMP
    if (N >= ETRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int js  = ((long long)N * tid) / nt;
            int je  = ((long long)N * (tid + 1)) / nt;
            etrsm_blocked_chunk(V, js, je, M, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    etrsm_blocked_chunk(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
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

/* ── SIDE='R' wrappers: one parallel region partitions the M (row) axis.
 *    Gates on M (the partition axis) >= ETRSM_OMP_N_MIN. The entry guard
 *    already excludes nested calls, so no omp_in_parallel() check here. */
#ifdef _OPENMP
#define ETRSM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= ETRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
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

ETRSM_OMP_WRAP_R(trsm_rln, etrsm_rln_core)
ETRSM_OMP_WRAP_R(trsm_run, etrsm_run_core)
ETRSM_OMP_WRAP_R(trsm_rlt, etrsm_rlt_core)
ETRSM_OMP_WRAP_R(trsm_rut, etrsm_rut_core)

/* ── Entry point ──────────────────────────────────────────────── */

void etrsm_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        etrsm_serial(side, uplo, transa, diag, m_, n_, alpha_, a, lda_,
                     b, ldb_, side_len, uplo_len, transa_len, diag_len);
        return;
    }
#endif
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
        const int use_blocked = (M >= 2 * etrsm_nb());
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

#undef B_
