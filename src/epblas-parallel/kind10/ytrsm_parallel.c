/*
 * ytrsm_ — kind10 complex (COMPLEX(KIND=10) / _Complex long double)
 * triangular solve, the public Fortran entry and threading-orchestration
 * half of the ytrsm overlay (see ytrsm_kernel.h for the split rationale;
 * all the math lives in ytrsm_serial.c).
 *
 * Two parallel shapes, both partitioning a single axis across one team:
 *   SIDE='L'  — partition B's columns; each thread runs the serial
 *               blocked (or unblocked) solve on its own column slice.
 *   SIDE='R'  — partition B's rows; the diagonal walk stays serial per
 *               thread, rows are independent.
 *
 * One `omp parallel` per solve (not one per block), and ygemm trailing
 * updates run single-thread (ygemm_serial) inside each outer thread.
 *
 * Nesting guard: when ytrsm_ is itself called from inside another
 * routine's parallel region, it delegates to ytrsm_serial and opens no
 * region of its own — opening a nested team here trips the libgomp
 * barrier wedge (memory project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ytrsm_kernel.h"

typedef ytrsm_T T;

/* Threshold below which OMP parallel-for on the partition axis isn't
 * worth the parallel-region setup. */
#define YTRSM_OMP_N_MIN 32

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0L + 0.0Li;

#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE='L' standalone unblocked wrappers: wrap a core in this solve's
 *    own parallel region if N is big enough. Called when M < 2·nb (the
 *    blocked path doesn't kick in). ─────────────────────────────── */
#ifdef _OPENMP
#define YTRSM_OMP_WRAP_LLN_LUN(name, core)                                  \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit)        \
    {                                                                       \
        if (N >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = (int)((long long)N * tid / nt);                   \
                int je  = (int)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }            \
    }
#define YTRSM_OMP_WRAP_TC(name, core, cflag)                                \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit)        \
    {                                                                       \
        if (N >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = (int)((long long)N * tid / nt);                   \
                int je  = (int)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#else
#define YTRSM_OMP_WRAP_LLN_LUN(name, core)                                  \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#define YTRSM_OMP_WRAP_TC(name, core, cflag)                                \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#endif

YTRSM_OMP_WRAP_LLN_LUN(ytrsm_lln, ytrsm_lln_core)
YTRSM_OMP_WRAP_LLN_LUN(ytrsm_lun, ytrsm_lun_core)
YTRSM_OMP_WRAP_TC(ytrsm_llt, ytrsm_llTC_core, 0)
YTRSM_OMP_WRAP_TC(ytrsm_lut, ytrsm_luTC_core, 0)
YTRSM_OMP_WRAP_TC(ytrsm_llc, ytrsm_llTC_core, 1)
YTRSM_OMP_WRAP_TC(ytrsm_luc, ytrsm_luTC_core, 1)

/* ── Blocked SIDE='L': one outer `omp parallel` partitions B's columns
 *    across threads; each thread runs the serial blocked worker on its
 *    own column chunk. ─────────────────────────────────────────── */
static void blocked_dispatch(enum ytrsm_variant V, int M, int N, T alpha,
                             const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = ytrsm_nb();
#ifdef _OPENMP
    if (N >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int js  = (int)((long long)N * tid / nt);
            int je  = (int)((long long)N * (tid + 1) / nt);
            ytrsm_blocked_chunk(V, js, je, M, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    ytrsm_blocked_chunk(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
}

/* ── Blocked SIDE='R': one outer `omp parallel` partitions B's rows
 *    across threads; each thread runs the serial blocked R worker on its
 *    own row band. Mirrors blocked_dispatch (SIDE='L'). ──────────── */
static void blocked_dispatch_R(int upper, int trans, int conj,
                               int M, int N, T alpha,
                               const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = ytrsm_nb();
#ifdef _OPENMP
    if (M >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int is  = (int)((long long)M * tid / nt);
            int ie  = (int)((long long)M * (tid + 1) / nt);
            ytrsm_R_blocked_chunk(upper, trans, conj, is, ie, N, nb, alpha,
                                  a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    ytrsm_R_blocked_chunk(upper, trans, conj, 0, M, N, nb, alpha,
                          a, lda, b, ldb, nounit);
}

/* ── SIDE='R' wrappers: one parallel region partitions the M (row) axis.
 *    Gates on M (the partition axis) >= YTRSM_OMP_N_MIN. The entry guard
 *    already excludes nested calls, so no omp_in_parallel() check here. */
#ifdef _OPENMP
#define YTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int is  = (int)((long long)M * tid / nt);                   \
                int ie  = (int)((long long)M * (tid + 1) / nt);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }            \
    }
#define YTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
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
#define YTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                       \
    }
#define YTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, M, N, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#endif

YTRSM_OMP_WRAP_R   (ytrsm_rln, ytrsm_rln_core)
YTRSM_OMP_WRAP_R   (ytrsm_run, ytrsm_run_core)
YTRSM_OMP_WRAP_R_TC(ytrsm_rlt, ytrsm_rlTC_core, 0)
YTRSM_OMP_WRAP_R_TC(ytrsm_rut, ytrsm_ruTC_core, 0)
YTRSM_OMP_WRAP_R_TC(ytrsm_rlc, ytrsm_rlTC_core, 1)
YTRSM_OMP_WRAP_R_TC(ytrsm_ruc, ytrsm_ruTC_core, 1)

/* ── Entry point ──────────────────────────────────────────────── */

void ytrsm_(
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
        ytrsm_serial(side, uplo, transa, diag, m_, n_, alpha_, a, lda_,
                     b, ldb_, side_len, uplo_len, transa_len, diag_len);
        return;
    }
#endif
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

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * ytrsm_nb());
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch(YLLN, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch(YLUN, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_lun(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else if (TR == 'T') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch(YLLT, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch(YLUT, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_lut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else { /* 'C' */
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch(YLLC, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_llc(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch(YLUC, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_luc(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    } else {
        const int use_blocked = (N >= 2 * ytrsm_nb());
        if (use_blocked) {
            blocked_dispatch_R(UPLO == 'U', TR != 'N', TR == 'C',
                               M, N, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'N') {
            if (UPLO == 'L') ytrsm_rln(M, N, alpha, a, lda, b, ldb, nounit);
            else             ytrsm_run(M, N, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') ytrsm_rlt(M, N, alpha, a, lda, b, ldb, nounit);
            else             ytrsm_rut(M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') ytrsm_rlc(M, N, alpha, a, lda, b, ldb, nounit);
            else             ytrsm_ruc(M, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}

#undef B_
