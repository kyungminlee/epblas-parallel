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
 * barrier wedge.
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ytrsm_kernel.h"
#include "../common/epblas_facade.h"

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
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)        \
    {                                                                       \
        if (N >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nt  = omp_get_num_threads();                            \
                ptrdiff_t js  = (ptrdiff_t)((long long)N * tid / nt);                   \
                ptrdiff_t je  = (ptrdiff_t)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }            \
    }
#define YTRSM_OMP_WRAP_TC(name, core, cflag)                                \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)        \
    {                                                                       \
        if (N >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nt  = omp_get_num_threads();                            \
                ptrdiff_t js  = (ptrdiff_t)((long long)N * tid / nt);                   \
                ptrdiff_t je  = (ptrdiff_t)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#else
#define YTRSM_OMP_WRAP_LLN_LUN(name, core)                                  \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {     \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#define YTRSM_OMP_WRAP_TC(name, core, cflag)                                \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {     \
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
static void blocked_dispatch(enum ytrsm_variant V, ptrdiff_t M, ptrdiff_t N, T alpha,
                             const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    const ptrdiff_t nb = ytrsm_nb();
#ifdef _OPENMP
    if (N >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            ptrdiff_t tid = omp_get_thread_num();
            ptrdiff_t nt  = omp_get_num_threads();
            ptrdiff_t js  = (ptrdiff_t)((long long)N * tid / nt);
            ptrdiff_t je  = (ptrdiff_t)((long long)N * (tid + 1) / nt);
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
static void blocked_dispatch_R(ptrdiff_t upper, ptrdiff_t trans, ptrdiff_t conj,
                               ptrdiff_t M, ptrdiff_t N, T alpha,
                               const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    const ptrdiff_t nb = ytrsm_nb();
#ifdef _OPENMP
    if (M >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            ptrdiff_t tid = omp_get_thread_num();
            ptrdiff_t nt  = omp_get_num_threads();
            ptrdiff_t is  = (ptrdiff_t)((long long)M * tid / nt);
            ptrdiff_t ie  = (ptrdiff_t)((long long)M * (tid + 1) / nt);
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
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        if (M >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nt  = omp_get_num_threads();                            \
                ptrdiff_t is  = (ptrdiff_t)((long long)M * tid / nt);                   \
                ptrdiff_t ie  = (ptrdiff_t)((long long)M * (tid + 1) / nt);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }            \
    }
#define YTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        if (M >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nt  = omp_get_num_threads();                            \
                ptrdiff_t is  = (ptrdiff_t)((long long)M * tid / nt);                   \
                ptrdiff_t ie  = (ptrdiff_t)((long long)M * (tid + 1) / nt);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#else
#define YTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                       \
    }
#define YTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
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

static void ytrsm_core(
    char side, char uplo, char transa, char diag,
    ptrdiff_t M, ptrdiff_t N,
    const T *alpha_,
    const T *a, ptrdiff_t lda,
    T *b, ptrdiff_t ldb)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        ytrsm_serial(side, uplo, transa, diag, M, N, alpha_, a, lda, b, ldb);
        return;
    }
#endif
    const T alpha = *alpha_;
    const char SIDE = up(&side);
    const char UPLO = up(&uplo);
    const char TR = up(&transa);
    const ptrdiff_t nounit = (up(&diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == ZERO) {
        for (ptrdiff_t j = 0; j < N; ++j)
            for (ptrdiff_t i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }

    if (SIDE == 'L') {
        const ptrdiff_t use_blocked = (M >= 2 * ytrsm_nb());
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
        const ptrdiff_t use_blocked = (N >= 2 * ytrsm_nb());
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

EPBLAS_FACADE_TRMM(ytrsm, T)

#undef B_
