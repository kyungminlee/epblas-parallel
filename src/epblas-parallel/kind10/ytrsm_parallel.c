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
#include <stdbool.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ytrsm_kernel.h"
#include "../common/epblas_facade.h"

typedef ytrsm_TC TC;

/* Threshold below which OMP parallel-for on the partition axis isn't
 * worth the parallel-region setup. */
#define YTRSM_OMP_N_MIN 32


static const TC ZERO = 0.0L + 0.0Li;

#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE='L' standalone unblocked wrappers: wrap a core in this solve's
 *    own parallel region if N is big enough. Called when M < 2·nb (the
 *    blocked path doesn't kick in). ─────────────────────────────── */
#ifdef _OPENMP
#define YTRSM_OMP_WRAP_LLN_LUN(name, core)                                  \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                                 \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)        \
    {                                                                       \
        if (n >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t js  = blas_part_bound(n, tid, nth);                   \
                ptrdiff_t je  = blas_part_bound(n, tid + 1, nth);             \
                core(js, je, m, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, n, m, alpha, a, lda, b, ldb, nounit); }            \
    }
#define YTRSM_OMP_WRAP_TC(name, core, cflag)                                \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                                 \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)        \
    {                                                                       \
        if (n >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t js  = blas_part_bound(n, tid, nth);                   \
                ptrdiff_t je  = blas_part_bound(n, tid + 1, nth);             \
                core(js, je, m, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, n, m, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#else
#define YTRSM_OMP_WRAP_LLN_LUN(name, core)                                  \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                                 \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) {     \
        core(0, n, m, alpha, a, lda, b, ldb, nounit);                       \
    }
#define YTRSM_OMP_WRAP_TC(name, core, cflag)                                \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                                 \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) {     \
        core(0, n, m, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#endif

YTRSM_OMP_WRAP_LLN_LUN(ytrsm_lln, ytrsm_lln_core)
YTRSM_OMP_WRAP_LLN_LUN(ytrsm_lun, ytrsm_lun_core)
YTRSM_OMP_WRAP_TC(ytrsm_llt, ytrsm_lltc_core, 0)
YTRSM_OMP_WRAP_TC(ytrsm_lut, ytrsm_lutc_core, 0)
YTRSM_OMP_WRAP_TC(ytrsm_llc, ytrsm_lltc_core, 1)
YTRSM_OMP_WRAP_TC(ytrsm_luc, ytrsm_lutc_core, 1)

/* ── Blocked SIDE='L': one outer `omp parallel` partitions B's columns
 *    across threads; each thread runs the serial blocked worker on its
 *    own column chunk. ─────────────────────────────────────────── */
static void blocked_dispatch(enum ytrsm_variant V, ptrdiff_t m, ptrdiff_t n, ptrdiff_t nb, TC alpha,
                             const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)
{
#ifdef _OPENMP
    if (n >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            ptrdiff_t tid = omp_get_thread_num();
            ptrdiff_t nth  = omp_get_num_threads();
            ptrdiff_t js  = blas_part_bound(n, tid, nth);
            ptrdiff_t je  = blas_part_bound(n, tid + 1, nth);
            ytrsm_blocked_chunk(V, js, je, m, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    ytrsm_blocked_chunk(V, 0, n, m, nb, alpha, a, lda, b, ldb, nounit);
}

/* ── Blocked SIDE='R': one outer `omp parallel` partitions B's rows
 *    across threads; each thread runs the serial blocked R worker on its
 *    own row band. Mirrors blocked_dispatch (SIDE='L'). ──────────── */
static void blocked_dispatch_R(bool upper, bool trans, bool conj,
                               ptrdiff_t m, ptrdiff_t n, ptrdiff_t nb, TC alpha,
                               const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit)
{
#ifdef _OPENMP
    if (m >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            ptrdiff_t tid = omp_get_thread_num();
            ptrdiff_t nth  = omp_get_num_threads();
            ptrdiff_t is  = blas_part_bound(m, tid, nth);
            ptrdiff_t ie  = blas_part_bound(m, tid + 1, nth);
            ytrsm_R_blocked_chunk(upper, trans, conj, is, ie, n, nb, alpha,
                                  a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    ytrsm_R_blocked_chunk(upper, trans, conj, 0, m, n, nb, alpha,
                          a, lda, b, ldb, nounit);
}

/* ── SIDE='R' wrappers: one parallel region partitions the M (row) axis.
 *    Gates on M (the partition axis) >= YTRSM_OMP_N_MIN. The entry guard
 *    already excludes nested calls, so no omp_in_parallel() check here. */
#ifdef _OPENMP
#define YTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                                 \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) {      \
        if (m >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t is  = blas_part_bound(m, tid, nth);                   \
                ptrdiff_t ie  = blas_part_bound(m, tid + 1, nth);             \
                core(is, ie, n, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, m, n, alpha, a, lda, b, ldb, nounit); }            \
    }
#define YTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                                 \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) {      \
        if (m >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t is  = blas_part_bound(m, tid, nth);                   \
                ptrdiff_t ie  = blas_part_bound(m, tid + 1, nth);             \
                core(is, ie, n, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, m, n, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#else
#define YTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                                 \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) {      \
        core(0, m, n, alpha, a, lda, b, ldb, nounit);                       \
    }
#define YTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                                 \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) {      \
        core(0, m, n, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#endif

YTRSM_OMP_WRAP_R   (ytrsm_rln, ytrsm_rln_core)
YTRSM_OMP_WRAP_R   (ytrsm_run, ytrsm_run_core)
YTRSM_OMP_WRAP_R_TC(ytrsm_rlt, ytrsm_rltc_core, 0)
YTRSM_OMP_WRAP_R_TC(ytrsm_rut, ytrsm_rutc_core, 0)
YTRSM_OMP_WRAP_R_TC(ytrsm_rlc, ytrsm_rltc_core, 1)
YTRSM_OMP_WRAP_R_TC(ytrsm_ruc, ytrsm_rutc_core, 1)

/* ── Entry point ──────────────────────────────────────────────── */

static void ytrsm_core(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const TC *alpha_,
    const TC *a, ptrdiff_t lda,
    TC *b, ptrdiff_t ldb)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        ytrsm_serial(side, uplo, transa, diag, m, n, alpha_, a, lda, b, ldb);
        return;
    }
#endif
    const TC alpha = *alpha_;
    const char SIDE = blas_up(side);
    const char UPLO = blas_up(uplo);
    const char TRANS = blas_up(transa);
    const bool nounit = (blas_up(diag) != 'U');

    if (m == 0 || n == 0) return;

    if (alpha == ZERO) {
        for (ptrdiff_t j = 0; j < n; ++j)
            for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) = ZERO;
        return;
    }

    if (SIDE == 'L') {
        const ptrdiff_t nb = ytrsm_block_size(m, TRANS == 'T');
        const ptrdiff_t use_blocked = (nb > 0);
        if (TRANS == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch(YLLN, m, n, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_lln(m, n, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch(YLUN, m, n, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_lun(m, n, alpha, a, lda, b, ldb, nounit);
            }
        } else if (TRANS == 'T') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch(YLLT, m, n, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_llt(m, n, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch(YLUT, m, n, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_lut(m, n, alpha, a, lda, b, ldb, nounit);
            }
        } else { /* 'C' */
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch(YLLC, m, n, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_llc(m, n, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch(YLUC, m, n, nb, alpha, a, lda, b, ldb, nounit);
                else             ytrsm_luc(m, n, alpha, a, lda, b, ldb, nounit);
            }
        }
    } else {
        const ptrdiff_t nb = ytrsm_block_size(n, false);
        const ptrdiff_t use_blocked = (nb > 0);
        if (use_blocked) {
            blocked_dispatch_R(UPLO == 'U', TRANS != 'N', TRANS == 'C',
                               m, n, nb, alpha, a, lda, b, ldb, nounit);
        } else if (TRANS == 'N') {
            if (UPLO == 'L') ytrsm_rln(m, n, alpha, a, lda, b, ldb, nounit);
            else             ytrsm_run(m, n, alpha, a, lda, b, ldb, nounit);
        } else if (TRANS == 'T') {
            if (UPLO == 'L') ytrsm_rlt(m, n, alpha, a, lda, b, ldb, nounit);
            else             ytrsm_rut(m, n, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') ytrsm_rlc(m, n, alpha, a, lda, b, ldb, nounit);
            else             ytrsm_ruc(m, n, alpha, a, lda, b, ldb, nounit);
        }
    }
}

EPBLAS_FACADE_TRMM(ytrsm, TC)

#undef B_
