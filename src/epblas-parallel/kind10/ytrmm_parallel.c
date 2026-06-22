/*
 * ytrmm_ — kind10 complex (COMPLEX(KIND=10) / _Complex long double)
 * triangular multiply, the public Fortran entry and threading-orchestration
 * half of the ytrmm overlay (see ytrmm_kernel.h for the split rationale;
 * all the math lives in ytrmm_serial.c).
 *
 * Two parallel shapes, both partitioning a single axis across one team:
 *   SIDE='L'  — partition B's columns; each thread runs the serial
 *               blocked (or unblocked) multiply on its own column slice.
 *   SIDE='R'  — partition B's rows; each thread owns a disjoint row slice.
 *
 * One `omp parallel` per call, and ygemm trailing updates run single-thread
 * (ygemm_serial) inside each outer thread.
 *
 * Nesting guard: when ytrmm_ is itself called from inside another routine's
 * parallel region, it delegates to ytrmm_serial and opens no region of its
 * own — opening a nested team here trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "ytrmm_kernel.h"
#include "../common/epblas_facade.h"

typedef ytrmm_T T;

#define YTRMM_OMP_MIN 32

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0L + 0.0Li;

#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── Blocked dispatchers: one outer `omp parallel` partitions B's
 *    columns (SIDE='L') or rows (SIDE='R') across the team. ──────── */
static void blocked_dispatch_L(enum ytrmm_variant_L V, ptrdiff_t M, ptrdiff_t N, T alpha,
                               const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    const ptrdiff_t nb = ytrmm_nb();
#ifdef _OPENMP
    if (N >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            ptrdiff_t tid = omp_get_thread_num();
            ptrdiff_t nt  = omp_get_num_threads();
            ptrdiff_t js  = blas_part_bound(N, tid, nt);
            ptrdiff_t je  = blas_part_bound(N, tid + 1, nt);
            ytrmm_blocked_chunk_L(V, js, je, M, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    ytrmm_blocked_chunk_L(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
}

static void blocked_dispatch_R(enum ytrmm_variant_R V, ptrdiff_t M, ptrdiff_t N, T alpha,
                               const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit)
{
    const ptrdiff_t nb = ytrmm_nb();
#ifdef _OPENMP
    if (M >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            ptrdiff_t tid = omp_get_thread_num();
            ptrdiff_t nt  = omp_get_num_threads();
            ptrdiff_t is  = blas_part_bound(M, tid, nt);
            ptrdiff_t ie  = blas_part_bound(M, tid + 1, nt);
            ytrmm_blocked_chunk_R(V, is, ie, N, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    ytrmm_blocked_chunk_R(V, 0, M, N, nb, alpha, a, lda, b, ldb, nounit);
}

/* ── Standalone OMP wrappers (unblocked fallback for small M/N). ──── */

#ifdef _OPENMP
#define YTRMM_OMP_WRAP_L(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        if (N >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nt  = omp_get_num_threads();                            \
                ptrdiff_t js  = blas_part_bound(N, tid, nt);                   \
                ptrdiff_t je  = blas_part_bound(N, tid + 1, nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }            \
    }
#define YTRMM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        if (N >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nt  = omp_get_num_threads();                            \
                ptrdiff_t js  = blas_part_bound(N, tid, nt);                   \
                ptrdiff_t je  = blas_part_bound(N, tid + 1, nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#define YTRMM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        if (M >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nt  = omp_get_num_threads();                            \
                ptrdiff_t is  = blas_part_bound(M, tid, nt);                   \
                ptrdiff_t ie  = blas_part_bound(M, tid + 1, nt);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }            \
    }
#define YTRMM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        if (M >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nt  = omp_get_num_threads();                            \
                ptrdiff_t is  = blas_part_bound(M, tid, nt);                   \
                ptrdiff_t ie  = blas_part_bound(M, tid + 1, nt);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#else
#define YTRMM_OMP_WRAP_L(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#define YTRMM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#define YTRMM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                       \
    }
#define YTRMM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(ptrdiff_t M, ptrdiff_t N, T alpha,                                 \
                     const T *a, ptrdiff_t lda, T *b, ptrdiff_t ldb, ptrdiff_t nounit) {      \
        core(0, M, N, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#endif

YTRMM_OMP_WRAP_L   (ytrmm_lln, ytrmm_lln_core)
YTRMM_OMP_WRAP_L   (ytrmm_lun, ytrmm_lun_core)
YTRMM_OMP_WRAP_L_TC(ytrmm_llt, ytrmm_llTC_core, 0)
YTRMM_OMP_WRAP_L_TC(ytrmm_lut, ytrmm_luTC_core, 0)
YTRMM_OMP_WRAP_L_TC(ytrmm_llc, ytrmm_llTC_core, 1)
YTRMM_OMP_WRAP_L_TC(ytrmm_luc, ytrmm_luTC_core, 1)
YTRMM_OMP_WRAP_R   (ytrmm_rln, ytrmm_rln_core)
YTRMM_OMP_WRAP_R   (ytrmm_run, ytrmm_run_core)
YTRMM_OMP_WRAP_R_TC(ytrmm_rlt, ytrmm_rlTC_core, 0)
YTRMM_OMP_WRAP_R_TC(ytrmm_rut, ytrmm_ruTC_core, 0)
YTRMM_OMP_WRAP_R_TC(ytrmm_rlc, ytrmm_rlTC_core, 1)
YTRMM_OMP_WRAP_R_TC(ytrmm_ruc, ytrmm_ruTC_core, 1)

static void ytrmm_core(
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
        ytrmm_serial(side, uplo, transa, diag, M, N, alpha_, a, lda, b, ldb);
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

    const ptrdiff_t nb = ytrmm_nb();

    if (SIDE == 'L') {
        const ptrdiff_t use_blocked = (M >= 2 * nb);
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_L(YLLN, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_L(YLUN, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_lun(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else if (TR == 'T') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_L(YLLT, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_L(YLUT, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_lut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else { /* 'C' */
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_L(YLLC, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_llc(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_L(YLUC, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_luc(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    } else {
        const ptrdiff_t use_blocked = (N >= 2 * nb);
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_R(YRLN, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_rln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_R(YRUN, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_run(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else if (TR == 'T') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_R(YRLT, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_rlt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_R(YRUT, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_rut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_R(YRLC, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_rlc(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_R(YRUC, M, N, alpha, a, lda, b, ldb, nounit);
                else             ytrmm_ruc(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    }
}

EPBLAS_FACADE_TRMM(ytrmm, T)

#undef B_
