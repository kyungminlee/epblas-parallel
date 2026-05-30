/*
 * etrmm_ — kind10 real (long double) triangular matrix-multiply, the public
 * Fortran entry and threading-orchestration half of the etrmm overlay (see
 * etrmm_kernel.h; all the math lives in etrmm_serial.c).
 *
 * Parallel shape: the unblocked cores are partitioned coarsely over the
 * free axis (B columns for SIDE='L', rows for SIDE='R') by a single
 * `omp parallel` region; the blocked path opens one team and gives each
 * thread a column/row slice through the blocked chunk worker.
 *
 * Nesting guard: when etrmm_ is itself called from inside another routine's
 * parallel region, it delegates to etrmm_serial and opens no region of its
 * own (the libgomp barrier wedge guard, project-etrsm-omp4-wedge).
 */

#include "etrmm_kernel.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef etrmm_T T;

#define ETRMM_OMP_MIN 32

/* ── Coarse-axis OMP wrappers for the unblocked cores ─────────── */

#ifdef _OPENMP
#define ETRMM_OMP_WRAP_L(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        if (N >= ETRMM_OMP_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                      \
                int tid = omp_get_thread_num();                            \
                int nt  = omp_get_num_threads();                           \
                int js  = (int)((long long)N * tid / nt);                  \
                int je  = (int)((long long)N * (tid + 1) / nt);            \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);            \
            }                                                              \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }           \
    }
#define ETRMM_OMP_WRAP_R(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        if (M >= ETRMM_OMP_MIN && blas_omp_max_threads() > 1) {            \
            _Pragma("omp parallel") {                                      \
                int tid = omp_get_thread_num();                            \
                int nt  = omp_get_num_threads();                           \
                int is  = (int)((long long)M * tid / nt);                  \
                int ie  = (int)((long long)M * (tid + 1) / nt);            \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);            \
            }                                                              \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }           \
    }
#else
#define ETRMM_OMP_WRAP_L(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                      \
    }
#define ETRMM_OMP_WRAP_R(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                      \
    }
#endif

ETRMM_OMP_WRAP_L(trmm_lln, etrmm_lln_core)
ETRMM_OMP_WRAP_L(trmm_lun, etrmm_lun_core)
ETRMM_OMP_WRAP_L(trmm_llt, etrmm_llt_core)
ETRMM_OMP_WRAP_L(trmm_lut, etrmm_lut_core)
ETRMM_OMP_WRAP_R(trmm_rln, etrmm_rln_core)
ETRMM_OMP_WRAP_R(trmm_run, etrmm_run_core)
ETRMM_OMP_WRAP_R(trmm_rlt, etrmm_rlt_core)
ETRMM_OMP_WRAP_R(trmm_rut, etrmm_rut_core)

/* ── Blocked dispatch: one team, each thread a column/row slice ── */

static void blocked_dispatch_L(enum etrmm_variant_L V, int M, int N, T alpha,
                               const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = etrmm_nb();
#ifdef _OPENMP
    if (N >= ETRMM_OMP_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int js  = ((long long)N * tid) / nt;
            int je  = ((long long)N * (tid + 1)) / nt;
            etrmm_blocked_chunk_L(V, js, je, M, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    etrmm_blocked_chunk_L(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
}

static void blocked_dispatch_R(enum etrmm_variant_R V, int M, int N, T alpha,
                               const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = etrmm_nb();
#ifdef _OPENMP
    if (M >= ETRMM_OMP_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int is  = ((long long)M * tid) / nt;
            int ie  = ((long long)M * (tid + 1)) / nt;
            etrmm_blocked_chunk_R(V, is, ie, N, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    etrmm_blocked_chunk_R(V, 0, M, N, nb, alpha, a, lda, b, ldb, nounit);
}

/* ── Entry point ──────────────────────────────────────────────── */

void etrmm_(
    const char *side, const char *uplo, const char *transa, const char *diag,
    const int *m_, const int *n_,
    const T *alpha_,
    const T *a, const int *lda_,
    T *b, const int *ldb_,
    size_t side_len, size_t uplo_len, size_t transa_len, size_t diag_len)
{
#ifdef _OPENMP
    /* Already inside a team → run serially, no nested region (wedge guard). */
    if (omp_in_parallel()) {
        etrmm_serial(side, uplo, transa, diag, m_, n_, alpha_, a, lda_, b, ldb_,
                     side_len, uplo_len, transa_len, diag_len);
        return;
    }
#endif
    (void)side_len; (void)uplo_len; (void)transa_len; (void)diag_len;
    const int M = *m_, N = *n_;
    const int lda = *lda_, ldb = *ldb_;
    const T alpha = *alpha_;
    const char SIDE = (char)toupper((unsigned char)*side);
    const char UPLO = (char)toupper((unsigned char)*uplo);
    char TR = (char)toupper((unsigned char)*transa);
    if (TR == 'C') TR = 'T';
    const int nounit = ((char)toupper((unsigned char)*diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == 0.0L) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) b[(size_t)j * ldb + i] = 0.0L;
        return;
    }

    const int nb = etrmm_nb();

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * nb);
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_L(ETRMM_LLN, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_L(ETRMM_LUN, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_lun(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_L(ETRMM_LLT, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_L(ETRMM_LUT, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_lut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    } else {
        const int use_blocked = (N >= 2 * nb);
        if (TR == 'N') {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_R(ETRMM_RLN, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_rln(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_R(ETRMM_RUN, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_run(M, N, alpha, a, lda, b, ldb, nounit);
            }
        } else {
            if (UPLO == 'L') {
                if (use_blocked) blocked_dispatch_R(ETRMM_RLT, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_rlt(M, N, alpha, a, lda, b, ldb, nounit);
            } else {
                if (use_blocked) blocked_dispatch_R(ETRMM_RUT, M, N, alpha, a, lda, b, ldb, nounit);
                else             trmm_rut(M, N, alpha, a, lda, b, ldb, nounit);
            }
        }
    }
}
