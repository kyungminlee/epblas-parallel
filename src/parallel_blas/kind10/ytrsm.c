/*
 * ytrsm — kind10 complex (COMPLEX(KIND=10) / `_Complex long double`)
 *         triangular solve.
 *
 * Same scaffolding as xtrsm (kind16 complex) with TRANSA='C'
 * (conjugate transpose) handled as a distinct case from 'T'. No
 * SIMD path — x87 long-double has no SIMD register file on x86_64.
 * Calls ygemm for the SIDE='L' trailing-matrix update.
 */

#include <stddef.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YTRSM_OMP_N_MIN 32

typedef _Complex long double T;

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}

static int g_nb_trsm = 0;
static int trsm_nb(void) {
    if (g_nb_trsm == 0) g_nb_trsm = env_int("YTRSM_NB", 64);
    return g_nb_trsm;
}

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;
static const T NEG_ONE = -1.0L + 0.0Li;

static inline T cconj(T a) { return ~a; }

extern void ygemm_(
    const char *transa, const char *transb,
    const int *m, const int *n, const int *k,
    const T *alpha,
    const T *a, const int *lda,
    const T *b, const int *ldb,
    const T *beta,
    T *c, const int *ldc,
    size_t transa_len, size_t transb_len);

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline T A_op(const T *a, int lda, int row, int col, int conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── Scalar column-range cores ───────────────────────────────── */

static inline void ytrsm_lln_core(int j_start, int j_end, int M, T alpha,
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

static inline void ytrsm_lun_core(int j_start, int j_end, int M, T alpha,
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

/* (L, L, T or C): inner-product form on op(A)ᵀ where op may conj. */
static inline void ytrsm_llTC_core(int j_start, int j_end, int M, T alpha,
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

/* (L, U, T or C). */
static inline void ytrsm_luTC_core(int j_start, int j_end, int M, T alpha,
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

/* ── SIDE = 'R' row-range cores ───────────────────────────────────
 *
 * The j (column) loop walks the diagonal serially (each B[:,j] depends
 * on prior B[:,k]), but within each step every row of B is processed
 * identically. Each thread owns a disjoint row slice [i_start, i_end)
 * of B and reads shared A read-only — race-free, no barriers needed. */

static inline void ytrsm_rln_core(int i_start, int i_end, int N, T alpha,
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

static inline void ytrsm_run_core(int i_start, int i_end, int N, T alpha,
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

static inline void ytrsm_rlTC_core(int i_start, int i_end, int N, T alpha,
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

static inline void ytrsm_ruTC_core(int i_start, int i_end, int N, T alpha,
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

/* Standalone OMP wrappers for SIDE='L' core variants. */
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

/* SIDE='R' OMP wrappers: one parallel region partitions the M (row)
 * axis. Gates on M (the partition axis) >= YTRSM_OMP_N_MIN. */
#ifdef _OPENMP
#define YTRSM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1              \
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
#define YTRSM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1              \
                                && !omp_in_parallel()) {                    \
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

/* ── Blocked SIDE='L' variants. */

static void prescale_chunk(int j_start, int j_end, int M, T alpha, T *b, int ldb)
{
    if (alpha == ONE) return;
    if (alpha == ZERO) {
        for (int j = j_start; j < j_end; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = ZERO;
        return;
    }
    for (int j = j_start; j < j_end; ++j)
        for (int i = 0; i < M; ++i) B_(i, j) *= alpha;
}

enum ytrsm_variant { YLLN, YLUN, YLLT, YLUT, YLLC, YLUC };

static void blocked_chunk(enum ytrsm_variant V, int j_start, int j_end,
                          int M, int nb, T alpha,
                          const T *a, int lda, T *b, int ldb, int nounit)
{
    const int my_N = j_end - j_start;
    if (my_N <= 0) return;
    prescale_chunk(j_start, j_end, M, alpha, b, ldb);

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    T *B_chunk = &B_(0, j_start);

    if (V == YLLN) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                ygemm_(NN, NN, &ib, &my_N, &ic, &NEG_ONE,
                       &A_(ic, 0), &lda,
                       B_chunk, &ldb, &ONE,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            ytrsm_lln_core(j_start, j_end, ib, ONE,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
        }
    } else if (V == YLUN) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int j0 = ic + ib;
                ygemm_(NN, NN, &ib, &my_N, &trailing, &NEG_ONE,
                       &A_(ic, j0), &lda,
                       &B_chunk[j0], &ldb, &ONE,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            ytrsm_lun_core(j_start, j_end, ib, ONE,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            ic -= nb;
        }
    } else if (V == YLLT || V == YLLC) {
        const int conj_flag = (V == YLLC) ? 1 : 0;
        const char *trans_gemm = conj_flag ? CN : TN;
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int i0 = ic + ib;
                ygemm_(trans_gemm, NN, &ib, &my_N, &trailing, &NEG_ONE,
                       &A_(i0, ic), &lda,
                       &B_chunk[i0], &ldb, &ONE,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            ytrsm_llTC_core(j_start, j_end, ib, ONE,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb,
                            nounit, conj_flag);
            ic -= nb;
        }
    } else { /* YLUT or YLUC */
        const int conj_flag = (V == YLUC) ? 1 : 0;
        const char *trans_gemm = conj_flag ? CN : TN;
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            if (ic > 0) {
                ygemm_(trans_gemm, NN, &ib, &my_N, &ic, &NEG_ONE,
                       &A_(0, ic), &lda,
                       B_chunk, &ldb, &ONE,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            ytrsm_luTC_core(j_start, j_end, ib, ONE,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb,
                            nounit, conj_flag);
        }
    }
}

static void blocked_dispatch(enum ytrsm_variant V, int M, int N, T alpha,
                             const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = trsm_nb();
#ifdef _OPENMP
    if (N >= YTRSM_OMP_N_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int js  = (int)((long long)N * tid / nt);
            int je  = (int)((long long)N * (tid + 1) / nt);
            blocked_chunk(V, js, je, M, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    blocked_chunk(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
}

void ytrsm_(
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

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * trsm_nb());
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
        if (TR == 'N') {
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

#undef A_
#undef B_
