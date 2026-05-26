/*
 * ytrmm — kind10 complex (COMPLEX(KIND=10) / `_Complex long double`)
 *         triangular multiply.
 *
 * Same blocked scaffold as etrmm with conjugate-transpose 'C' handled
 * as a distinct case from 'T'. Calls ygemm for the trailing-matrix
 * update inside the blocked path (egemm-class arithmetic gain at
 * kind10 — see doc/parallel-blas-design.md).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define YTRMM_OMP_MIN 32

typedef _Complex long double T;

static int env_int(const char *name, int dflt) {
    const char *s = getenv(name);
    if (!s || !*s) return dflt;
    int v = atoi(s);
    return v > 0 ? v : dflt;
}
static int g_nb_trmm = 0;
static int trmm_nb(void) {
    if (g_nb_trmm == 0) g_nb_trmm = env_int("YTRMM_NB", 32);
    return g_nb_trmm;
}

extern void ygemm_(
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

static const T ZERO = 0.0L + 0.0Li;
static const T ONE  = 1.0L + 0.0Li;

static inline T cconj(T a) { return ~a; }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline T A_op(const T *a, int lda, int row, int col, int conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── Scalar column-range cores (SIDE='L') ───────────────────────── */

static inline void ytrmm_lln_core(int j_start, int j_end, int M, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != ZERO) {
                T temp = alpha * B_(k, j);
                for (int i = M - 1; i > k; --i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

static inline void ytrmm_lun_core(int j_start, int j_end, int M, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != ZERO) {
                T temp = alpha * B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

static inline void ytrmm_llTC_core(int j_start, int j_end, int M, T alpha,
                                   const T *a, int lda, T *b, int ldb,
                                   int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = B_(i, j);
            if (nounit) t *= A_op(a, lda, i, i, conj_flag);
            for (int k = i + 1; k < M; ++k)
                t += A_op(a, lda, k, i, conj_flag) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

static inline void ytrmm_luTC_core(int j_start, int j_end, int M, T alpha,
                                   const T *a, int lda, T *b, int ldb,
                                   int nounit, int conj_flag)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t *= A_op(a, lda, i, i, conj_flag);
            for (int k = 0; k < i; ++k)
                t += A_op(a, lda, k, i, conj_flag) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── Scalar row-range cores (SIDE='R') ──────────────────────────── */

static inline void ytrmm_rln_core(int i_start, int i_end, int N, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

static inline void ytrmm_run_core(int i_start, int i_end, int N, T alpha,
                                  const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != ZERO) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

static inline void ytrmm_rlTC_core(int i_start, int i_end, int N, T alpha,
                                   const T *a, int lda, T *b, int ldb,
                                   int nounit, int conj_flag)
{
    for (int k = N - 1; k >= 0; --k) {
        for (int j = k + 1; j < N; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                const T scaled = alpha * ajk;
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += scaled * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_op(a, lda, k, k, conj_flag);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

static inline void ytrmm_ruTC_core(int i_start, int i_end, int N, T alpha,
                                   const T *a, int lda, T *b, int ldb,
                                   int nounit, int conj_flag)
{
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < k; ++j) {
            const T ajk = A_op(a, lda, j, k, conj_flag);
            if (ajk != ZERO) {
                const T scaled = alpha * ajk;
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += scaled * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_op(a, lda, k, k, conj_flag);
        if (t != ONE)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

/* ── Blocked SIDE='L' variants ──────────────────────────────────── */

enum ytrmm_variant_L { YLLN, YLUN, YLLT, YLUT, YLLC, YLUC };

static void blocked_chunk_L(enum ytrmm_variant_L V, int j_start, int j_end,
                            int M, int nb, T alpha,
                            const T *a, int lda, T *b, int ldb, int nounit)
{
    const int my_N = j_end - j_start;
    if (my_N <= 0) return;

    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    T *B_chunk = &B_(0, j_start);

    if (V == YLLN) {
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_lln_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            if (ic > 0) {
                ygemm_(NN, NN, &ib, &my_N, &ic, &alpha,
                       &A_(ic, 0), &lda,
                       B_chunk, &ldb, &ONE,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    } else if (V == YLUN) {
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_lun_core(j_start, j_end, ib, alpha,
                           &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int j0 = ic + ib;
                ygemm_(NN, NN, &ib, &my_N, &trailing, &alpha,
                       &A_(ic, j0), &lda,
                       &B_chunk[j0], &ldb, &ONE,
                       &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else if (V == YLLT || V == YLLC) {
        const int conj_flag = (V == YLLC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        for (int ic = 0; ic < M; ic += nb) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_llTC_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            const int trailing = M - (ic + ib);
            if (trailing > 0) {
                const int i0 = ic + ib;
                ygemm_(gemm_trans, NN, &ib, &my_N, &trailing, &alpha,
                       &A_(i0, ic), &lda,
                       &B_chunk[i0], &ldb, &ONE,
                       &B_chunk[ic], &ldb, 1, 1);
            }
        }
    } else { /* YLUT or YLUC */
        const int conj_flag = (V == YLUC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        int ic = ((M - 1) / nb) * nb;
        while (ic >= 0) {
            const int ib = (M - ic < nb) ? (M - ic) : nb;
            ytrmm_luTC_core(j_start, j_end, ib, alpha,
                            &A_(ic, ic), lda, &B_(ic, 0), ldb, nounit, conj_flag);
            if (ic > 0) {
                ygemm_(gemm_trans, NN, &ib, &my_N, &ic, &alpha,
                       &A_(0, ic), &lda,
                       B_chunk, &ldb, &ONE,
                       &B_chunk[ic], &ldb, 1, 1);
            }
            ic -= nb;
        }
    }
}

static void blocked_dispatch_L(enum ytrmm_variant_L V, int M, int N, T alpha,
                               const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = trmm_nb();
#ifdef _OPENMP
    if (N >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int js  = ((long long)N * tid) / nt;
            int je  = ((long long)N * (tid + 1)) / nt;
            blocked_chunk_L(V, js, je, M, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    blocked_chunk_L(V, 0, N, M, nb, alpha, a, lda, b, ldb, nounit);
}

/* ── Blocked SIDE='R' variants ──────────────────────────────────── */

enum ytrmm_variant_R { YRLN, YRUN, YRLT, YRUT, YRLC, YRUC };

static void blocked_chunk_R(enum ytrmm_variant_R V, int i_start, int i_end,
                            int N, int nb, T alpha,
                            const T *a, int lda, T *b, int ldb, int nounit)
{
    const char NN[1] = {'N'};
    const char TN[1] = {'T'};
    const char CN[1] = {'C'};
    const int my_M = i_end - i_start;
    if (my_M <= 0) return;
    T *B_chunk = &B_(i_start, 0);

    if (V == YRLN) {
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_rln_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                ygemm_(NN, NN, &my_M, &jb, &trailing, &alpha,
                       &B_chunk[(size_t)k0 * ldb], &ldb,
                       &A_(k0, jc), &lda, &ONE,
                       &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
        }
    } else if (V == YRUN) {
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_run_core(i_start, i_end, jb, alpha,
                           &A_(jc, jc), lda, &B_(0, jc), ldb, nounit);
            if (jc > 0) {
                ygemm_(NN, NN, &my_M, &jb, &jc, &alpha,
                       B_chunk, &ldb,
                       &A_(0, jc), &lda, &ONE,
                       &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else if (V == YRLT || V == YRLC) {
        const int conj_flag = (V == YRLC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        int jc = ((N - 1) / nb) * nb;
        while (jc >= 0) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_rlTC_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
            if (jc > 0) {
                ygemm_(NN, gemm_trans, &my_M, &jb, &jc, &alpha,
                       B_chunk, &ldb,
                       &A_(jc, 0), &lda, &ONE,
                       &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
            jc -= nb;
        }
    } else { /* YRUT or YRUC */
        const int conj_flag = (V == YRUC) ? 1 : 0;
        const char *gemm_trans = conj_flag ? CN : TN;
        for (int jc = 0; jc < N; jc += nb) {
            const int jb = (N - jc < nb) ? (N - jc) : nb;
            ytrmm_ruTC_core(i_start, i_end, jb, alpha,
                            &A_(jc, jc), lda, &B_(0, jc), ldb, nounit, conj_flag);
            const int trailing = N - (jc + jb);
            if (trailing > 0) {
                const int k0 = jc + jb;
                ygemm_(NN, gemm_trans, &my_M, &jb, &trailing, &alpha,
                       &B_chunk[(size_t)k0 * ldb], &ldb,
                       &A_(jc, k0), &lda, &ONE,
                       &B_chunk[(size_t)jc * ldb], &ldb, 1, 1);
            }
        }
    }
}

static void blocked_dispatch_R(enum ytrmm_variant_R V, int M, int N, T alpha,
                               const T *a, int lda, T *b, int ldb, int nounit)
{
    const int nb = trmm_nb();
#ifdef _OPENMP
    if (M >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {
        #pragma omp parallel
        {
            int tid = omp_get_thread_num();
            int nt  = omp_get_num_threads();
            int is  = ((long long)M * tid) / nt;
            int ie  = ((long long)M * (tid + 1)) / nt;
            blocked_chunk_R(V, is, ie, N, nb, alpha, a, lda, b, ldb, nounit);
        }
        return;
    }
#endif
    blocked_chunk_R(V, 0, M, N, nb, alpha, a, lda, b, ldb, nounit);
}

/* ── Standalone OMP wrappers (unblocked fallback for small M/N). */

#ifdef _OPENMP
#define YTRMM_OMP_WRAP_L(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (N >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = (int)((long long)N * tid / nt);                   \
                int je  = (int)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }            \
    }
#define YTRMM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (N >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = (int)((long long)N * tid / nt);                   \
                int je  = (int)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#define YTRMM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int is  = (int)((long long)M * tid / nt);                   \
                int ie  = (int)((long long)M * (tid + 1) / nt);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }            \
    }
#define YTRMM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= YTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
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
#define YTRMM_OMP_WRAP_L(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#define YTRMM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#define YTRMM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                       \
    }
#define YTRMM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
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

void ytrmm_(
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

    const int nb = trmm_nb();

    if (SIDE == 'L') {
        const int use_blocked = (M >= 2 * nb);
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
        const int use_blocked = (N >= 2 * nb);
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

#undef A_
#undef B_
