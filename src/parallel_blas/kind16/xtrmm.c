/*
 * xtrmm — kind16 complex (COMPLEX(KIND=16) / `__complex128`)
 *         triangular multiply.
 *
 * Unblocked Netlib reference algorithm with OpenMP coarse-grain
 * parallelism. SIDE='L' partitions columns of B; SIDE='R' partitions
 * rows. TRANSA='C' (conjugate transpose) is a distinct case from 'T'.
 *
 * See qtrmm.c for the (M×M) layout / signature commentary. The
 * complex flop count is 4× the real one per inner mul-add.
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XTRMM_OMP_MIN 32

typedef __complex128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0Q + 0.0Qi;
static const T ONE  = 1.0Q + 0.0Qi;

static inline T cconj(T a) { return conjq(a); }

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

static inline T A_op(const T *a, int lda, int row, int col, int conj_flag) {
    return conj_flag ? cconj(A_(row, col)) : A_(row, col);
}

/* ── SIDE = 'L' column-range cores ──────────────────────────────── */

static inline void xtrmm_lln_core(int j_start, int j_end, int M, T alpha,
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

static inline void xtrmm_lun_core(int j_start, int j_end, int M, T alpha,
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

static inline void xtrmm_llTC_core(int j_start, int j_end, int M, T alpha,
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

static inline void xtrmm_luTC_core(int j_start, int j_end, int M, T alpha,
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

/* ── SIDE = 'R' row-range cores ─────────────────────────────────── */

static inline void xtrmm_rln_core(int i_start, int i_end, int N, T alpha,
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
                for (int i = i_start; i < i_end; ++i) B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

static inline void xtrmm_run_core(int i_start, int i_end, int N, T alpha,
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
                for (int i = i_start; i < i_end; ++i) B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

static inline void xtrmm_rlTC_core(int i_start, int i_end, int N, T alpha,
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

static inline void xtrmm_ruTC_core(int i_start, int i_end, int N, T alpha,
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

/* ── OMP wrappers ────────────────────────────────────────────────── */

#ifdef _OPENMP
#define XTRMM_OMP_WRAP_L(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (N >= XTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = (int)((long long)N * tid / nt);                   \
                int je  = (int)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }            \
    }
#define XTRMM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (N >= XTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int js  = (int)((long long)N * tid / nt);                   \
                int je  = (int)((long long)N * (tid + 1) / nt);             \
                core(js, je, M, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#define XTRMM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= XTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
            _Pragma("omp parallel") {                                       \
                int tid = omp_get_thread_num();                             \
                int nt  = omp_get_num_threads();                            \
                int is  = (int)((long long)M * tid / nt);                   \
                int ie  = (int)((long long)M * (tid + 1) / nt);             \
                core(is, ie, N, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, M, N, alpha, a, lda, b, ldb, nounit); }            \
    }
#define XTRMM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (M >= XTRMM_OMP_MIN && blas_omp_max_threads() > 1) {              \
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
#define XTRMM_OMP_WRAP_L(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                       \
    }
#define XTRMM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, N, M, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#define XTRMM_OMP_WRAP_R(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                       \
    }
#define XTRMM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        core(0, M, N, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#endif

XTRMM_OMP_WRAP_L   (xtrmm_lln, xtrmm_lln_core)
XTRMM_OMP_WRAP_L   (xtrmm_lun, xtrmm_lun_core)
XTRMM_OMP_WRAP_L_TC(xtrmm_llt, xtrmm_llTC_core, 0)
XTRMM_OMP_WRAP_L_TC(xtrmm_lut, xtrmm_luTC_core, 0)
XTRMM_OMP_WRAP_L_TC(xtrmm_llc, xtrmm_llTC_core, 1)
XTRMM_OMP_WRAP_L_TC(xtrmm_luc, xtrmm_luTC_core, 1)
XTRMM_OMP_WRAP_R   (xtrmm_rln, xtrmm_rln_core)
XTRMM_OMP_WRAP_R   (xtrmm_run, xtrmm_run_core)
XTRMM_OMP_WRAP_R_TC(xtrmm_rlt, xtrmm_rlTC_core, 0)
XTRMM_OMP_WRAP_R_TC(xtrmm_rut, xtrmm_ruTC_core, 0)
XTRMM_OMP_WRAP_R_TC(xtrmm_rlc, xtrmm_rlTC_core, 1)
XTRMM_OMP_WRAP_R_TC(xtrmm_ruc, xtrmm_ruTC_core, 1)

void xtrmm_(
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
        if (TR == 'N') {
            if (UPLO == 'L') xtrmm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_lun(M, N, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') xtrmm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_lut(M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') xtrmm_llc(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_luc(M, N, alpha, a, lda, b, ldb, nounit);
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') xtrmm_rln(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_run(M, N, alpha, a, lda, b, ldb, nounit);
        } else if (TR == 'T') {
            if (UPLO == 'L') xtrmm_rlt(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_rut(M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') xtrmm_rlc(M, N, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_ruc(M, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}

#undef A_
#undef B_
