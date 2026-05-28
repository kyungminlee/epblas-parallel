/*
 * qtrmm — kind16 (REAL(KIND=16) / `__float128`) triangular multiply.
 *
 * Computes one of:
 *   B := alpha · op(A) · B          (SIDE='L')
 *   B := alpha · B · op(A)          (SIDE='R')
 *
 * where op(A) ∈ {A, Aᵀ, Aᴴ}; for real types Aᴴ ≡ Aᵀ. A is M×M (or
 * N×N) triangular (upper or lower; optionally unit-diagonal). B is
 * overwritten in place.
 *
 * Implementation: unblocked Netlib reference algorithm with one
 * OpenMP parallel region per call. SIDE='L' partitions columns of B
 * across threads; SIDE='R' partitions rows. Same pattern as qtrsm —
 * see feedback note in MEMORY for the rationale (libquadmath
 * dominates, blocking adds overhead without speedup).
 */

#include <stddef.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QTRMM_OMP_MIN 32

typedef __float128 T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── SIDE = 'L' column-range cores ────────────────────────────────
 * B := alpha · op(A) · B, A is M×M, B is M×N.
 * Each thread owns a column slice [j_start, j_end) of B. */

/* B := alpha · L · B */
static inline void trmm_lln_core(int j_start, int j_end, int M, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = M - 1; k >= 0; --k) {
            if (B_(k, j) != 0.0Q) {
                T temp = alpha * B_(k, j);
                for (int i = M - 1; i > k; --i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

/* B := alpha · U · B */
static inline void trmm_lun_core(int j_start, int j_end, int M, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int k = 0; k < M; ++k) {
            if (B_(k, j) != 0.0Q) {
                T temp = alpha * B_(k, j);
                for (int i = 0; i < k; ++i)
                    B_(i, j) += temp * A_(i, k);
                if (nounit) temp *= A_(k, k);
                B_(k, j) = temp;
            }
        }
    }
}

/* B := alpha · Lᵀ · B */
static inline void trmm_llt_core(int j_start, int j_end, int M, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = 0; i < M; ++i) {
            T t = B_(i, j);
            if (nounit) t *= A_(i, i);
            for (int k = i + 1; k < M; ++k) t += A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* B := alpha · Uᵀ · B */
static inline void trmm_lut_core(int j_start, int j_end, int M, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = j_start; j < j_end; ++j) {
        for (int i = M - 1; i >= 0; --i) {
            T t = B_(i, j);
            if (nounit) t *= A_(i, i);
            for (int k = 0; k < i; ++k) t += A_(k, i) * B_(k, j);
            B_(i, j) = alpha * t;
        }
    }
}

/* ── SIDE = 'R' row-range cores ────────────────────────────────────
 * B := alpha · B · op(A), A is N×N, B is M×N.
 * Each thread owns a row slice [i_start, i_end) of B. */

/* B := alpha · B · L */
static inline void trmm_rln_core(int i_start, int i_end, int N, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = 0; j < N; ++j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != 1.0Q)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = j + 1; k < N; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

/* B := alpha · B · U */
static inline void trmm_run_core(int i_start, int i_end, int N, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int j = N - 1; j >= 0; --j) {
        T t = alpha;
        if (nounit) t *= A_(j, j);
        if (t != 1.0Q)
            for (int i = i_start; i < i_end; ++i) B_(i, j) *= t;
        for (int k = 0; k < j; ++k) {
            if (A_(k, j) != 0.0Q) {
                const T akj = alpha * A_(k, j);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += akj * B_(i, k);
            }
        }
    }
}

/* B := alpha · B · Lᵀ */
static inline void trmm_rlt_core(int i_start, int i_end, int N, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = N - 1; k >= 0; --k) {
        for (int j = k + 1; j < N; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = alpha * A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += ajk * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_(k, k);
        if (t != 1.0Q)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

/* B := alpha · B · Uᵀ */
static inline void trmm_rut_core(int i_start, int i_end, int N, T alpha,
                                 const T *a, int lda, T *b, int ldb, int nounit)
{
    for (int k = 0; k < N; ++k) {
        for (int j = 0; j < k; ++j) {
            if (A_(j, k) != 0.0Q) {
                const T ajk = alpha * A_(j, k);
                for (int i = i_start; i < i_end; ++i)
                    B_(i, j) += ajk * B_(i, k);
            }
        }
        T t = alpha;
        if (nounit) t *= A_(k, k);
        if (t != 1.0Q)
            for (int i = i_start; i < i_end; ++i) B_(i, k) *= t;
    }
}

/* ── OMP wrappers ────────────────────────────────────────────────── */

#ifdef _OPENMP
#define QTRMM_OMP_WRAP_L(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        if (N >= QTRMM_OMP_MIN && blas_omp_max_threads() > 1) {             \
            _Pragma("omp parallel") {                                      \
                int tid = omp_get_thread_num();                            \
                int nt  = omp_get_num_threads();                           \
                int js  = (int)((long long)N * tid / nt);                  \
                int je  = (int)((long long)N * (tid + 1) / nt);            \
                core(js, je, M, alpha, a, lda, b, ldb, nounit);            \
            }                                                              \
        } else { core(0, N, M, alpha, a, lda, b, ldb, nounit); }           \
    }
#define QTRMM_OMP_WRAP_R(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        if (M >= QTRMM_OMP_MIN && blas_omp_max_threads() > 1) {             \
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
#define QTRMM_OMP_WRAP_L(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        core(0, N, M, alpha, a, lda, b, ldb, nounit);                      \
    }
#define QTRMM_OMP_WRAP_R(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        core(0, M, N, alpha, a, lda, b, ldb, nounit);                      \
    }
#endif

QTRMM_OMP_WRAP_L(trmm_lln, trmm_lln_core)
QTRMM_OMP_WRAP_L(trmm_lun, trmm_lun_core)
QTRMM_OMP_WRAP_L(trmm_llt, trmm_llt_core)
QTRMM_OMP_WRAP_L(trmm_lut, trmm_lut_core)
QTRMM_OMP_WRAP_R(trmm_rln, trmm_rln_core)
QTRMM_OMP_WRAP_R(trmm_run, trmm_run_core)
QTRMM_OMP_WRAP_R(trmm_rlt, trmm_rlt_core)
QTRMM_OMP_WRAP_R(trmm_rut, trmm_rut_core)

void qtrmm_(
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
    if (TR == 'C') TR = 'T';
    const int nounit = (up(diag) != 'U');

    if (M == 0 || N == 0) return;

    if (alpha == 0.0Q) {
        for (int j = 0; j < N; ++j)
            for (int i = 0; i < M; ++i) B_(i, j) = 0.0Q;
        return;
    }

    if (SIDE == 'L') {
        if (TR == 'N') {
            if (UPLO == 'L') trmm_lln(M, N, alpha, a, lda, b, ldb, nounit);
            else             trmm_lun(M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') trmm_llt(M, N, alpha, a, lda, b, ldb, nounit);
            else             trmm_lut(M, N, alpha, a, lda, b, ldb, nounit);
        }
    } else {
        if (TR == 'N') {
            if (UPLO == 'L') trmm_rln(M, N, alpha, a, lda, b, ldb, nounit);
            else             trmm_run(M, N, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') trmm_rlt(M, N, alpha, a, lda, b, ldb, nounit);
            else             trmm_rut(M, N, alpha, a, lda, b, ldb, nounit);
        }
    }
}

#undef A_
#undef B_
