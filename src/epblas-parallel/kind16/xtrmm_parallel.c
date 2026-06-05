/*
 * xtrmm_ — kind16 complex (COMPLEX(KIND=16) / __complex128) triangular
 * multiply, public Fortran entry. THREADING ORCHESTRATION ONLY: all the
 * math lives in the range-parameterized cores in xtrmm_serial.c, shared
 * through xtrmm_kernel.h.
 *
 *   B := alpha · op(A) · B          (SIDE='L')
 *   B := alpha · B · op(A)          (SIDE='R')
 *
 * One OpenMP parallel region per call. SIDE='L' partitions the columns of B
 * across the team; SIDE='R' partitions the rows. TRANSA='C' (conjugate
 * transpose) is a distinct case from 'T' (the *_TC cores carry a conj_flag
 * the wrappers bind). Each column / row is independent, so the static
 * [start,end) partition is bitwise-identical to the serial run. Below
 * XTRMM_OMP_MIN, single-threaded, or already inside a parallel region, the
 * matching core runs serially over the full range.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths;
 * COMPLEX(KIND=16) ↔ __complex128.
 */

#include "xtrmm_kernel.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define XTRMM_OMP_MIN 32

typedef xtrmm_T T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

static const T ZERO = 0.0Q + 0.0Qi;

#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── OMP wrappers ────────────────────────────────────────────────── */

#ifdef _OPENMP
#define XTRMM_OMP_WRAP_L(name, core)                                        \
    static void name(int M, int N, T alpha,                                 \
                     const T *a, int lda, T *b, int ldb, int nounit) {      \
        if (N >= XTRMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {              \
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
        if (N >= XTRMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {              \
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
        if (M >= XTRMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {              \
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
        if (M >= XTRMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {              \
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

#undef B_
