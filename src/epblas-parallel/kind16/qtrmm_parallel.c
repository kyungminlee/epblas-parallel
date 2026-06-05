/*
 * qtrmm_ — kind16 (REAL(KIND=16) / __float128) triangular multiply, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in the
 * range-parameterized cores in qtrmm_serial.c, shared through
 * qtrmm_kernel.h.
 *
 *   B := alpha · op(A) · B          (SIDE='L')
 *   B := alpha · B · op(A)          (SIDE='R')
 *
 * One OpenMP parallel region per call. SIDE='L' partitions the columns of B
 * across the team; SIDE='R' partitions the rows. Each column / row is
 * independent, so the static [start,end) partition is bitwise-identical to
 * the serial run. Below QTRMM_OMP_MIN, single-threaded, or already inside a
 * parallel region, the matching core runs serially over the full range.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths; REAL(KIND=16)
 * ↔ __float128.
 */

#include "qtrmm_kernel.h"
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QTRMM_OMP_MIN 32

typedef qtrmm_T T;

static inline char up(const char *p) {
    return (char)toupper((unsigned char)*p);
}

#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── OMP wrappers ────────────────────────────────────────────────── */

#ifdef _OPENMP
#define QTRMM_OMP_WRAP_L(name, core)                                       \
    static void name(int M, int N, T alpha,                                \
                     const T *a, int lda, T *b, int ldb, int nounit) {     \
        if (N >= QTRMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {             \
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
        if (M >= QTRMM_OMP_MIN && blas_omp_max_threads() > 1 && !omp_in_parallel()) {             \
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

#undef B_
