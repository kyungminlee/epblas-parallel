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
 * character args are bare char* — no hidden trailing lengths (see
 * common/epblas_facade.h; never re-add them); REAL(KIND=16) ↔ __float128.
 */

#include "qtrmm_kernel.h"
#include "../common/blas_char.h"
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdbool.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#define QTRMM_OMP_MIN 32

typedef qtrmm_TR TR;


#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── OMP wrappers ────────────────────────────────────────────────── */

#ifdef _OPENMP
#define QTRMM_OMP_WRAP_L(name, core)                                       \
    static void name(ptrdiff_t m, ptrdiff_t n, TR alpha,                                \
                     const TR *a, ptrdiff_t lda, TR *b, ptrdiff_t ldb, bool nounit) {     \
        if (n >= QTRMM_OMP_MIN && blas_omp_should_thread()) {             \
            _Pragma("omp parallel") {                                      \
                ptrdiff_t tid = omp_get_thread_num();                            \
                ptrdiff_t nth  = omp_get_num_threads();                           \
                ptrdiff_t js  = blas_part_bound(n, tid, nth);                  \
                ptrdiff_t je  = blas_part_bound(n, tid + 1, nth);            \
                core(js, je, m, alpha, a, lda, b, ldb, nounit);            \
            }                                                              \
        } else { core(0, n, m, alpha, a, lda, b, ldb, nounit); }           \
    }
#define QTRMM_OMP_WRAP_R(name, core)                                       \
    static void name(ptrdiff_t m, ptrdiff_t n, TR alpha,                                \
                     const TR *a, ptrdiff_t lda, TR *b, ptrdiff_t ldb, bool nounit) {     \
        if (m >= QTRMM_OMP_MIN && blas_omp_should_thread()) {             \
            _Pragma("omp parallel") {                                      \
                ptrdiff_t tid = omp_get_thread_num();                            \
                ptrdiff_t nth  = omp_get_num_threads();                           \
                ptrdiff_t is  = blas_part_bound(m, tid, nth);                  \
                ptrdiff_t ie  = blas_part_bound(m, tid + 1, nth);            \
                core(is, ie, n, alpha, a, lda, b, ldb, nounit);            \
            }                                                              \
        } else { core(0, m, n, alpha, a, lda, b, ldb, nounit); }           \
    }
#else
#define QTRMM_OMP_WRAP_L(name, core)                                       \
    static void name(ptrdiff_t m, ptrdiff_t n, TR alpha,                                \
                     const TR *a, ptrdiff_t lda, TR *b, ptrdiff_t ldb, bool nounit) {     \
        core(0, n, m, alpha, a, lda, b, ldb, nounit);                      \
    }
#define QTRMM_OMP_WRAP_R(name, core)                                       \
    static void name(ptrdiff_t m, ptrdiff_t n, TR alpha,                                \
                     const TR *a, ptrdiff_t lda, TR *b, ptrdiff_t ldb, bool nounit) {     \
        core(0, m, n, alpha, a, lda, b, ldb, nounit);                      \
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

static void qtrmm_core(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const TR *alpha_,
    const TR *a, ptrdiff_t lda,
    TR *b, ptrdiff_t ldb)
{
    const TR alpha = *alpha_;
    const char SIDE   = blas_up(side);
    const char UPLO   = blas_up(uplo);
    char TRANS           = blas_up(transa);
    if (TRANS == 'C') TRANS = 'T';
    const bool nounit = (blas_up(diag) != 'U');

    if (m == 0 || n == 0) return;

    if (alpha == 0.0Q) {
        for (ptrdiff_t j = 0; j < n; ++j)
            for (ptrdiff_t i = 0; i < m; ++i) B_(i, j) = 0.0Q;
        return;
    }

    if (SIDE == 'L') {
        if (TRANS == 'N') {
            if (UPLO == 'L') trmm_lln(m, n, alpha, a, lda, b, ldb, nounit);
            else             trmm_lun(m, n, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') trmm_llt(m, n, alpha, a, lda, b, ldb, nounit);
            else             trmm_lut(m, n, alpha, a, lda, b, ldb, nounit);
        }
    } else {
        if (TRANS == 'N') {
            if (UPLO == 'L') trmm_rln(m, n, alpha, a, lda, b, ldb, nounit);
            else             trmm_run(m, n, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') trmm_rlt(m, n, alpha, a, lda, b, ldb, nounit);
            else             trmm_rut(m, n, alpha, a, lda, b, ldb, nounit);
        }
    }
}

EPBLAS_FACADE_TRMM(qtrmm, TR)

#undef B_
