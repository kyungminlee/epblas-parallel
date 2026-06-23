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

#define XTRMM_OMP_MIN 32

typedef xtrmm_TC TC;


static const TC ZERO = 0.0Q + 0.0Qi;

#define B_(i, j)  b[(size_t)(j) * ldb + (i)]

/* ── OMP wrappers ────────────────────────────────────────────────── */

#ifdef _OPENMP
#define XTRMM_OMP_WRAP_L(name, core)                                        \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                     \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) { \
        if (n >= XTRMM_OMP_MIN && blas_omp_should_thread()) {              \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t js  = blas_part_bound(n, tid, nth);                   \
                ptrdiff_t je  = blas_part_bound(n, tid + 1, nth);             \
                core(js, je, m, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, n, m, alpha, a, lda, b, ldb, nounit); }            \
    }
#define XTRMM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                     \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) { \
        if (n >= XTRMM_OMP_MIN && blas_omp_should_thread()) {              \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t js  = blas_part_bound(n, tid, nth);                   \
                ptrdiff_t je  = blas_part_bound(n, tid + 1, nth);             \
                core(js, je, m, alpha, a, lda, b, ldb, nounit, cflag);      \
            }                                                               \
        } else { core(0, n, m, alpha, a, lda, b, ldb, nounit, cflag); }     \
    }
#define XTRMM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                     \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) { \
        if (m >= XTRMM_OMP_MIN && blas_omp_should_thread()) {              \
            _Pragma("omp parallel") {                                       \
                ptrdiff_t tid = omp_get_thread_num();                             \
                ptrdiff_t nth  = omp_get_num_threads();                            \
                ptrdiff_t is  = blas_part_bound(m, tid, nth);                   \
                ptrdiff_t ie  = blas_part_bound(m, tid + 1, nth);             \
                core(is, ie, n, alpha, a, lda, b, ldb, nounit);             \
            }                                                               \
        } else { core(0, m, n, alpha, a, lda, b, ldb, nounit); }            \
    }
#define XTRMM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                     \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) { \
        if (m >= XTRMM_OMP_MIN && blas_omp_should_thread()) {              \
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
#define XTRMM_OMP_WRAP_L(name, core)                                        \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                     \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) { \
        core(0, n, m, alpha, a, lda, b, ldb, nounit);                       \
    }
#define XTRMM_OMP_WRAP_L_TC(name, core, cflag)                              \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                     \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) { \
        core(0, n, m, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#define XTRMM_OMP_WRAP_R(name, core)                                        \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                     \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) { \
        core(0, m, n, alpha, a, lda, b, ldb, nounit);                       \
    }
#define XTRMM_OMP_WRAP_R_TC(name, core, cflag)                              \
    static void name(ptrdiff_t m, ptrdiff_t n, TC alpha,                     \
                     const TC *a, ptrdiff_t lda, TC *b, ptrdiff_t ldb, bool nounit) { \
        core(0, m, n, alpha, a, lda, b, ldb, nounit, cflag);                \
    }
#endif

XTRMM_OMP_WRAP_L   (xtrmm_lln, xtrmm_lln_core)
XTRMM_OMP_WRAP_L   (xtrmm_lun, xtrmm_lun_core)
XTRMM_OMP_WRAP_L_TC(xtrmm_llt, xtrmm_lltc_core, 0)
XTRMM_OMP_WRAP_L_TC(xtrmm_lut, xtrmm_lutc_core, 0)
XTRMM_OMP_WRAP_L_TC(xtrmm_llc, xtrmm_lltc_core, 1)
XTRMM_OMP_WRAP_L_TC(xtrmm_luc, xtrmm_lutc_core, 1)
XTRMM_OMP_WRAP_R   (xtrmm_rln, xtrmm_rln_core)
XTRMM_OMP_WRAP_R   (xtrmm_run, xtrmm_run_core)
XTRMM_OMP_WRAP_R_TC(xtrmm_rlt, xtrmm_rltc_core, 0)
XTRMM_OMP_WRAP_R_TC(xtrmm_rut, xtrmm_rutc_core, 0)
XTRMM_OMP_WRAP_R_TC(xtrmm_rlc, xtrmm_rltc_core, 1)
XTRMM_OMP_WRAP_R_TC(xtrmm_ruc, xtrmm_rutc_core, 1)

static void xtrmm_core(
    char side, char uplo, char transa, char diag,
    ptrdiff_t m, ptrdiff_t n,
    const TC *alpha_,
    const TC *a, ptrdiff_t lda,
    TC *b, ptrdiff_t ldb)
{
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
        if (TRANS == 'N') {
            if (UPLO == 'L') xtrmm_lln(m, n, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_lun(m, n, alpha, a, lda, b, ldb, nounit);
        } else if (TRANS == 'T') {
            if (UPLO == 'L') xtrmm_llt(m, n, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_lut(m, n, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') xtrmm_llc(m, n, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_luc(m, n, alpha, a, lda, b, ldb, nounit);
        }
    } else {
        if (TRANS == 'N') {
            if (UPLO == 'L') xtrmm_rln(m, n, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_run(m, n, alpha, a, lda, b, ldb, nounit);
        } else if (TRANS == 'T') {
            if (UPLO == 'L') xtrmm_rlt(m, n, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_rut(m, n, alpha, a, lda, b, ldb, nounit);
        } else {
            if (UPLO == 'L') xtrmm_rlc(m, n, alpha, a, lda, b, ldb, nounit);
            else             xtrmm_ruc(m, n, alpha, a, lda, b, ldb, nounit);
        }
    }
}

EPBLAS_FACADE_TRMM(xtrmm, TC)

#undef B_
