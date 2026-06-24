/*
 * yhemm_ — kind10 complex (_Complex long double) Hermitian matrix-multiply,
 * the public Fortran entry and threading-orchestration half of the yhemm
 * overlay (see yhemm_kernel.h; all the math lives in yhemm_serial.c).
 *
 * Parallel shape: one `omp parallel for schedule(static)` over the outer
 * panel axis — J column panels of C for SIDE='L', I row panels for
 * SIDE='R'. Each thread owns a disjoint slice of C, so the inner block
 * loops and the ygemm trailing updates run single-thread inside the worker.
 *
 * Nesting guard: when yhemm_ is itself called from inside another routine's
 * parallel region, it delegates to yhemm_serial and opens no region of its
 * own — opening a nested team here trips the libgomp barrier wedge (memory
 * project-etrsm-omp4-wedge).
 */

#include <stddef.h>
#include "../common/blas_char.h"
#include <ctype.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

#include "yhemm_kernel.h"
#include "../common/epblas_facade.h"

typedef yhemm_TC TC;

#define YHEMM_OMP_MIN 32

static const TC ZERO = 0.0L + 0.0Li;
static const TC ONE  = 1.0L + 0.0Li;

static void yhemm_core(
    char side, char uplo,
    ptrdiff_t m, ptrdiff_t n,
    const TC *alpha_,
    const TC *restrict a, ptrdiff_t lda,
    const TC *restrict b, ptrdiff_t ldb,
    const TC *beta_,
    TC *restrict c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own (the libgomp wedge guard). */
    if (omp_in_parallel()) {
        yhemm_serial(side, uplo, m, n, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const TC alpha = *alpha_, beta = *beta_;
    const char SIDE = blas_up(side);
    const char UPLO = blas_up(uplo);

    if (m == 0 || n == 0) return;

    if (alpha == ZERO) {
        if (beta == ONE) return;
#ifdef _OPENMP
        const ptrdiff_t axis = (SIDE == 'L') ? n : m;
        const bool use_omp = (axis >= YHEMM_OMP_MIN && blas_omp_max_threads() > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < n; ++j)
            yhemm_beta_only(j, j + 1, m, beta, c, ldc);
        return;
    }

    ptrdiff_t nthreads = 1;
#ifdef _OPENMP
    nthreads = blas_omp_max_threads();
#endif
    const ptrdiff_t nb = yhemm_nb();

    /* SIDE='L' unblocked per-column Hermitian sweep (yhemm_L_singleblock = the
     * faithful Netlib zhemm port) for (a) small m<=nb, OR (b) UPLO='L' at any m.
     * For LL the unblocked sweep beats the blocked panel path: gfortran's
     * unblocked zhemm runs ~43M vs our blocked ~47M at N=128..256. Columns of C
     * are disjoint, so threading over j stays exact. UPLO='U' keeps the blocked
     * path below (par already beats gfortran there) — see yhemm_serial.c. */
    if (SIDE == 'L' && (m <= nb || UPLO == 'L')) {
#ifdef _OPENMP
        const bool use_omp = (n >= YHEMM_OMP_MIN && nthreads > 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t j = 0; j < n; ++j)
            yhemm_L_singleblock(j, j + 1, m, alpha, beta, a, lda, b, ldb, c, ldc, UPLO);
        return;
    }

    if (SIDE == 'L') {
        ptrdiff_t pw = nb;
#ifdef _OPENMP
        const bool use_omp = (n >= YHEMM_OMP_MIN && nthreads > 1);
        /* Thin the column panels so the team has ~1 panel/thread at small N
         * (N=64, nb=32 -> 2 panels -> 2 idle threads); inner nb is preserved
         * for the trailing-GEMM blocking. Rectangular work -> ppt=1, static. */
        if (use_omp) pw = blas_omp_panel_width(n, nthreads, nb, 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t jc = 0; jc < n; jc += pw) {
            const ptrdiff_t jb = (n - jc < pw) ? (n - jc) : pw;
            yhemm_L_panel(jc, jb, m, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    } else {
        ptrdiff_t pw = nb;
#ifdef _OPENMP
        const bool use_omp = (m >= YHEMM_OMP_MIN && nthreads > 1);
        if (use_omp) pw = blas_omp_panel_width(m, nthreads, nb, 1);
        #pragma omp parallel for if(use_omp) schedule(static)
#endif
        for (ptrdiff_t ic = 0; ic < m; ic += pw) {
            const ptrdiff_t ib = (m - ic < pw) ? (m - ic) : pw;
            yhemm_R_panel(ic, ib, n, alpha, beta, a, lda, b, ldb, c, ldc, UPLO, nb);
        }
    }
}

EPBLAS_FACADE_SYMM(yhemm, TC)
