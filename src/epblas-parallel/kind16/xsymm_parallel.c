/*
 * xsymm_ — kind16 complex SYMM (COMPLEX(KIND=16) / __complex128), the
 * public Fortran entry and threading-orchestration half of the xsymm
 * overlay (see xsymm_kernel.h; all the math lives in xsymm_serial.c).
 *
 *   C := alpha * A * B + beta * C    (SIDE=L, A symmetric M×M)
 *   C := alpha * B * A + beta * C    (SIDE=R, A symmetric N×N)
 *
 * Parallel shape (the OpenBLAS level3.c SMP shape): one `omp parallel`
 * block-partitions the M axis across the team (rounded to MR). Each thread
 * keeps a private Ap pack buffer; the Bp panel is shared and packed once
 * per (js, ls) under `omp single`, bracketed by an explicit barrier and the
 * single's implicit end barrier. The beta pre-pass runs once up front.
 *
 * Nesting guard: when the core is called from inside another routine's
 * parallel region it delegates to xsymm_serial and opens no region.
 *
 * ABI: the LP64 `xsymm_` and ILP64 `xsymm_64_` Fortran facades are emitted
 * by EPBLAS_FACADE_SYMM around this single ptrdiff_t `xsymm_core`.
 */

#include "xsymm_kernel.h"
#include "../common/blas_char.h"
#include "../common/blas_math.h"
#include "xl3_complex.h"
#include "../common/epblas_facade.h"
#include <ctype.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef __float128 R;
typedef xsymm_TC TC;

#define MR QBLAS_XGEMM_MR


static void xsymm_core(
    char side, char uplo,
    ptrdiff_t m, ptrdiff_t n,
    const xsymm_TC *alpha_,
    const xsymm_TC *a, ptrdiff_t lda,
    const xsymm_TC *b, ptrdiff_t ldb,
    const xsymm_TC *beta_,
    xsymm_TC *c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        xsymm_serial(side, uplo, m, n, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const R alphar = __real__ *alpha_, alphai = __imag__ *alpha_;
    const R beta_r = __real__ *beta_,  beta_i = __imag__ *beta_;
    const char sd = blas_up(side);
    const char up = blas_up(uplo);

    if (m <= 0 || n <= 0) return;

    R *C = (R *)c;
    qblas_xgemm_beta(m, n, beta_r, beta_i, C, ldc);
    if (alphar == 0.0Q && alphai == 0.0Q) return;

    const R *A_eff = (const R *)((sd == 'L') ? a : b);
    const R *B_eff = (const R *)((sd == 'L') ? b : a);
    const ptrdiff_t lda_eff = (sd == 'L') ? lda : ldb;
    const ptrdiff_t ldb_eff = (sd == 'L') ? ldb : lda;

    xsymm_plan_t p;
    xsymm_make_plan(m, n, sd, up, &p);
    if (p.k == 0) return;

#ifdef _OPENMP
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif

    long mnk = (long)m * (long)n * (long)p.k;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

    if (nthreads == 1) {
        R *Ap = aligned_alloc(64, (p.ap_bytes + 63) & ~(size_t)63);
        if (!Ap) return;
        R *Bp = aligned_alloc(64, (p.bp_bytes + 63) & ~(size_t)63);
        if (!Bp) { free(Ap); return; }
        for (ptrdiff_t js = 0; js < n; js += p.NC) {
            ptrdiff_t jb = (n - js < p.NC) ? (n - js) : p.NC;
            for (ptrdiff_t ls = 0; ls < p.k; ls += p.KC) {
                ptrdiff_t pb = (p.k - ls < p.KC) ? (p.k - ls) : p.KC;
                xsymm_pack_B(&p, B_eff, ldb_eff, js, ls, pb, jb, Bp);
                xsymm_level3_slab(0, m, &p, alphar, alphai,
                                  A_eff, lda_eff, Ap, Bp, js, ls, pb, jb, C, ldc);
            }
        }
        free(Bp);
        free(Ap);
        return;
    }

    R *Bp = aligned_alloc(64, (p.bp_bytes + 63) & ~(size_t)63);
    if (!Bp) return;
    R **Ap_arr = calloc((size_t)nthreads, sizeof(R *));
    if (!Ap_arr) { free(Bp); return; }
    bool alloc_ok = 1;
    for (ptrdiff_t t = 0; t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (p.ap_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (ptrdiff_t t = 0; t < nthreads; ++t) free(Ap_arr[t]);
        free(Ap_arr); free(Bp);
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        ptrdiff_t tid = omp_get_thread_num();
        ptrdiff_t nth = omp_get_num_threads();
#else
        ptrdiff_t tid = 0, nth = 1;
#endif
        R *Ap = Ap_arr[tid];

        ptrdiff_t m_chunk = blas_round_up((m + nth - 1) / nth, MR);
        ptrdiff_t m_lo = (ptrdiff_t)tid * m_chunk;
        ptrdiff_t m_hi = m_lo + m_chunk;
        if (m_hi > m) m_hi = m;

        for (ptrdiff_t js = 0; js < n; js += p.NC) {
            ptrdiff_t jb = (n - js < p.NC) ? (n - js) : p.NC;
            for (ptrdiff_t ls = 0; ls < p.k; ls += p.KC) {
                ptrdiff_t pb = (p.k - ls < p.KC) ? (p.k - ls) : p.KC;
#ifdef _OPENMP
                #pragma omp barrier
                #pragma omp single
#endif
                {
                    xsymm_pack_B(&p, B_eff, ldb_eff, js, ls, pb, jb, Bp);
                }
                if (m_lo < m_hi)
                    xsymm_level3_slab(m_lo, m_hi, &p, alphar, alphai,
                                      A_eff, lda_eff, Ap, Bp, js, ls, pb, jb, C, ldc);
            }
        }
    }

    for (ptrdiff_t t = 0; t < nthreads; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}

EPBLAS_FACADE_SYMM(xsymm, TC)
