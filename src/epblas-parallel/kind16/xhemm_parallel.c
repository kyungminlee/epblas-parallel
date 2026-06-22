/*
 * xhemm_ — kind16 complex HEMM (COMPLEX(KIND=16) / __complex128), the
 * public Fortran entry and threading-orchestration half of the xhemm
 * overlay (see xhemm_kernel.h; all the math lives in xhemm_serial.c).
 *
 *   C := alpha * A * B + beta * C    (SIDE=L, A Hermitian M×M)
 *   C := alpha * B * A + beta * C    (SIDE=R, A Hermitian N×N)
 *
 * Parallel shape (the OpenBLAS level3.c SMP shape): one `omp parallel`
 * block-partitions the M axis across the team (rounded to MR). Each thread
 * keeps a private Ap pack buffer; the Bp panel is shared and packed once
 * per (js, ls) under `omp single`, bracketed by an explicit barrier and the
 * single's implicit end barrier. The beta pre-pass runs once up front.
 *
 * Nesting guard: when the core is called from inside another routine's
 * parallel region it delegates to xhemm_serial and opens no region.
 *
 * ABI: the LP64 `xhemm_` and ILP64 `xhemm_64_` Fortran facades are emitted
 * by EPBLAS_FACADE_SYMM around this single ptrdiff_t `xhemm_core`.
 */

#include "xhemm_kernel.h"
#include "xl3_complex.h"
#include "../common/epblas_facade.h"
#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef __float128 R;
typedef xhemm_T T;

#define MR QBLAS_YGEMM_MR

static ptrdiff_t round_up(ptrdiff_t v, ptrdiff_t m) { return ((v + m - 1) / m) * m; }

static void xhemm_core(
    char side, char uplo,
    ptrdiff_t M, ptrdiff_t N,
    const xhemm_T *alpha_,
    const xhemm_T *a, ptrdiff_t lda,
    const xhemm_T *b, ptrdiff_t ldb,
    const xhemm_T *beta_,
    xhemm_T *c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        xhemm_serial(side, uplo, M, N, alpha_, a, lda, b, ldb, beta_, c, ldc);
        return;
    }
#endif
    const R alphar = __real__ *alpha_, alphai = __imag__ *alpha_;
    const R beta_r = __real__ *beta_,  beta_i = __imag__ *beta_;
    const int sd = (char)toupper((unsigned char)side);
    const int up = (char)toupper((unsigned char)uplo);

    if (M <= 0 || N <= 0) return;

    R *C = (R *)c;
    qblas_ygemm_beta(M, N, beta_r, beta_i, C, ldc);
    if (alphar == 0.0Q && alphai == 0.0Q) return;

    const R *A_eff = (const R *)((sd == 'L') ? a : b);
    const R *B_eff = (const R *)((sd == 'L') ? b : a);
    const ptrdiff_t lda_eff = (sd == 'L') ? lda : ldb;
    const ptrdiff_t ldb_eff = (sd == 'L') ? ldb : lda;

    xhemm_plan_t p;
    xhemm_make_plan(M, N, sd, up, &p);
    if (p.K == 0) return;

#ifdef _OPENMP
    int nthreads = blas_omp_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    int nthreads = 1;
#endif

    long mnk = (long)M * (long)N * (long)p.K;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

    if (nthreads == 1) {
        R *Ap = aligned_alloc(64, (p.ap_bytes + 63) & ~(size_t)63);
        if (!Ap) return;
        R *Bp = aligned_alloc(64, (p.bp_bytes + 63) & ~(size_t)63);
        if (!Bp) { free(Ap); return; }
        for (ptrdiff_t js = 0; js < N; js += p.NC) {
            ptrdiff_t jb = (N - js < p.NC) ? (N - js) : p.NC;
            for (ptrdiff_t ls = 0; ls < p.K; ls += p.KC) {
                ptrdiff_t pb = (p.K - ls < p.KC) ? (p.K - ls) : p.KC;
                xhemm_pack_B(&p, B_eff, ldb_eff, js, ls, pb, jb, Bp);
                xhemm_level3_slab(0, M, &p, alphar, alphai,
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
    int alloc_ok = 1;
    for (int t = 0; t < nthreads; ++t) {
        Ap_arr[t] = aligned_alloc(64, (p.ap_bytes + 63) & ~(size_t)63);
        if (!Ap_arr[t]) { alloc_ok = 0; break; }
    }
    if (!alloc_ok) {
        for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
        free(Ap_arr); free(Bp);
        return;
    }

#ifdef _OPENMP
    #pragma omp parallel num_threads(nthreads)
#endif
    {
#ifdef _OPENMP
        int tid = omp_get_thread_num();
        int nth = omp_get_num_threads();
#else
        int tid = 0, nth = 1;
#endif
        R *Ap = Ap_arr[tid];

        ptrdiff_t m_chunk = round_up((M + nth - 1) / nth, MR);
        ptrdiff_t m_lo = (ptrdiff_t)tid * m_chunk;
        ptrdiff_t m_hi = m_lo + m_chunk;
        if (m_hi > M) m_hi = M;

        for (ptrdiff_t js = 0; js < N; js += p.NC) {
            ptrdiff_t jb = (N - js < p.NC) ? (N - js) : p.NC;
            for (ptrdiff_t ls = 0; ls < p.K; ls += p.KC) {
                ptrdiff_t pb = (p.K - ls < p.KC) ? (p.K - ls) : p.KC;
#ifdef _OPENMP
                #pragma omp barrier
                #pragma omp single
#endif
                {
                    xhemm_pack_B(&p, B_eff, ldb_eff, js, ls, pb, jb, Bp);
                }
                if (m_lo < m_hi)
                    xhemm_level3_slab(m_lo, m_hi, &p, alphar, alphai,
                                      A_eff, lda_eff, Ap, Bp, js, ls, pb, jb, C, ldc);
            }
        }
    }

    for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}

EPBLAS_FACADE_SYMM(xhemm, T)
