/*
 * xgemm_ — kind16 complex GEMM (COMPLEX(KIND=16) / __complex128), the
 * public Fortran entry and threading-orchestration half of the xgemm
 * overlay (see xgemm_kernel.h for the split rationale; all the math lives
 * in xgemm_serial.c).
 *
 *   C := alpha * op(A) * op(B) + beta * C
 *
 * Parallel shape: one `omp parallel` block-partitions the M axis across the
 * team (rounded to MR). Each thread keeps a private Ap pack buffer; the Bp
 * panel is shared and packed once per (js, ls) under `omp single`, bracketed
 * by an explicit barrier (no thread still reading the previous Bp) and the
 * single's implicit end barrier (new Bp visible before any kernel run). This
 * is the OpenBLAS level3.c SMP shape with a single-level OpenMP team in place
 * of the blas_queue runtime. The beta pre-pass runs once up front.
 *
 * Nesting guard: when xgemm_ is itself called from inside another routine's
 * parallel region (the complex L3 family — xtrsm, xtrmm, xsyrk, … runs
 * xgemm trailing updates inside its own `omp parallel`), it delegates to
 * xgemm_serial_ and opens no region of its own.
 */

#include "xgemm_kernel.h"
#include "xl3_complex.h"
#include "../common/epblas_facade.h"
#include <stddef.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef __float128 R;
typedef xgemm_TC TC;

#define MR QBLAS_YGEMM_MR

static ptrdiff_t round_up(ptrdiff_t v, ptrdiff_t m) { return ((v + m - 1) / m) * m; }

static void xgemm_core(
    char transa, char transb,
    ptrdiff_t m, ptrdiff_t n, ptrdiff_t k,
    const xgemm_TC *alpha_,
    const xgemm_TC *a, ptrdiff_t lda,
    const xgemm_TC *b, ptrdiff_t ldb,
    const xgemm_TC *beta_,
    xgemm_TC *c, ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Called from inside another routine's parallel region: run fully
     * serial, opening no team of our own. */
    if (omp_in_parallel()) {
        xgemm_serial(transa, transb, m, n, k, alpha_, a, lda,
                     b, ldb, beta_, c, ldc);
        return;
    }
#endif

    const R alphar = __real__ *alpha_, alphai = __imag__ *alpha_;
    const R beta_r = __real__ *beta_,  beta_i = __imag__ *beta_;
    const char ta = xgemm_trans_code(transa);
    const char tb = xgemm_trans_code(transb);

    if (m <= 0 || n <= 0) return;

    const R *A = (const R *)a;
    const R *B = (const R *)b;
    R *C = (R *)c;

    qblas_ygemm_beta(m, n, beta_r, beta_i, C, ldc);
    if (k == 0 || (alphar == 0.0Q && alphai == 0.0Q)) return;

    xgemm_plan_t p;
    xgemm_make_plan(m, n, k, ta, tb, &p);

#ifdef _OPENMP
    ptrdiff_t nthreads = blas_omp_max_threads();
    if (nthreads < 1) nthreads = 1;
#else
    ptrdiff_t nthreads = 1;
#endif

    /* Don't fan out for tiny problems — overhead exceeds work. */
    long mnk = (long)m * (long)n * (long)k;
    if (mnk < 64L * 64L * 64L) nthreads = 1;

    if (nthreads == 1) {
        R *Ap = aligned_alloc(64, (p.ap_bytes + 63) & ~(size_t)63);
        if (!Ap) return;
        R *Bp = aligned_alloc(64, (p.bp_bytes + 63) & ~(size_t)63);
        if (!Bp) { free(Ap); return; }
        for (ptrdiff_t js = 0; js < n; js += p.NC) {
            ptrdiff_t jb = (n - js < p.NC) ? (n - js) : p.NC;
            for (ptrdiff_t ls = 0; ls < k; ls += p.KC) {
                ptrdiff_t pb = (k - ls < p.KC) ? (k - ls) : p.KC;
                xgemm_pack_B(&p, B, ldb, js, ls, pb, jb, Bp);
                xgemm_level3_slab(0, m, &p, alphar, alphai,
                                  A, lda, Ap, Bp, js, ls, pb, jb, C, ldc);
            }
        }
        free(Bp);
        free(Ap);
        return;
    }

    /* Allocate per-thread Ap and the shared Bp BEFORE the parallel region so
     * every thread in the team hits every `omp barrier` / `omp single`. */
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

        ptrdiff_t m_chunk = round_up((m + nth - 1) / nth, MR);
        ptrdiff_t m_lo = (ptrdiff_t)tid * m_chunk;
        ptrdiff_t m_hi = m_lo + m_chunk;
        if (m_hi > m) m_hi = m;

        for (ptrdiff_t js = 0; js < n; js += p.NC) {
            ptrdiff_t jb = (n - js < p.NC) ? (n - js) : p.NC;
            for (ptrdiff_t ls = 0; ls < k; ls += p.KC) {
                ptrdiff_t pb = (k - ls < p.KC) ? (k - ls) : p.KC;
#ifdef _OPENMP
                #pragma omp barrier
                #pragma omp single
#endif
                {
                    xgemm_pack_B(&p, B, ldb, js, ls, pb, jb, Bp);
                }
                if (m_lo < m_hi)
                    xgemm_level3_slab(m_lo, m_hi, &p, alphar, alphai,
                                      A, lda, Ap, Bp, js, ls, pb, jb, C, ldc);
            }
        }
    }

    for (ptrdiff_t t = 0; t < nthreads; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}

EPBLAS_FACADE_GEMM(xgemm, TC)
