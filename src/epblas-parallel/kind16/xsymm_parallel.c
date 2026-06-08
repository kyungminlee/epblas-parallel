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
 * Nesting guard: when xsymm_ is called from inside another routine's
 * parallel region it delegates to xsymm_serial_ and opens no region.
 */

#include "xsymm_kernel.h"
#include "xl3_complex.h"
#include <ctype.h>
#include <stddef.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef __float128 R;

#define MR QBLAS_YGEMM_MR

static int round_up(int v, int m) { return ((v + m - 1) / m) * m; }

void xsymm_(
    const char *side, const char *uplo,
    const int *m_, const int *n_,
    const xsymm_T *alpha_,
    const xsymm_T *a, const int *lda_,
    const xsymm_T *b, const int *ldb_,
    const xsymm_T *beta_,
    xsymm_T *c, const int *ldc_,
    size_t side_len, size_t uplo_len)
{
#ifdef _OPENMP
    if (omp_in_parallel()) {
        xsymm_serial_(side, uplo, m_, n_, alpha_, a, lda_, b, ldb_,
                      beta_, c, ldc_, side_len, uplo_len);
        return;
    }
#endif
    (void)side_len; (void)uplo_len;
    const int M = *m_, N = *n_;
    const R alphar = __real__ *alpha_, alphai = __imag__ *alpha_;
    const R beta_r = __real__ *beta_,  beta_i = __imag__ *beta_;
    const int sd = (char)toupper((unsigned char)*side);
    const int up = (char)toupper((unsigned char)*uplo);
    const int ldc = *ldc_;

    if (M <= 0 || N <= 0) return;

    R *C = (R *)c;
    qblas_ygemm_beta((ptrdiff_t)M, (ptrdiff_t)N, beta_r, beta_i, C, (ptrdiff_t)ldc);
    if (alphar == 0.0Q && alphai == 0.0Q) return;

    const R *A_eff = (const R *)((sd == 'L') ? a : b);
    const R *B_eff = (const R *)((sd == 'L') ? b : a);
    const int lda_eff = (sd == 'L') ? *lda_ : *ldb_;
    const int ldb_eff = (sd == 'L') ? *ldb_ : *lda_;

    xsymm_plan_t p;
    xsymm_make_plan(M, N, sd, up, &p);
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
        for (int js = 0; js < N; js += p.NC) {
            int jb = (N - js < p.NC) ? (N - js) : p.NC;
            for (int ls = 0; ls < p.K; ls += p.KC) {
                int pb = (p.K - ls < p.KC) ? (p.K - ls) : p.KC;
                xsymm_pack_B(&p, B_eff, ldb_eff, js, ls, pb, jb, Bp);
                xsymm_level3_slab(0, M, &p, alphar, alphai,
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

        int m_chunk = round_up((M + nth - 1) / nth, MR);
        int m_lo = tid * m_chunk;
        int m_hi = m_lo + m_chunk;
        if (m_hi > M) m_hi = M;

        for (int js = 0; js < N; js += p.NC) {
            int jb = (N - js < p.NC) ? (N - js) : p.NC;
            for (int ls = 0; ls < p.K; ls += p.KC) {
                int pb = (p.K - ls < p.KC) ? (p.K - ls) : p.KC;
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

    for (int t = 0; t < nthreads; ++t) free(Ap_arr[t]);
    free(Ap_arr);
    free(Bp);
}
