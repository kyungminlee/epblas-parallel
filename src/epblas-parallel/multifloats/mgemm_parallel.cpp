/*
 * mgemm_ — multifloats real GEMM (float64x2, double-double), public Fortran
 * entry. THREADING ORCHESTRATION ONLY: all the math lives in mgemm_serial.cpp
 * (scalar + SIMD packers and micro-kernels, block-size policy), shared
 * through mgemm_kernel.h.
 *
 *   C := alpha * op(A) * op(B) + beta * C
 *
 * Two regimes:
 *   - Called from inside another OpenMP team (the L3 family — mtrsm, mtrmm,
 *     mgemmtr, msyrk, msymm, msyr2k, and the w* twins, run mgemm trailing
 *     updates inside their own `omp parallel`): open NO nested region, run
 *     the single-thread kernel in the calling thread. Previously this case
 *     was handled implicitly — the inner `#pragma omp parallel` collapsed to
 *     a team of one under OpenMP's default-off nesting — but an explicit
 *     serial delegate avoids that per-call team setup and matches kind10/16.
 *   - Called at top level: fan the jc loop (N axis) across the team, each
 *     thread holding private Ap / Bp packing buffers.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths; REAL64x2 ↔ the
 * POD `float64x2` (sizeof == 2*double), matching gfortran's `type(real64x2)`.
 */

#include "mgemm_kernel.h"
#include <cstddef>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf = multifloats;
using T = mf::float64x2;

namespace {
const T zero_dd{0.0, 0.0};
const T one_dd {1.0, 0.0};
}  // namespace

extern "C" void mgemm_(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    std::size_t transa_len, std::size_t transb_len)
{
#ifdef _OPENMP
    /* Already inside a team → run serially in this thread, no nested region. */
    if (omp_in_parallel()) {
        mgemm_serial(transa, transb, m_, n_, k_, alpha_, a, lda_,
                     b, ldb_, beta_, c, ldc_, transa_len, transb_len);
        return;
    }
#endif

    const int M = *m_, N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const int ta = mgemm_trans_code(transa, transa_len);
    const int tb = mgemm_trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    /* beta pre-pass runs serially in the calling thread (matches the
     * pre-split mgemm_). */
    for (int j = 0; j < N; ++j) {
        T *cj = &c[static_cast<std::size_t>(j) * ldc];
        if (beta == zero_dd) {
            for (int i = 0; i < M; ++i) cj[i] = zero_dd;
        } else if (beta != one_dd) {
            for (int i = 0; i < M; ++i) cj[i] = cj[i] * beta;
        }
    }
    if (alpha == zero_dd || K == 0) return;

    int MC, KC, NC;
    mgemm_choose_blocks(&MC, &KC, &NC);

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        T *Ap = static_cast<T *>(std::aligned_alloc(
            64, static_cast<std::size_t>(MC) * KC * sizeof(T)));
#ifdef MBLAS_SIMD_DD
        /* SoA Bp: round NC up to W = NR_LANE * NR_PAN for the trailing panel. */
        const int W_simd = mgemm_simd_pack_W();
        const int NC_pad = ((NC + W_simd - 1) / W_simd) * W_simd;
        double *Bp_hi = static_cast<double *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
        double *Bp_lo = static_cast<double *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
        if (Ap && Bp_hi && Bp_lo) {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int jc = 0; jc < N; jc += NC) {
                const int jb = (N - jc < NC) ? (N - jc) : NC;
                for (int pc = 0; pc < K; pc += KC) {
                    const int pb = (K - pc < KC) ? (K - pc) : KC;
                    mgemm_pack_B_soa(b, ldb, pc, jc, pb, jb, tb, Bp_hi, Bp_lo);
                    for (int ic = 0; ic < M; ic += MC) {
                        const int ib = (M - ic < MC) ? (M - ic) : MC;
                        mgemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                        mgemm_inner_kernel_simd(ib, jb, pb, alpha, Ap, Bp_hi, Bp_lo,
                                                &c[static_cast<std::size_t>(jc) * ldc + ic],
                                                ldc);
                    }
                }
            }
        }
        std::free(Ap);
        std::free(Bp_hi);
        std::free(Bp_lo);
#else
        T *Bp = static_cast<T *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC * sizeof(T)));
        if (Ap && Bp) {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (int jc = 0; jc < N; jc += NC) {
                const int jb = (N - jc < NC) ? (N - jc) : NC;
                for (int pc = 0; pc < K; pc += KC) {
                    const int pb = (K - pc < KC) ? (K - pc) : KC;
                    mgemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                    for (int ic = 0; ic < M; ic += MC) {
                        const int ib = (M - ic < MC) ? (M - ic) : MC;
                        mgemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                        mgemm_inner_kernel(ib, jb, pb, alpha, Ap, Bp,
                                           &c[static_cast<std::size_t>(jc) * ldc + ic],
                                           ldc);
                    }
                }
            }
        }
        std::free(Ap);
        std::free(Bp);
#endif /* MBLAS_SIMD_DD */
    }
}
