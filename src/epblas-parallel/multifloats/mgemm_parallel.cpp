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
#include "../common/epblas_facade.h"
#include <cstddef>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf = multifloats;
using TR = mf::float64x2;

namespace {
const TR zero_dd{0.0, 0.0};
const TR one_dd {1.0, 0.0};
}  // namespace

static void mgemm_core(
    char transa, char transb,
    std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
    const TR *alpha_,
    const TR *a, std::ptrdiff_t lda,
    const TR *b, std::ptrdiff_t ldb,
    const TR *beta_,
    TR *c, std::ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Already inside a team → run serially in this thread, no nested region. */
    if (omp_in_parallel()) {
        mgemm_serial(transa, transb, m, n, k, alpha_, a, lda,
                     b, ldb, beta_, c, ldc);
        return;
    }
#endif

    const TR alpha = *alpha_, beta = *beta_;
    const std::ptrdiff_t ta = mgemm_trans_code(&transa, 1);
    const std::ptrdiff_t tb = mgemm_trans_code(&transb, 1);

    if (m <= 0 || n <= 0) return;

    /* beta pre-pass runs serially in the calling thread (matches the
     * pre-split mgemm_). */
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        TR *cj = &c[static_cast<std::size_t>(j) * ldc];
        if (beta == zero_dd) {
            for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = zero_dd;
        } else if (beta != one_dd) {
            for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cj[i] * beta;
        }
    }
    if (alpha == zero_dd || k == 0) return;

    std::ptrdiff_t MC, KC, NC;
    mgemm_choose_blocks(&MC, &KC, &NC);

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        TR *Ap = static_cast<TR *>(std::aligned_alloc(
            64, static_cast<std::size_t>(MC) * KC * sizeof(TR)));
#ifdef MBLAS_SIMD_DD
        /* SoA Bp: round NC up to W = NR_LANE * NR_PAN for the trailing panel. */
        const std::ptrdiff_t W_simd = mgemm_simd_pack_W();
        const std::ptrdiff_t NC_pad = ((NC + W_simd - 1) / W_simd) * W_simd;
        double *Bp_hi = static_cast<double *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
        double *Bp_lo = static_cast<double *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
        if (Ap && Bp_hi && Bp_lo) {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t jc = 0; jc < n; jc += NC) {
                const std::ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
                for (std::ptrdiff_t pc = 0; pc < k; pc += KC) {
                    const std::ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
                    mgemm_pack_B_soa(b, ldb, pc, jc, pb, jb, tb, Bp_hi, Bp_lo);
                    for (std::ptrdiff_t ic = 0; ic < m; ic += MC) {
                        const std::ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
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
        TR *Bp = static_cast<TR *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC * sizeof(TR)));
        if (Ap && Bp) {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t jc = 0; jc < n; jc += NC) {
                const std::ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
                for (std::ptrdiff_t pc = 0; pc < k; pc += KC) {
                    const std::ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
                    mgemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                    for (std::ptrdiff_t ic = 0; ic < m; ic += MC) {
                        const std::ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
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

extern "C" {
EPBLAS_FACADE_GEMM(mgemm, TR)
}
