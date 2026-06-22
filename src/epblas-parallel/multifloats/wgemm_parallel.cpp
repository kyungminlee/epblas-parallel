/*
 * wgemm_ — multifloats complex GEMM (complex64x2, complex double-double),
 * public Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * wgemm_serial.cpp (scalar + SIMD complex packers and micro-kernels, block
 * policy), shared through wgemm_kernel.h.
 *
 *   C := alpha * op(A) * op(B) + beta * C
 *
 * Two regimes (mirrors mgemm_parallel.cpp):
 *   - Called from inside another OpenMP team (the L3 family running wgemm
 *     trailing updates inside their own `omp parallel`): open NO nested
 *     region, run the single-thread kernel in the calling thread via
 *     wgemm_serial. Previously handled implicitly by OpenMP's default-off
 *     nesting collapsing the inner team to one; the explicit delegate avoids
 *     that per-call team setup and matches kind10/16.
 *   - Called at top level: fan the jc loop (N axis) across the team, each
 *     thread holding private Ap / Bp packing buffers.
 */

#include "wgemm_kernel.h"
#include "mf_kernels.h"
#include "mf_pred.h"
#include "../common/epblas_facade.h"
#include <cstddef>
#include <cstdlib>
#ifdef _OPENMP
#include <omp.h>
#endif

namespace mf = multifloats;
using R = mf::float64x2;
using T = mf::complex64x2;

namespace {
using mf_kernels::cmul;
using mf_pred::ceq0;
using mf_pred::ceq1;
const T zero_cdd{ R{0.0, 0.0}, R{0.0, 0.0} };
}  // namespace

static void wgemm_core(
    char transa, char transb,
    std::ptrdiff_t m, std::ptrdiff_t n, std::ptrdiff_t k,
    const T *alpha_,
    const T *a, std::ptrdiff_t lda,
    const T *b, std::ptrdiff_t ldb,
    const T *beta_,
    T *c, std::ptrdiff_t ldc)
{
#ifdef _OPENMP
    /* Already inside a team → run serially in this thread, no nested region. */
    if (omp_in_parallel()) {
        wgemm_serial(transa, transb, m, n, k, alpha_, a, lda,
                     b, ldb, beta_, c, ldc);
        return;
    }
#endif

    const T alpha = *alpha_, beta = *beta_;
    const std::ptrdiff_t ta = wgemm_trans_code(&transa, 1);
    const std::ptrdiff_t tb = wgemm_trans_code(&transb, 1);

    if (m <= 0 || n <= 0) return;

    /* beta pre-pass runs serially in the calling thread (matches the
     * pre-split wgemm_). */
    for (std::ptrdiff_t j = 0; j < n; ++j) {
        T *cj = &c[static_cast<std::size_t>(j) * ldc];
        if (ceq0(beta)) {
            for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = zero_cdd;
        } else if (!ceq1(beta)) {
            for (std::ptrdiff_t i = 0; i < m; ++i) cj[i] = cmul(cj[i], beta);
        }
    }
    if (ceq0(alpha) || k == 0) return;

    std::ptrdiff_t MC, KC, NC;
    wgemm_choose_blocks(&MC, &KC, &NC);

#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
        T *Ap = static_cast<T *>(std::aligned_alloc(
            64, static_cast<std::size_t>(MC) * KC * sizeof(T)));
#ifdef WBLAS_SIMD_DD
        const std::ptrdiff_t W_simd = wgemm_simd_pack_W();
        const std::ptrdiff_t NC_pad = ((NC + W_simd - 1) / W_simd) * W_simd;
        double *Bp_rh = static_cast<double *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
        double *Bp_rl = static_cast<double *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
        double *Bp_ih = static_cast<double *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
        double *Bp_il = static_cast<double *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC_pad * sizeof(double)));
        if (Ap && Bp_rh && Bp_rl && Bp_ih && Bp_il) {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t jc = 0; jc < n; jc += NC) {
                const std::ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
                for (std::ptrdiff_t pc = 0; pc < k; pc += KC) {
                    const std::ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
                    wgemm_pack_B_soa_complex(b, ldb, pc, jc, pb, jb, tb,
                                             Bp_rh, Bp_rl, Bp_ih, Bp_il);
                    for (std::ptrdiff_t ic = 0; ic < m; ic += MC) {
                        const std::ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
                        wgemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                        wgemm_inner_kernel_simd_complex(ib, jb, pb, alpha, Ap,
                                                        Bp_rh, Bp_rl, Bp_ih, Bp_il,
                                                        &c[static_cast<std::size_t>(jc) * ldc + ic],
                                                        ldc);
                    }
                }
            }
        }
        std::free(Ap);
        std::free(Bp_rh);
        std::free(Bp_rl);
        std::free(Bp_ih);
        std::free(Bp_il);
#else
        T *Bp = static_cast<T *>(std::aligned_alloc(
            64, static_cast<std::size_t>(KC) * NC * sizeof(T)));
        if (Ap && Bp) {
#ifdef _OPENMP
            #pragma omp for schedule(static)
#endif
            for (std::ptrdiff_t jc = 0; jc < n; jc += NC) {
                const std::ptrdiff_t jb = (n - jc < NC) ? (n - jc) : NC;
                for (std::ptrdiff_t pc = 0; pc < k; pc += KC) {
                    const std::ptrdiff_t pb = (k - pc < KC) ? (k - pc) : KC;
                    wgemm_pack_B(b, ldb, pc, jc, pb, jb, tb, Bp);
                    for (std::ptrdiff_t ic = 0; ic < m; ic += MC) {
                        const std::ptrdiff_t ib = (m - ic < MC) ? (m - ic) : MC;
                        wgemm_pack_A(a, lda, ic, pc, ib, pb, ta, Ap);
                        wgemm_inner_kernel(ib, jb, pb, alpha, Ap, Bp,
                                           &c[static_cast<std::size_t>(jc) * ldc + ic],
                                           ldc);
                    }
                }
            }
        }
        std::free(Ap);
        std::free(Bp);
#endif /* WBLAS_SIMD_DD */
    }
}

extern "C" {
EPBLAS_FACADE_GEMM(wgemm, T)
}
