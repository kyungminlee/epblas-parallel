/*
 * xgemm_ — kind16 complex (COMPLEX(KIND=16) / __complex128) GEMM, public
 * Fortran entry. THREADING ORCHESTRATION ONLY: all the math lives in
 * xgemm_serial.c (the per-tile compute kernel and trans decode), shared
 * through xgemm_kernel.h.
 *
 *   C := alpha * op(A) * op(B) + beta * C
 *
 * 2D tile decomposition: the (M, N) output is partitioned into MB × NB
 * tiles; threads consume tiles via `collapse(2)`. K stays serial inside
 * each tile. Tile side defaults to ≈ sqrt(M*N / (4*nthreads)) clamped to
 * [16, 128]; env overrides XGEMM_MB / XGEMM_NB pin specific sides.
 *
 * As a defensive guard against accidental nested parallelism, falls back to
 * the serial kernel if invoked from inside another parallel region.
 *
 * Fortran ABI: name lowercased + trailing underscore; scalars by pointer;
 * character args followed by hidden trailing size_t lengths; COMPLEX(KIND=16)
 * ↔ __complex128.
 */

#include "xgemm_kernel.h"
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef xgemm_T T;

/* Cached env overrides. 0 = uninitialized; >0 = override; <0 = use heuristic. */
static int xgemm_mb_cached = 0;
static int xgemm_nb_cached = 0;

static int xgemm_read_dim_env(const char *name) {
    const char *s = getenv(name);
    if (!s || !*s) return -1;
    int v = atoi(s);
    return (v > 0) ? v : -1;
}

static int xgemm_mb_override(void) {
    int v = __atomic_load_n(&xgemm_mb_cached, __ATOMIC_RELAXED);
    if (__builtin_expect(v == 0, 0)) {
        v = xgemm_read_dim_env("XGEMM_MB");
        if (v == 0) v = -1;
        __atomic_store_n(&xgemm_mb_cached, v, __ATOMIC_RELAXED);
    }
    return v;
}

static int xgemm_nb_override(void) {
    int v = __atomic_load_n(&xgemm_nb_cached, __ATOMIC_RELAXED);
    if (__builtin_expect(v == 0, 0)) {
        v = xgemm_read_dim_env("XGEMM_NB");
        if (v == 0) v = -1;
        __atomic_store_n(&xgemm_nb_cached, v, __ATOMIC_RELAXED);
    }
    return v;
}

/* Pick a square tile side ≈ sqrt(M*N / (4*nthreads)) clamped to
 * power-of-two sides in [16, 128]. */
static int xgemm_tile_side(int M, int N, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    const size_t area = (size_t)M * (size_t)N;
    const size_t target_tiles = (size_t)nthreads * 4u;
    const size_t per_tile = target_tiles ? (area / target_tiles) : area;
    int s = 16;
    while (s < 128 && (size_t)(s * 2) * (size_t)(s * 2) <= per_tile) s *= 2;
    return s;
}

void xgemm_(
    const char *transa, const char *transb,
    const int *m_, const int *n_, const int *k_,
    const T *alpha_,
    const T *a, const int *lda_,
    const T *b, const int *ldb_,
    const T *beta_,
    T *c, const int *ldc_,
    size_t transa_len, size_t transb_len)
{
    const int M = *m_, N = *n_, K = *k_;
    const int lda = *lda_, ldb = *ldb_, ldc = *ldc_;
    const T alpha = *alpha_, beta = *beta_;
    const int ta = xgemm_trans_code(transa, transa_len);
    const int tb = xgemm_trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    const int conj_a = (ta == 'C');
    const int conj_b = (tb == 'C');
    const int trans_a = (ta != 'N');
    const int trans_b = (tb != 'N');

#ifdef _OPENMP
    const int nthreads_max = blas_omp_max_threads();
    const int in_parallel  = omp_in_parallel();
#else
    const int nthreads_max = 1;
    const int in_parallel  = 0;
#endif

    int MB, NB;
    {
        int mb_env = xgemm_mb_override();
        int nb_env = xgemm_nb_override();
        if (mb_env > 0 && nb_env > 0) {
            MB = mb_env; NB = nb_env;
        } else {
            int side = xgemm_tile_side(M, N, nthreads_max);
            MB = (mb_env > 0) ? mb_env : side;
            NB = (nb_env > 0) ? nb_env : side;
        }
    }
    if (MB > M) MB = M;
    if (NB > N) NB = N;
    const int nt_m = (M + MB - 1) / MB;
    const int nt_n = (N + NB - 1) / NB;
    const int total_tiles = nt_m * nt_n;

#ifdef _OPENMP
    const int use_omp = (total_tiles >= 2 && nthreads_max > 1 && !in_parallel);
    #pragma omp parallel for collapse(2) if(use_omp) schedule(static)
#endif
    for (int jt = 0; jt < nt_n; ++jt) {
        for (int it = 0; it < nt_m; ++it) {
            const int j0 = jt * NB;
            const int j1 = (j0 + NB < N) ? (j0 + NB) : N;
            const int i0 = it * MB;
            const int i1 = (i0 + MB < M) ? (i0 + MB) : M;
            xgemm_tile_compute(i0, i1, j0, j1, K,
                               trans_a, conj_a, trans_b, conj_b,
                               alpha, beta, a, lda, b, ldb, c, ldc);
        }
    }
}
