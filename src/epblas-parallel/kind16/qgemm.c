/*
 * qgemm — kind16 (REAL(KIND=16) / __float128) GEMM overlay.
 *
 * Two public entry points share a single per-tile compute kernel:
 *
 *   qgemm_         — top-level entry. Opens its own `#pragma omp parallel
 *                    for collapse(2)` over the (M, N) tile grid. Use this
 *                    from any non-OMP caller (LAPACK, user code, etc.).
 *
 *   qgemm_serial_  — bare serial entry. No OpenMP pragma anywhere on the
 *                    call path. Safe to invoke from inside another
 *                    function's `#pragma omp parallel` region — callers
 *                    that have already opened a parallel region (e.g. a
 *                    refactored qtrsv_blocked) hand each thread a slice
 *                    of the trailing matrix and call this on the slice.
 *
 * Both go through `qgemm_tile_compute` for numerics so the parallel and
 * serial paths produce bitwise-identical results.
 *
 * As a defensive guard, qgemm_ also checks `omp_in_parallel()` and skips
 * its own parallel region if invoked inside one — so a stray call from
 * an OMP region won't trigger nested parallelism.
 *
 * 2D tile decomposition: the (M, N) output is partitioned into MB × NB
 * tiles; threads consume tiles via `collapse(2)`. K stays serial inside
 * each tile. Tile side defaults to ≈ sqrt(M*N / (4*nthreads)) clamped to
 * [16, 128]; env overrides QGEMM_MB / QGEMM_NB pin specific sides.
 *
 * Beta scaling: applied once per tile to its (i, j) slice — each output
 * element belongs to exactly one tile under `collapse(2)`, so the
 * race-free per-tile beta pass matches the column-wise scheme.
 *
 * TRANSA / TRANSB independently in {N, T} (C ≡ T for real).
 */

#include <stddef.h>
#include <stdlib.h>
#include <ctype.h>
#include <quadmath.h>
#ifdef _OPENMP
#include <omp.h>
#include "../common/blas_omp.h"
#endif

typedef __float128 T;

static int trans_code(const char *p, size_t len) {
    (void)len;
    char c = (char)toupper((unsigned char)*p);
    return (c == 'C') ? 'T' : c;
}

#define A_(i, j)  a[(size_t)(j) * lda + (i)]
#define B_(i, j)  b[(size_t)(j) * ldb + (i)]
#define C_(i, j)  c[(size_t)(j) * ldc + (i)]

/* Cached env overrides. 0 = uninitialized; >0 = override; <0 = use heuristic. */
static int qgemm_mb_cached = 0;
static int qgemm_nb_cached = 0;

static int qgemm_read_dim_env(const char *name) {
    const char *s = getenv(name);
    if (!s || !*s) return -1;
    int v = atoi(s);
    return (v > 0) ? v : -1;
}

static int qgemm_mb_override(void) {
    int v = __atomic_load_n(&qgemm_mb_cached, __ATOMIC_RELAXED);
    if (__builtin_expect(v == 0, 0)) {
        v = qgemm_read_dim_env("QGEMM_MB");
        if (v == 0) v = -1;
        __atomic_store_n(&qgemm_mb_cached, v, __ATOMIC_RELAXED);
    }
    return v;
}

static int qgemm_nb_override(void) {
    int v = __atomic_load_n(&qgemm_nb_cached, __ATOMIC_RELAXED);
    if (__builtin_expect(v == 0, 0)) {
        v = qgemm_read_dim_env("QGEMM_NB");
        if (v == 0) v = -1;
        __atomic_store_n(&qgemm_nb_cached, v, __ATOMIC_RELAXED);
    }
    return v;
}

/* Pick a square tile side ≈ sqrt(M*N / (4*nthreads)) clamped to
 * power-of-two sides in [16, 128]. */
static int qgemm_tile_side(int M, int N, int nthreads) {
    if (nthreads < 1) nthreads = 1;
    const size_t area = (size_t)M * (size_t)N;
    const size_t target_tiles = (size_t)nthreads * 4u;
    const size_t per_tile = target_tiles ? (area / target_tiles) : area;
    int s = 16;
    while (s < 128 && (size_t)(s * 2) * (size_t)(s * 2) <= per_tile) s *= 2;
    return s;
}

/* Compute one tile of C[i0:i1, j0:j1]:
 *
 *   C[i,j] = beta * C[i,j] + alpha * op(A) * op(B) [k summed]
 *
 * Each (i, j) is owned by exactly one tile, so the beta pass is
 * race-free and the K accumulation writes only into this tile's slice.
 * No OpenMP pragmas — pure sequential per-tile work.
 */
static void qgemm_tile_compute(
    int i0, int i1, int j0, int j1, int K,
    int trans_a, int trans_b,
    T alpha, T beta,
    const T *a, int lda,
    const T *b, int ldb,
    T *c, int ldc)
{
    const T zero = 0.0Q;
    const T one  = 1.0Q;

    /* Beta pass. */
    for (int j = j0; j < j1; ++j) {
        T *cj = &C_(0, j);
        if (beta == zero)      for (int i = i0; i < i1; ++i) cj[i]  = zero;
        else if (beta != one)  for (int i = i0; i < i1; ++i) cj[i] *= beta;
    }

    if (alpha == zero || K == 0) return;

    if (!trans_a) {
        /* Rank-1 (axpy) form: TEMP = alpha · op(B)[k,j], then
         * C[:,j] += TEMP · A[:,k]. Hoist trans_b out of the k loop. */
        if (!trans_b) {
            for (int j = j0; j < j1; ++j) {
                T *cj = &C_(0, j);
                for (int k = 0; k < K; ++k) {
                    const T bkj = B_(k, j);
                    if (bkj != zero) {
                        const T t = alpha * bkj;
                        const T *ak = &A_(0, k);
                        for (int i = i0; i < i1; ++i) cj[i] += t * ak[i];
                    }
                }
            }
        } else {
            for (int j = j0; j < j1; ++j) {
                T *cj = &C_(0, j);
                for (int k = 0; k < K; ++k) {
                    const T bjk = B_(j, k);
                    if (bjk != zero) {
                        const T t = alpha * bjk;
                        const T *ak = &A_(0, k);
                        for (int i = i0; i < i1; ++i) cj[i] += t * ak[i];
                    }
                }
            }
        }
    } else {
        /* Inner-product (DDOT) form. */
        if (!trans_b) {
            for (int j = j0; j < j1; ++j) {
                T *cj = &C_(0, j);
                for (int i = i0; i < i1; ++i) {
                    T s = zero;
                    for (int k = 0; k < K; ++k) s += A_(k, i) * B_(k, j);
                    cj[i] += alpha * s;
                }
            }
        } else {
            for (int j = j0; j < j1; ++j) {
                T *cj = &C_(0, j);
                for (int i = i0; i < i1; ++i) {
                    T s = zero;
                    for (int k = 0; k < K; ++k) s += A_(k, i) * B_(j, k);
                    cj[i] += alpha * s;
                }
            }
        }
    }
}

/* Pure-serial entry. No OpenMP anywhere on this call path; safe to
 * invoke from inside another function's `#pragma omp parallel` region.
 * Callers are responsible for partitioning if they want thread parallelism. */
void qgemm_serial_(
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
    const int ta = trans_code(transa, transa_len);
    const int tb = trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

    const int trans_a = (ta != 'N');
    const int trans_b = (tb != 'N');

    qgemm_tile_compute(0, M, 0, N, K,
                       trans_a, trans_b,
                       alpha, beta, a, lda, b, ldb, c, ldc);
}

/* Parallel entry. Opens its own `#pragma omp parallel for collapse(2)`
 * over the 2D tile grid. As a defensive guard against accidental nested
 * parallelism, falls back to serial if invoked from inside another
 * parallel region. */
void qgemm_(
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
    const int ta = trans_code(transa, transa_len);
    const int tb = trans_code(transb, transb_len);

    if (M <= 0 || N <= 0) return;

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
        int mb_env = qgemm_mb_override();
        int nb_env = qgemm_nb_override();
        if (mb_env > 0 && nb_env > 0) {
            MB = mb_env; NB = nb_env;
        } else {
            int side = qgemm_tile_side(M, N, nthreads_max);
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
            qgemm_tile_compute(i0, i1, j0, j1, K,
                               trans_a, trans_b,
                               alpha, beta, a, lda, b, ldb, c, ldc);
        }
    }
}

#undef A_
#undef B_
#undef C_
