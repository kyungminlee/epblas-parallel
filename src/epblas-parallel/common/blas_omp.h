/* blas_omp.h — shared OpenMP runtime helpers for the parallel BLAS overlay.
 *
 * Returns omp_get_max_threads() directly (no caching) so callers always
 * observe the current `nthreads-var` ICV, including changes from
 * omp_set_num_threads() between calls.
 *
 * A previous version cached the first call's result to avoid a libgomp
 * ICV lookup on every dispatch. That cache locked itself to the first
 * caller's omp_set_num_threads() value — fine in production where
 * OMP_NUM_THREADS is set once at startup, but it silently disabled
 * parallelism in perf benches that alternate omp_set_num_threads(1)
 * and omp_set_num_threads(N). At libquadmath rate the ICV-lookup cost
 * is well below the per-call BLAS overhead, so the cache wasn't pulling
 * its weight anyway.
 */
#ifndef EPBLAS_PARALLEL_BLAS_OMP_H
#define EPBLAS_PARALLEL_BLAS_OMP_H

#ifdef __cplusplus
extern "C" {
#endif

#ifdef _OPENMP
#include <omp.h>
static inline int blas_omp_max_threads(void) {
    int v = omp_get_max_threads();
    return (v < 1) ? 1 : v;
}
#else
static inline int blas_omp_max_threads(void) { return 1; }
#endif

/* Whether threading is worth attempting at all: more than one thread is
 * available. The canonical spelling of the `blas_omp_max_threads() > 1` capability
 * check that gates every threaded dispatch — wrap it once so the "can we thread"
 * policy lives in one place (re-entrant helpers still pair it with their own
 * !omp_in_parallel() guard, which stays explicit at the call site). */
static inline int blas_omp_available(void) { return blas_omp_max_threads() > 1; }

#include <stddef.h>

/* Balanced 1-D partition bound: the low index of chunk `idx` of `nparts` over
 * [0, total). Front-loaded-remainder scheme — every chunk gets base = total/nparts
 * elements and the first rem = total%nparts chunks get one extra, so chunk sizes
 * differ by at most one and the heavier chunks are threads 0..rem-1:
 *     lo(idx) = idx*base + min(idx, rem).
 * Overflow-safe in pure 64-bit with no wide type or libgcc soft-division: idx*base
 * <= nparts*base <= total always, so the product cannot exceed ptrdiff_t (the
 * `(long long)total*idx` floor idiom this replaced WAS a real ILP64 overflow once
 * total > ~2^63/nparts). total >= 0, 0 <= idx <= nparts, nparts >= 1; result is in
 * [0,total]. Computed once per thread per dispatch, never per element. This is the
 * single partition primitive — every unit-granularity range split routes here so
 * the scheme (and its overflow guard) lives in one place. */
static inline ptrdiff_t blas_part_bound(ptrdiff_t total, ptrdiff_t idx,
                                        ptrdiff_t nparts) {
    const ptrdiff_t base = total / nparts;
    const ptrdiff_t rem  = total % nparts;
    return idx * base + (idx < rem ? idx : rem);
}

/* Outer panel width for a threaded L3 panel loop. The cache-tuned serial
 * block `nb` leaves too few panels to feed the team at small problem sizes
 * (axis=64, nb=32 -> 2 panels, so 2 of 4 threads sit idle). When the default
 * step already yields enough panels (>= nt*ppt) the cache-optimal `nb` is
 * returned unchanged, so larger N is never disturbed. Otherwise the width is
 * shrunk to deliver about nt*ppt panels, floored at 8 so each panel still
 * amortizes its trailing-GEMM overhead, and never wider than nb. Use ppt=1
 * for equal-work (rectangular) panels balanced by schedule(static); a larger
 * ppt for triangular work that needs finer granularity for dynamic balance. */
static inline ptrdiff_t blas_omp_panel_width(ptrdiff_t axis, int nt,
                                             ptrdiff_t nb, int ppt) {
    if (nt <= 1) return nb;
    const ptrdiff_t want = (ptrdiff_t)nt * ppt;
    if ((axis + nb - 1) / nb >= want) return nb;   /* nb already feeds the team */
    ptrdiff_t pw = axis / want;
    if (pw < 8)  pw = 8;
    if (pw > nb) pw = nb;
    return pw;
}

#ifdef __cplusplus
}
#endif

#endif /* EPBLAS_PARALLEL_BLAS_OMP_H */
