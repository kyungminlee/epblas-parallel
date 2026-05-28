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

#ifdef __cplusplus
}
#endif

#endif /* EPBLAS_PARALLEL_BLAS_OMP_H */
