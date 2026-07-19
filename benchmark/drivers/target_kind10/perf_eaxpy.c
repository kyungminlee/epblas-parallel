/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for eaxpy (overlay vs migrated).
 * Built per-executable with -ffunction-sections / --gc-sections.
 */
#include "../perf_common.h"

#include <complex.h>
#ifdef __cplusplus
#define BLAS_EXTERN extern "C"
#else
#define BLAS_EXTERN extern
#endif
typedef long double R10;
typedef _Complex long double C10;
#define R10_FROM(d) ((R10)(d))
#define C10_FROM(re, im) ((R10)(re) + 1.0iL * (R10)(im))
static inline R10 Tr_from_d(double d) { return (R10)d; }
static inline C10 Tc_from_d(double d) { return (C10)d; }


BLAS_EXTERN void eaxpy_(const int *, const R10 *, const R10 *, const int *, R10 *, const int *);
BLAS_EXTERN void eaxpy_migrated_(const int *, const R10 *, const R10 *, const int *, R10 *, const int *);

static void run_eaxpy(int N, int iters, int warmup) {
    int one = 1;
    R10 alpha = R10_FROM(0.7);
    R10 *X  = PERF_ALLOC(R10, N);
    R10 *Y  = PERF_ALLOC(R10, N);
    R10 *Yi = PERF_ALLOC(R10, N);
    PERF_FILL_R(R10, X,  N, 0);
    PERF_FILL_R(R10, Yi, N, 1);
    PERF_RESET(Y, Yi, N, R10);

    for (int r = 0; r < warmup; ++r) {
        eaxpy_(&N, &alpha, X, &one, Y, &one);          PERF_RESET(Y, Yi, N, R10);
        eaxpy_migrated_(&N, &alpha, X, &one, Y, &one); PERF_RESET(Y, Yi, N, R10);
    }

    double t_subject, t_mg;
    PERF_RESET(Y, Yi, N, R10); PERF_TIME(t_subject, iters, eaxpy_(&N, &alpha, X, &one, Y, &one));
    PERF_RESET(Y, Yi, N, R10); PERF_TIME(t_mg,      iters, eaxpy_migrated_(&N, &alpha, X, &one, Y, &one));

    PERF_EMIT("eaxpy", "-", N, iters, t_subject, t_mg);
    free(X); free(Y); free(Yi);
}

static const int default_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_eaxpy(sizes[i], iters, warmup);
    return 0;
}
