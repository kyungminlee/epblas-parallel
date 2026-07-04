/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for yaxpy (overlay vs migrated).
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


BLAS_EXTERN void yaxpy_(const int *, const C10 *, const C10 *, const int *, C10 *, const int *);
BLAS_EXTERN void yaxpy_migrated_(const int *, const C10 *, const C10 *, const int *, C10 *, const int *);

static void run_yaxpy(int N, int iters, int warmup) {
    int one = 1;
    C10 alpha = C10_FROM(0.7, 0.0);
    C10 *X  = PERF_ALLOC(C10, N);
    C10 *Y  = PERF_ALLOC(C10, N);
    C10 *Yi = PERF_ALLOC(C10, N);
    PERF_FILL_C(C10, X,  N, 0);
    PERF_FILL_C(C10, Yi, N, 1);
    PERF_RESET(Y, Yi, N, C10);

    for (int r = 0; r < warmup; ++r) {
        yaxpy_(&N, &alpha, X, &one, Y, &one);          PERF_RESET(Y, Yi, N, C10);
        yaxpy_migrated_(&N, &alpha, X, &one, Y, &one); PERF_RESET(Y, Yi, N, C10);
    }

    double t_subject, t_mg;
    PERF_RESET(Y, Yi, N, C10); PERF_TIME(t_subject, iters, yaxpy_(&N, &alpha, X, &one, Y, &one));
    PERF_RESET(Y, Yi, N, C10); PERF_TIME(t_mg,      iters, yaxpy_migrated_(&N, &alpha, X, &one, Y, &one));

    double flops = 8.0 * (double)N;
    PERF_EMIT("yaxpy", "-", N, iters, flops, t_subject, t_mg);
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
    for (int i = 0; i < n; ++i) run_yaxpy(sizes[i], iters, warmup);
    return 0;
}
