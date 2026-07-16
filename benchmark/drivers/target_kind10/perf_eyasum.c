/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for eyasum (overlay vs migrated).
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


BLAS_EXTERN R10 eyasum_(const int *, const C10 *, const int *);
BLAS_EXTERN R10 eyasum_migrated_(const int *, const C10 *, const int *);

static void run_one(int N, int iters, int warmup) {
    int one = 1;
    R10 r;
    C10 *X = PERF_ALLOC(C10, N);
    PERF_FILL_C(C10, X, N, 0);
    for (int r2 = 0; r2 < warmup; ++r2) {
        r = eyasum_(&N, X, &one);
        r = eyasum_migrated_(&N, X, &one);
    }
    double t_subject, t_mg;
    PERF_TIME(t_subject, iters, r = eyasum_(&N, X, &one));
    PERF_TIME(t_mg,      iters, r = eyasum_migrated_(&N, X, &one));
    double flops = 2.0 * (double)N;
    PERF_EMIT("eyasum", "-", N, iters, flops, t_subject, t_mg);
    if ((double)(*((double*)&r)) == -123e30) { free(X); return; }
    free(X);
}

static const int default_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_one(sizes[i], iters, warmup);
    return 0;
}
