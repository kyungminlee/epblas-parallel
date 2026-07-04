/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for xaxpy (overlay vs migrated).
 * Built per-executable with -ffunction-sections / --gc-sections.
 */
#include "../perf_common.h"

#include <quadmath.h>
#ifdef __cplusplus
#define BLAS_EXTERN extern "C"
#else
#define BLAS_EXTERN extern
#endif
typedef __float128 Q16;
typedef _Complex float __attribute__((mode(TC))) X16;
#define Q16_FROM(d) ((Q16)(double)(d))
#define X16_FROM(re, im) ((X16)((Q16)(double)(re) + 1.0i * (Q16)(double)(im)))
static inline Q16 Tr_from_d(double d) { return (Q16)d; }
static inline X16 Tc_from_d(double d) { return (X16)((Q16)d); }


BLAS_EXTERN void xaxpy_(const int *, const X16 *, const X16 *, const int *, X16 *, const int *);
BLAS_EXTERN void xaxpy_migrated_(const int *, const X16 *, const X16 *, const int *, X16 *, const int *);

static void run_xaxpy(int N, int iters, int warmup) {
    int one = 1;
    X16 alpha = X16_FROM(0.7, 0.0);
    X16 *X  = PERF_ALLOC(X16, N);
    X16 *Y  = PERF_ALLOC(X16, N);
    X16 *Yi = PERF_ALLOC(X16, N);
    PERF_FILL_C(X16, X,  N, 0);
    PERF_FILL_C(X16, Yi, N, 1);
    PERF_RESET(Y, Yi, N, X16);

    for (int r = 0; r < warmup; ++r) {
        xaxpy_(&N, &alpha, X, &one, Y, &one);          PERF_RESET(Y, Yi, N, X16);
        xaxpy_migrated_(&N, &alpha, X, &one, Y, &one); PERF_RESET(Y, Yi, N, X16);
    }

    double t_subject, t_mg;
    PERF_RESET(Y, Yi, N, X16); PERF_TIME(t_subject, iters, xaxpy_(&N, &alpha, X, &one, Y, &one));
    PERF_RESET(Y, Yi, N, X16); PERF_TIME(t_mg,      iters, xaxpy_migrated_(&N, &alpha, X, &one, Y, &one));

    double flops = 8.0 * (double)N;
    PERF_EMIT("xaxpy", "-", N, iters, flops, t_subject, t_mg);
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
    for (int i = 0; i < n; ++i) run_xaxpy(sizes[i], iters, warmup);
    return 0;
}
