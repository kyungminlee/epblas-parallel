/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for xscal (overlay vs migrated).
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


BLAS_EXTERN void xscal_(const int *, const X16 *, X16 *, const int *);
BLAS_EXTERN void xscal_migrated_(const int *, const X16 *, X16 *, const int *);

static void run_xscal(int N, int iters, int warmup) {
    int one = 1;
    X16 alpha = X16_FROM(0.7, 0.0);
    X16 *X  = PERF_ALLOC(X16, N);
    X16 *Xi = PERF_ALLOC(X16, N);
    PERF_FILL_C(X16, Xi, N, 0);
    PERF_RESET(X, Xi, N, X16);
    for (int r = 0; r < warmup; ++r) {
        xscal_(&N, &alpha, X, &one);          PERF_RESET(X, Xi, N, X16);
        xscal_migrated_(&N, &alpha, X, &one); PERF_RESET(X, Xi, N, X16);
    }
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(X, Xi, N, X16), xscal_(&N, &alpha, X, &one));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(X, Xi, N, X16), xscal_migrated_(&N, &alpha, X, &one));
    double flops = 6.0 * (double)N;
    PERF_EMIT("xscal", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Xi);
}

static const int default_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_xscal(sizes[i], iters, warmup);
    return 0;
}
