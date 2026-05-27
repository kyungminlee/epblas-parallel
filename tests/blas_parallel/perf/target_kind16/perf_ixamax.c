/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for ixamax (overlay vs migrated).
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


BLAS_EXTERN int ixamax_(const int *, const X16 *, const int *);
BLAS_EXTERN int ixamax_migrated_(const int *, const X16 *, const int *);

static void run_one(int N, int iters, int warmup) {
    int one = 1, r = 0;
    X16 *X = (X16 *)perf_aligned_alloc(64, (size_t)N * sizeof(X16));
    for (int i = 0; i < N; ++i) { int s = 0; X[i] = X16_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (int r2 = 0; r2 < warmup; ++r2) {
        r ^= ixamax_(&N, X, &one);
        r ^= ixamax_migrated_(&N, X, &one);
    }
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) r ^= ixamax_(&N, X, &one);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) r ^= ixamax_migrated_(&N, X, &one);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = 2.0 * (double)N;
    perf_emit("ixamax", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("ixamax", "-", N, iters, flops, t_subject, t_mg);
    if (r == -123) { free(X); return; }
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
