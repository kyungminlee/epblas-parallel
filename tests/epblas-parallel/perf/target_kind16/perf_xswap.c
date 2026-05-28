/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for xswap (overlay vs migrated).
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


BLAS_EXTERN void xswap_(const int *, X16 *, const int *, X16 *, const int *);
BLAS_EXTERN void xswap_migrated_(const int *, X16 *, const int *, X16 *, const int *);

static void run_xswap(int N, int iters, int warmup) {
    int one = 1;
    X16 *X = (X16 *)perf_aligned_alloc(64, (size_t)N * sizeof(X16));
    X16 *Y = (X16 *)perf_aligned_alloc(64, (size_t)N * sizeof(X16));
    X16 *Xi = (X16 *)perf_aligned_alloc(64, (size_t)N * sizeof(X16));
    X16 *Yi = (X16 *)perf_aligned_alloc(64, (size_t)N * sizeof(X16));
    for (int i = 0; i < N; ++i) { int s = 0; Xi[i] = X16_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (int i = 0; i < N; ++i) { int s = 1; Yi[i] = X16_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    memcpy(X, Xi, (size_t)N * sizeof(X16));
    memcpy(Y, Yi, (size_t)N * sizeof(X16));
    for (int r = 0; r < warmup; ++r) {
        xswap_(&N, X, &one, Y, &one);
        memcpy(X, Xi, (size_t)N * sizeof(X16));
        memcpy(Y, Yi, (size_t)N * sizeof(X16));
        xswap_migrated_(&N, X, &one, Y, &one);
        memcpy(X, Xi, (size_t)N * sizeof(X16));
        memcpy(Y, Yi, (size_t)N * sizeof(X16));
    }
    /* Per-call kernel-only timing — keep the reset memcpy OUT of the
     * timed window so a single-threaded reset doesn't Amdahl-cap the
     * measured MT scaling at large N. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        xswap_(&N, X, &one, Y, &one);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, (size_t)N * sizeof(X16)); memcpy(Y, Yi, (size_t)N * sizeof(X16));
    }
    double t_subject = t_sum / (iters ? iters : 1);

    t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        xswap_migrated_(&N, X, &one, Y, &one);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, (size_t)N * sizeof(X16)); memcpy(Y, Yi, (size_t)N * sizeof(X16));
    }
    double t_mg = t_sum / (iters ? iters : 1);
    /* Bytes moved per call: copy=2N*sizeof(T), swap=4N*sizeof(T). Report
     * as "flops" for uniform formatting. */
    double flops = 4.0 * (double)N * (double)sizeof(X16);
    perf_emit("xswap", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("xswap", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Y); free(Xi); free(Yi);
}

static const int default_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_xswap(sizes[i], iters, warmup);
    return 0;
}
