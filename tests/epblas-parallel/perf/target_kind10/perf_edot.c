/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for edot (overlay vs migrated).
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


BLAS_EXTERN R10 edot_(const int *, const R10 *, const int *, const R10 *, const int *);
BLAS_EXTERN R10 edot_migrated_(const int *, const R10 *, const int *, const R10 *, const int *);

static volatile unsigned long perf_sink = 0;

static inline void sink_T(const R10 *p) {
    /* extract first 8 bytes of T into volatile sink */
    unsigned long w;
    memcpy(&w, (const void *)p, sizeof(w));
    perf_sink ^= w;
}

static void run_one(int N, int iters, int warmup) {
    int one = 1;
    R10 r;
    R10 *X = (R10 *)perf_aligned_alloc(64, (size_t)N * sizeof(R10));
    R10 *Y = (R10 *)perf_aligned_alloc(64, (size_t)N * sizeof(R10));
    for (int i = 0; i < N; ++i) { int s = 0; X[i] = R10_FROM(perf_fill_double(i, s)); }
    for (int i = 0; i < N; ++i) { int s = 1; Y[i] = R10_FROM(perf_fill_double(i, s)); }
    for (int r2 = 0; r2 < warmup; ++r2) {
        r = edot_(&N, X, &one, Y, &one); sink_T(&r);
        r = edot_migrated_(&N, X, &one, Y, &one); sink_T(&r);
    }
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {
        r = edot_(&N, X, &one, Y, &one);
        sink_T(&r);
    }
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);

    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {
        r = edot_migrated_(&N, X, &one, Y, &one);
        sink_T(&r);
    }
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);

    double flops = 2.0 * (double)N;
    perf_emit("edot", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("edot", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Y);
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
