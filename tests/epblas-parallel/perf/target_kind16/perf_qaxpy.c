/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for qaxpy (overlay vs migrated).
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


BLAS_EXTERN void qaxpy_(const int *, const Q16 *, const Q16 *, const int *, Q16 *, const int *);
BLAS_EXTERN void qaxpy_migrated_(const int *, const Q16 *, const Q16 *, const int *, Q16 *, const int *);

static void run_qaxpy(int N, int iters, int warmup) {
    int one = 1;
    Q16 alpha = Q16_FROM(0.7);
    Q16 *X = (Q16 *)perf_aligned_alloc(64, (size_t)N * sizeof(Q16));
    Q16 *Y = (Q16 *)perf_aligned_alloc(64, (size_t)N * sizeof(Q16));
    Q16 *Yi = (Q16 *)perf_aligned_alloc(64, (size_t)N * sizeof(Q16));
    for (int s = 0; s < 1; ++s) {}
    for (int i = 0; i < N; ++i) { int s = 0; X[i] = Q16_FROM(perf_fill_double(i, s)); }
    for (int i = 0; i < N; ++i) { int s = 1; Yi[i] = Q16_FROM(perf_fill_double(i, s)); }
    memcpy(Y, Yi, (size_t)N * sizeof(Q16));

    for (int r = 0; r < warmup; ++r) {
        qaxpy_(&N, &alpha, X, &one, Y, &one);
        memcpy(Y, Yi, (size_t)N * sizeof(Q16));
        qaxpy_migrated_(&N, &alpha, X, &one, Y, &one);
        memcpy(Y, Yi, (size_t)N * sizeof(Q16));
    }

    memcpy(Y, Yi, (size_t)N * sizeof(Q16));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) qaxpy_(&N, &alpha, X, &one, Y, &one);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);

    memcpy(Y, Yi, (size_t)N * sizeof(Q16));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) qaxpy_migrated_(&N, &alpha, X, &one, Y, &one);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);

    double flops = 2.0 * (double)N;
    perf_emit("qaxpy", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("qaxpy", "-", N, iters, flops, t_subject, t_mg);
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
    for (int i = 0; i < n; ++i) run_qaxpy(sizes[i], iters, warmup);
    return 0;
}
