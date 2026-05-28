/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for erotg (overlay vs migrated).
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


BLAS_EXTERN void erotg_(R10 *, R10 *, R10 *, R10 *);
BLAS_EXTERN void erotg_migrated_(R10 *, R10 *, R10 *, R10 *);

static void run_one(int iters, int warmup) {
    R10 A = R10_FROM(0.7), B = R10_FROM(0.7);
    R10 C = Tr_from_d(0.0);
    R10 S = Tr_from_d(0.0);
    /* per call: regenerate fresh A, B inputs */
    for (int r = 0; r < warmup; ++r) {
        R10 a = A, b = B; erotg_(&a, &b, &C, &S);
        a = A; b = B; erotg_migrated_(&a, &b, &C, &S);
    }
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {
        R10 a = A, b = B; erotg_(&a, &b, &C, &S);
    }
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {
        R10 a = A, b = B; erotg_migrated_(&a, &b, &C, &S);
    }
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    /* report time per call as "flops" abuse: per-call flop count ~10. */
    double flops = 10.0;
    perf_emit("erotg", "-", iters, iters, flops, t_subject, t_mg);
    perf_emit_json("erotg", "-", iters, iters, flops, t_subject, t_mg);
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}
