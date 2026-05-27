/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for qrotmg (overlay vs migrated).
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


BLAS_EXTERN void qrotmg_(Q16 *, Q16 *, Q16 *, const Q16 *, Q16 *);
BLAS_EXTERN void qrotmg_migrated_(Q16 *, Q16 *, Q16 *, const Q16 *, Q16 *);

static void run_one(int iters, int warmup) {
    Q16 D1 = Q16_FROM(0.7), D2 = Q16_FROM(0.7), X1 = Q16_FROM(0.7), Y1 = Q16_FROM(0.7);
    Q16 PARAM[5];
    for (int r = 0; r < warmup; ++r) {
        Q16 d1 = D1, d2 = D2, x1 = X1;
        qrotmg_(&d1, &d2, &x1, &Y1, PARAM);
        d1 = D1; d2 = D2; x1 = X1;
        qrotmg_migrated_(&d1, &d2, &x1, &Y1, PARAM);
    }
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {
        Q16 d1 = D1, d2 = D2, x1 = X1;
        qrotmg_(&d1, &d2, &x1, &Y1, PARAM);
    }
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {
        Q16 d1 = D1, d2 = D2, x1 = X1;
        qrotmg_migrated_(&d1, &d2, &x1, &Y1, PARAM);
    }
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = 20.0;
    perf_emit("qrotmg", "-", iters, iters, flops, t_subject, t_mg);
    perf_emit_json("qrotmg", "-", iters, iters, flops, t_subject, t_mg);
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}
