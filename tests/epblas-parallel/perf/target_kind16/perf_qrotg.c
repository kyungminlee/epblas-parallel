/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for qrotg (overlay vs migrated).
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


BLAS_EXTERN void qrotg_(Q16 *, Q16 *, Q16 *, Q16 *);
BLAS_EXTERN void qrotg_migrated_(Q16 *, Q16 *, Q16 *, Q16 *);

static void run_one(int iters, int warmup) {
    Q16 A = Q16_FROM(0.7), B = Q16_FROM(0.7);
    Q16 C = Tr_from_d(0.0);
    Q16 S = Tr_from_d(0.0);
    /* per call: regenerate fresh A, B inputs */
    for (int r = 0; r < warmup; ++r) {
        Q16 a = A, b = B; qrotg_(&a, &b, &C, &S);
        a = A; b = B; qrotg_migrated_(&a, &b, &C, &S);
    }
    double t_subject, t_mg;
    PERF_TIME(t_subject, iters, Q16 a = A, b = B; qrotg_(&a, &b, &C, &S));
    PERF_TIME(t_mg,      iters, Q16 a = A, b = B; qrotg_migrated_(&a, &b, &C, &S));
    /* report time per call as "flops" abuse: per-call flop count ~10. */
    double flops = 10.0;
    PERF_EMIT("qrotg", "-", iters, iters, flops, t_subject, t_mg);
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}
