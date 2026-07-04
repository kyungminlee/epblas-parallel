/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for yrotg (overlay vs migrated).
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


BLAS_EXTERN void yrotg_(C10 *, C10 *, R10 *, C10 *);
BLAS_EXTERN void yrotg_migrated_(C10 *, C10 *, R10 *, C10 *);

static void run_one(int iters, int warmup) {
    C10 A = C10_FROM(0.7, 0.0), B = C10_FROM(0.7, 0.0);
    R10 C = Tr_from_d(0.0);
    C10 S = Tc_from_d(0.0);
    /* per call: regenerate fresh A, B inputs */
    for (int r = 0; r < warmup; ++r) {
        C10 a = A, b = B; yrotg_(&a, &b, &C, &S);
        a = A; b = B; yrotg_migrated_(&a, &b, &C, &S);
    }
    double t_subject, t_mg;
    PERF_TIME(t_subject, iters, C10 a = A, b = B; yrotg_(&a, &b, &C, &S));
    PERF_TIME(t_mg,      iters, C10 a = A, b = B; yrotg_migrated_(&a, &b, &C, &S));
    /* report time per call as "flops" abuse: per-call flop count ~10. */
    double flops = 10.0;
    PERF_EMIT("yrotg", "-", iters, iters, flops, t_subject, t_mg);
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}
