/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for erotmg (overlay vs migrated).
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


BLAS_EXTERN void erotmg_(R10 *, R10 *, R10 *, const R10 *, R10 *);
BLAS_EXTERN void erotmg_migrated_(R10 *, R10 *, R10 *, const R10 *, R10 *);

static void run_one(int iters, int warmup) {
    R10 D1 = R10_FROM(0.7), D2 = R10_FROM(0.7), X1 = R10_FROM(0.7), Y1 = R10_FROM(0.7);
    R10 PARAM[5];
    for (int r = 0; r < warmup; ++r) {
        R10 d1 = D1, d2 = D2, x1 = X1;
        erotmg_(&d1, &d2, &x1, &Y1, PARAM);
        d1 = D1; d2 = D2; x1 = X1;
        erotmg_migrated_(&d1, &d2, &x1, &Y1, PARAM);
    }
    double t_subject, t_mg;
    PERF_TIME(t_subject, iters, R10 d1 = D1, d2 = D2, x1 = X1; erotmg_(&d1, &d2, &x1, &Y1, PARAM));
    PERF_TIME(t_mg,      iters, R10 d1 = D1, d2 = D2, x1 = X1; erotmg_migrated_(&d1, &d2, &x1, &Y1, PARAM));
    PERF_EMIT("erotmg", "-", iters, iters, t_subject, t_mg);
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}
