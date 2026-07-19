/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for ecabs1 (overlay vs migrated).
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


BLAS_EXTERN R10 ecabs1_(const C10 *);
BLAS_EXTERN R10 ecabs1_migrated_(const C10 *);

static void run_one(int iters, int warmup) {
    C10 Z = C10_FROM(0.7, 0.0);
    R10 acc = Tr_from_d(0.0);
    for (int r = 0; r < warmup; ++r) {
        acc += ecabs1_(&Z);
        acc += ecabs1_migrated_(&Z);
    }
    double t_subject, t_mg;
    PERF_TIME(t_subject, iters, acc += ecabs1_(&Z));
    PERF_TIME(t_mg,      iters, acc += ecabs1_migrated_(&Z));
    /* per-call proxy: 2 abs + 1 add */
    PERF_EMIT("ecabs1", "-", iters, iters, t_subject, t_mg);
    if ((double)(*((double*)&acc)) == -123e30) return;
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}
