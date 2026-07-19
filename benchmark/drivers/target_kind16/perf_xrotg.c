/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for xrotg (overlay vs migrated).
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


BLAS_EXTERN void xrotg_(X16 *, X16 *, Q16 *, X16 *);
BLAS_EXTERN void xrotg_migrated_(X16 *, X16 *, Q16 *, X16 *);

static void run_one(int iters, int warmup) {
    X16 A = X16_FROM(0.7, 0.0), B = X16_FROM(0.7, 0.0);
    Q16 C = Tr_from_d(0.0);
    X16 S = Tc_from_d(0.0);
    /* per call: regenerate fresh A, B inputs */
    for (int r = 0; r < warmup; ++r) {
        X16 a = A, b = B; xrotg_(&a, &b, &C, &S);
        a = A; b = B; xrotg_migrated_(&a, &b, &C, &S);
    }
    double t_subject, t_mg;
    PERF_TIME(t_subject, iters, X16 a = A, b = B; xrotg_(&a, &b, &C, &S));
    PERF_TIME(t_mg,      iters, X16 a = A, b = B; xrotg_migrated_(&a, &b, &C, &S));
    PERF_EMIT("xrotg", "-", iters, iters, t_subject, t_mg);
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}
