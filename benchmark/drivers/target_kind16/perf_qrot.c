/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for qrot (overlay vs migrated).
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


BLAS_EXTERN void qrot_(const int *, Q16 *, const int *, Q16 *, const int *,
    const Q16 *, const Q16 *);
BLAS_EXTERN void qrot_migrated_(const int *, Q16 *, const int *, Q16 *, const int *,
    const Q16 *, const Q16 *);

static void run_one(int N, int iters, int warmup) {
    int one = 1;
    Q16 c_ = Q16_FROM(0.7), s_ = Q16_FROM(0.3);
    Q16 *X  = PERF_ALLOC(Q16, N);
    Q16 *Y  = PERF_ALLOC(Q16, N);
    Q16 *Xi = PERF_ALLOC(Q16, N);
    Q16 *Yi = PERF_ALLOC(Q16, N);
    PERF_FILL_R(Q16, Xi, N, 0);
    PERF_FILL_R(Q16, Yi, N, 1);
    PERF_RESET(X, Xi, N, Q16);
    PERF_RESET(Y, Yi, N, Q16);
    for (int r = 0; r < warmup; ++r) {
        qrot_(&N, X, &one, Y, &one, &c_, &s_);          PERF_RESET(X, Xi, N, Q16); PERF_RESET(Y, Yi, N, Q16);
        qrot_migrated_(&N, X, &one, Y, &one, &c_, &s_); PERF_RESET(X, Xi, N, Q16); PERF_RESET(Y, Yi, N, Q16);
    }
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(X, Xi, N, Q16); PERF_RESET(Y, Yi, N, Q16), qrot_(&N, X, &one, Y, &one, &c_, &s_));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(X, Xi, N, Q16); PERF_RESET(Y, Yi, N, Q16), qrot_migrated_(&N, X, &one, Y, &one, &c_, &s_));
    PERF_EMIT("qrot", "-", N, iters, t_subject, t_mg);
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
    for (int i = 0; i < n; ++i) run_one(sizes[i], iters, warmup);
    return 0;
}
