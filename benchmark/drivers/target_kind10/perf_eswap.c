/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for eswap (overlay vs migrated).
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


BLAS_EXTERN void eswap_(const int *, R10 *, const int *, R10 *, const int *);
BLAS_EXTERN void eswap_migrated_(const int *, R10 *, const int *, R10 *, const int *);

static void run_eswap(int N, int iters, int warmup) {
    int one = 1;
    R10 *X  = PERF_ALLOC(R10, N);
    R10 *Y  = PERF_ALLOC(R10, N);
    R10 *Xi = PERF_ALLOC(R10, N);
    R10 *Yi = PERF_ALLOC(R10, N);
    PERF_FILL_R(R10, Xi, N, 0);
    PERF_FILL_R(R10, Yi, N, 1);
    PERF_RESET(X, Xi, N, R10);
    PERF_RESET(Y, Yi, N, R10);
    for (int r = 0; r < warmup; ++r) {
        eswap_(&N, X, &one, Y, &one);          PERF_RESET(X, Xi, N, R10); PERF_RESET(Y, Yi, N, R10);
        eswap_migrated_(&N, X, &one, Y, &one); PERF_RESET(X, Xi, N, R10); PERF_RESET(Y, Yi, N, R10);
    }
    /* Per-call timing (reset out of the timed window — see PERF_TIME_PER_CALL). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(X, Xi, N, R10); PERF_RESET(Y, Yi, N, R10), eswap_(&N, X, &one, Y, &one));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(X, Xi, N, R10); PERF_RESET(Y, Yi, N, R10), eswap_migrated_(&N, X, &one, Y, &one));
    /* Bytes moved per call: copy=2N*sizeof(T), swap=4N*sizeof(T). Report
     * as "flops" for uniform formatting. */
    double flops = 4.0 * (double)N * (double)sizeof(R10);
    PERF_EMIT("eswap", "-", N, iters, flops, t_subject, t_mg);
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
    for (int i = 0; i < n; ++i) run_eswap(sizes[i], iters, warmup);
    return 0;
}
