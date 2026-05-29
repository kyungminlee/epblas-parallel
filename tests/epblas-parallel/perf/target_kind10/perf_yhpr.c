/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for yhpr (overlay vs migrated).
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


BLAS_EXTERN void yhpr_(const char *, const int *, const R10 *,
    const C10 *, const int *, C10 *, size_t);
BLAS_EXTERN void yhpr_migrated_(const char *, const int *, const R10 *,
    const C10 *, const int *, C10 *, size_t);

static void run_one(char uplo, int N, int incx, int iters, int warmup) {
    R10 alpha = R10_FROM(0.7);
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    size_t AP_LEN = (size_t)N * (size_t)(N + 1) / 2;
    C10 *AP  = PERF_ALLOC(C10, AP_LEN);
    C10 *APi = PERF_ALLOC(C10, AP_LEN);
    C10 *X   = PERF_ALLOC(C10, lenx);
    PERF_FILL_C(C10, APi, AP_LEN, 2);
    PERF_FILL_C(C10, X,   lenx, 3);
    PERF_RESET(AP, APi, AP_LEN, C10);
    for (int r = 0; r < warmup; ++r) {
        yhpr_(&uplo, &N, &alpha, X, &incx, AP, 1);          PERF_RESET(AP, APi, AP_LEN, C10);
        yhpr_migrated_(&uplo, &N, &alpha, X, &incx, AP, 1); PERF_RESET(AP, APi, AP_LEN, C10);
    }
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(AP, APi, AP_LEN, C10), yhpr_(&uplo, &N, &alpha, X, &incx, AP, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(AP, APi, AP_LEN, C10), yhpr_migrated_(&uplo, &N, &alpha, X, &incx, AP, 1));
    double flops = 4.0 * (double)N * (double)N;
    char key[16];
    if (incx == 1) {
        key[0] = uplo; key[1] = 0;
    } else {
        snprintf(key, sizeof(key), "%c/x%d", uplo, incx);
    }
    PERF_EMIT("yhpr", key, N, iters, flops, t_subject, t_mg);
    free(AP); free(APi); free(X);
}

static const int default_sizes[] = {128, 256, 512, 1024};
static const int default_incxs[] = {1, 2, -1};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    int incxs[8];
    int n_incx = perf_parse_int_list("BLAS_PERF_INCX", default_incxs,
        (int)(sizeof(default_incxs)/sizeof(default_incxs[0])), incxs, 8);
    perf_print_header();
    for (size_t u = 0; u < 2; ++u) {
        char uplo = (u == 0) ? 'U' : 'L';
        for (int xi = 0; xi < n_incx; ++xi) {
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int i = 0; i < n; ++i) run_one(uplo, sizes[i], incx, iters, warmup);
        }
    }
    return 0;
}
