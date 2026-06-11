/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for etpmv (overlay vs migrated).
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


BLAS_EXTERN void etpmv_(const char *, const char *, const char *, const int *,
    const R10 *, R10 *, const int *, size_t, size_t, size_t);
BLAS_EXTERN void etpmv_migrated_(const char *, const char *, const char *, const int *,
    const R10 *, R10 *, const int *, size_t, size_t, size_t);

static void run_one(char uplo, char trans, char diag, int N, int incx,
                    int iters, int warmup) {
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    size_t AP_LEN = (size_t)N * (size_t)(N + 1) / 2;
    R10 *AP = PERF_ALLOC(R10, AP_LEN);
    R10 *X  = PERF_ALLOC(R10, lenx);
    R10 *Xi = PERF_ALLOC(R10, lenx);
    PERF_FILL_R(R10, AP, AP_LEN, 2);
    /* Force diagonal to ~N for stability of tpsv */
    if (uplo == 'U') {
        size_t off = 0;
        for (int j = 0; j < N; ++j) { AP[off + j] = Tr_from_d((double)(N + 4)); off += (size_t)(j + 1); }
    } else {
        size_t off = 0;
        for (int j = 0; j < N; ++j) { AP[off] = Tr_from_d((double)(N + 4)); off += (size_t)(N - j); }
    }
    PERF_FILL_R(R10, Xi, lenx, 3);
    PERF_RESET(X, Xi, lenx, R10);
    for (int r = 0; r < warmup; ++r) {
        etpmv_(&uplo, &trans, &diag, &N, AP, X, &incx, 1, 1, 1);          PERF_RESET(X, Xi, lenx, R10);
        etpmv_migrated_(&uplo, &trans, &diag, &N, AP, X, &incx, 1, 1, 1); PERF_RESET(X, Xi, lenx, R10);
    }
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(X, Xi, lenx, R10), etpmv_(&uplo, &trans, &diag, &N, AP, X, &incx, 1, 1, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(X, Xi, lenx, R10), etpmv_migrated_(&uplo, &trans, &diag, &N, AP, X, &incx, 1, 1, 1));
    double flops = 1.0 * (double)N * (double)N;
    char key[16];
    if (incx == 1) {
        key[0] = uplo; key[1] = trans; key[2] = diag; key[3] = 0;
    } else {
        snprintf(key, sizeof(key), "%c%c%c/x%d", uplo, trans, diag, incx);
    }
    PERF_EMIT("etpmv", key, N, iters, flops, t_subject, t_mg);
    free(AP); free(X); free(Xi);
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
    const char transes[] = { 'N','T' };
    const char diags[]   = { 'N', 'U' };
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < sizeof(transes); ++t)
    for (size_t d = 0; d < sizeof(diags); ++d) {
        char uplo = (u == 0) ? 'U' : 'L';
        char trans = transes[t];
        char diag = diags[d];
        for (int xi = 0; xi < n_incx; ++xi) {
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int i = 0; i < n; ++i) run_one(uplo, trans, diag, sizes[i], incx, iters, warmup);
        }
    }
    return 0;
}
