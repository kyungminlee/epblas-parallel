/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for xher (overlay vs migrated).
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


BLAS_EXTERN void xher_(const char *, const int *, const Q16 *,
    const X16 *, const int *, X16 *, const int *, size_t);
BLAS_EXTERN void xher_migrated_(const char *, const int *, const Q16 *,
    const X16 *, const int *, X16 *, const int *, size_t);

static void run_one(char uplo, int N, int incx, int iters, int warmup) {
    Q16 alpha = Q16_FROM(0.7);
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    X16 *A  = (X16 *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof(X16));
    X16 *Ai = (X16 *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof(X16));
    X16 *X  = (X16 *)perf_aligned_alloc(64, lenx * sizeof(X16));
    for (size_t i = 0; i < (size_t)N*N; ++i) { int s = 2; Ai[i] = X16_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < lenx; ++i)      { int s = 3; X[i] = X16_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    memcpy(A, Ai, (size_t)N * (size_t)N * sizeof(X16));
    for (int r = 0; r < warmup; ++r) {
        xher_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof(X16));
        xher_migrated_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof(X16));
    }
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        xher_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof(X16));
    }
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        xher_migrated_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof(X16));
    }
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = 4.0 * (double)N * (double)N;
    char key[16];
    if (incx == 1) {
        key[0] = uplo; key[1] = 0;
    } else {
        snprintf(key, sizeof(key), "%c/x%d", uplo, incx);
    }
    perf_emit("xher", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("xher", key, N, iters, flops, t_subject, t_mg);
    free(A); free(Ai); free(X);
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
