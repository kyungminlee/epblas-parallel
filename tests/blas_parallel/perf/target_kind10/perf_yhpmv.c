/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for yhpmv (overlay vs migrated).
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


BLAS_EXTERN void yhpmv_(const char *, const int *, const C10 *, const C10 *,
    const C10 *, const int *, const C10 *, C10 *, const int *, size_t);
BLAS_EXTERN void yhpmv_migrated_(const char *, const int *, const C10 *, const C10 *,
    const C10 *, const int *, const C10 *, C10 *, const int *, size_t);

static void run_one(char uplo, int N, int incx, int incy, int iters, int warmup) {
    C10 alpha = C10_FROM(0.7, 0.0), beta = C10_FROM(0.3, 0.0);
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absy;
    size_t AP_LEN = (size_t)N * (size_t)(N + 1) / 2;
    C10 *AP = (C10 *)perf_aligned_alloc(64, AP_LEN * sizeof(C10));
    C10 *X  = (C10 *)perf_aligned_alloc(64, lenx * sizeof(C10));
    C10 *Y  = (C10 *)perf_aligned_alloc(64, leny * sizeof(C10));
    C10 *Yi = (C10 *)perf_aligned_alloc(64, leny * sizeof(C10));
    for (size_t i = 0; i < AP_LEN; ++i) { int s = 2; AP[i] = C10_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < lenx; ++i)   { int s = 3; X[i]  = C10_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < leny; ++i)   { int s = 4; Yi[i] = C10_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    memcpy(Y, Yi, leny * sizeof(C10));
    for (int r = 0; r < warmup; ++r) {
        yhpmv_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof(C10));
        yhpmv_migrated_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof(C10));
    }
    memcpy(Y, Yi, leny * sizeof(C10));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) yhpmv_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(Y, Yi, leny * sizeof(C10));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) yhpmv_migrated_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = 8.0 * (double)N * (double)N;
    char key[24];
    if (incx == 1 && incy == 1) {
        key[0] = uplo; key[1] = 0;
    } else if (incy == 1) {
        snprintf(key, sizeof(key), "%c/x%d", uplo, incx);
    } else if (incx == 1) {
        snprintf(key, sizeof(key), "%c/y%d", uplo, incy);
    } else {
        snprintf(key, sizeof(key), "%c/x%d/y%d", uplo, incx, incy);
    }
    perf_emit("yhpmv", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("yhpmv", key, N, iters, flops, t_subject, t_mg);
    free(AP); free(X); free(Y); free(Yi);
}

static const int default_sizes[] = {128, 256, 512, 1024};
static const int default_incxs[] = {1, 2, -1};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    int incxs[8], incys[8];
    int n_incx = perf_parse_int_list("BLAS_PERF_INCX", default_incxs,
        (int)(sizeof(default_incxs)/sizeof(default_incxs[0])), incxs, 8);
    int n_incy = perf_parse_int_list("BLAS_PERF_INCY", incxs, n_incx, incys, 8);
    perf_print_header();
    for (size_t u = 0; u < 2; ++u) {
        char uplo = (u == 0) ? 'U' : 'L';
        for (int xi = 0; xi < n_incx; ++xi) {
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int yi = 0; yi < n_incy; ++yi) {
                int incy = incys[yi]; if (incy == 0) continue;
                for (int i = 0; i < n; ++i) run_one(uplo, sizes[i], incx, incy, iters, warmup);
            }
        }
    }
    return 0;
}
