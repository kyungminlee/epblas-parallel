/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for xhpmv (overlay vs migrated).
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


BLAS_EXTERN void xhpmv_(const char *, const int *, const X16 *, const X16 *,
    const X16 *, const int *, const X16 *, X16 *, const int *, size_t);
BLAS_EXTERN void xhpmv_migrated_(const char *, const int *, const X16 *, const X16 *,
    const X16 *, const int *, const X16 *, X16 *, const int *, size_t);

static void run_one(char uplo, int N, int incx, int incy, int iters, int warmup) {
    X16 alpha = X16_FROM(0.7, 0.0), beta = X16_FROM(0.3, 0.0);
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absy;
    size_t AP_LEN = (size_t)N * (size_t)(N + 1) / 2;
    X16 *AP = PERF_ALLOC(X16, AP_LEN);
    X16 *X  = PERF_ALLOC(X16, lenx);
    X16 *Y  = PERF_ALLOC(X16, leny);
    X16 *Yi = PERF_ALLOC(X16, leny);
    PERF_FILL_C(X16, AP, AP_LEN, 2);
    PERF_FILL_C(X16, X,  lenx, 3);
    PERF_FILL_C(X16, Yi, leny, 4);
    PERF_RESET(Y, Yi, leny, X16);
    for (int r = 0; r < warmup; ++r) {
        xhpmv_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1);          PERF_RESET(Y, Yi, leny, X16);
        xhpmv_migrated_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1); PERF_RESET(Y, Yi, leny, X16);
    }
    double t_subject, t_mg;
    PERF_RESET(Y, Yi, leny, X16);
    PERF_TIME(t_subject, iters, xhpmv_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1));
    PERF_RESET(Y, Yi, leny, X16);
    PERF_TIME(t_mg,      iters, xhpmv_migrated_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1));
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
    PERF_EMIT("xhpmv", key, N, iters, flops, t_subject, t_mg);
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
