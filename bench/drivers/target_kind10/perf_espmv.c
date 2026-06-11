/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for espmv (overlay vs migrated).
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


BLAS_EXTERN void espmv_(const char *, const int *, const R10 *, const R10 *,
    const R10 *, const int *, const R10 *, R10 *, const int *, size_t);
BLAS_EXTERN void espmv_migrated_(const char *, const int *, const R10 *, const R10 *,
    const R10 *, const int *, const R10 *, R10 *, const int *, size_t);

static void run_one(char uplo, int N, int incx, int incy, int iters, int warmup) {
    R10 alpha = R10_FROM(0.7), beta = R10_FROM(0.3);
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absy;
    size_t AP_LEN = (size_t)N * (size_t)(N + 1) / 2;
    R10 *AP = PERF_ALLOC(R10, AP_LEN);
    R10 *X  = PERF_ALLOC(R10, lenx);
    R10 *Y  = PERF_ALLOC(R10, leny);
    R10 *Yi = PERF_ALLOC(R10, leny);
    PERF_FILL_R(R10, AP, AP_LEN, 2);
    PERF_FILL_R(R10, X,  lenx, 3);
    PERF_FILL_R(R10, Yi, leny, 4);
    PERF_RESET(Y, Yi, leny, R10);
    for (int r = 0; r < warmup; ++r) {
        espmv_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1);          PERF_RESET(Y, Yi, leny, R10);
        espmv_migrated_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1); PERF_RESET(Y, Yi, leny, R10);
    }
    double t_subject, t_mg;
    PERF_RESET(Y, Yi, leny, R10);
    PERF_TIME(t_subject, iters, espmv_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1));
    PERF_RESET(Y, Yi, leny, R10);
    PERF_TIME(t_mg,      iters, espmv_migrated_(&uplo, &N, &alpha, AP, X, &incx, &beta, Y, &incy, 1));
    double flops = 2.0 * (double)N * (double)N;
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
    PERF_EMIT("espmv", key, N, iters, flops, t_subject, t_mg);
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
