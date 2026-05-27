/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for espr2 (overlay vs migrated).
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

BLAS_EXTERN void espr2_(const char *, const int *, const R10 *,
    const R10 *, const int *, const R10 *, const int *,
    R10 *, size_t);
BLAS_EXTERN void espr2_migrated_(const char *, const int *, const R10 *,
    const R10 *, const int *, const R10 *, const int *,
    R10 *, size_t);

static void run_one(char uplo, int N, int incx, int incy, int iters, int warmup) {
    R10 alpha = R10_FROM(0.7);
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absy;
    size_t AP_LEN = (size_t)N * (size_t)(N + 1) / 2;
    R10 *AP  = (R10 *)perf_aligned_alloc(64, AP_LEN * sizeof(R10));
    R10 *APi = (R10 *)perf_aligned_alloc(64, AP_LEN * sizeof(R10));
    R10 *X   = (R10 *)perf_aligned_alloc(64, lenx * sizeof(R10));
    R10 *Y   = (R10 *)perf_aligned_alloc(64, leny * sizeof(R10));
    for (size_t i = 0; i < AP_LEN; ++i) { int s = 2; APi[i] = R10_FROM(perf_fill_double(i, s)); }
    for (size_t i = 0; i < lenx; ++i)    { int s = 3; X[i]   = R10_FROM(perf_fill_double(i, s)); }
    for (size_t i = 0; i < leny; ++i)    { int s = 4; Y[i]   = R10_FROM(perf_fill_double(i, s)); }
    memcpy(AP, APi, AP_LEN * sizeof(R10));
    for (int r = 0; r < warmup; ++r) {
        espr2_(&uplo, &N, &alpha, X, &incx, Y, &incy, AP, 1);
        memcpy(AP, APi, AP_LEN * sizeof(R10));
        espr2_migrated_(&uplo, &N, &alpha, X, &incx, Y, &incy, AP, 1);
        memcpy(AP, APi, AP_LEN * sizeof(R10));
    }
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        espr2_(&uplo, &N, &alpha, X, &incx, Y, &incy, AP, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(AP, APi, AP_LEN * sizeof(R10));
    }
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        espr2_migrated_(&uplo, &N, &alpha, X, &incx, Y, &incy, AP, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(AP, APi, AP_LEN * sizeof(R10));
    }
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = 2.0 * (double)N * (double)N;
    char key[20];
    if (incx == 1 && incy == 1) {
        key[0] = uplo; key[1] = 0;
    } else {
        snprintf(key, sizeof(key), "%c/x%d/y%d", uplo, incx, incy);
    }
    perf_emit("espr2", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("espr2", key, N, iters, flops, t_subject, t_mg);
    free(AP); free(APi); free(X); free(Y);
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
