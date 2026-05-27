/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for qger (overlay vs migrated).
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


BLAS_EXTERN void qger_(const int *, const int *, const Q16 *,
    const Q16 *, const int *, const Q16 *, const int *, Q16 *, const int *);
BLAS_EXTERN void qger_migrated_(const int *, const int *, const Q16 *,
    const Q16 *, const int *, const Q16 *, const int *, Q16 *, const int *);

static void run_one(int M, int N, int incx, int incy, int iters, int warmup) {
    Q16 alpha = Q16_FROM(0.7);
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(M - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absy;
    Q16 *A  = (Q16 *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(Q16));
    Q16 *Ai = (Q16 *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(Q16));
    Q16 *X  = (Q16 *)perf_aligned_alloc(64, lenx * sizeof(Q16));
    Q16 *Y  = (Q16 *)perf_aligned_alloc(64, leny * sizeof(Q16));
    for (size_t i = 0; i < (size_t)M*N; ++i) { int s = 2; Ai[i] = Q16_FROM(perf_fill_double(i, s)); }
    for (size_t i = 0; i < lenx; ++i)       { int s = 3; X[i] = Q16_FROM(perf_fill_double(i, s)); }
    for (size_t i = 0; i < leny; ++i)       { int s = 4; Y[i] = Q16_FROM(perf_fill_double(i, s)); }
    memcpy(A, Ai, (size_t)M * (size_t)N * sizeof(Q16));

    for (int r = 0; r < warmup; ++r) {
        qger_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M);
        memcpy(A, Ai, (size_t)M * (size_t)N * sizeof(Q16));
        qger_migrated_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M);
        memcpy(A, Ai, (size_t)M * (size_t)N * sizeof(Q16));
    }
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        qger_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(A, Ai, (size_t)M * (size_t)N * sizeof(Q16));
    }
    double t_subject = t_sum / (iters ? iters : 1);

    t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        qger_migrated_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(A, Ai, (size_t)M * (size_t)N * sizeof(Q16));
    }
    double t_mg = t_sum / (iters ? iters : 1);

    double flops = 2.0 * (double)M * (double)N;
    char key[24];
    if (incx == 1 && incy == 1) {
        key[0] = '-'; key[1] = 0;
    } else if (incy == 1) {
        snprintf(key, sizeof(key), "x%d", incx);
    } else if (incx == 1) {
        snprintf(key, sizeof(key), "y%d", incy);
    } else {
        snprintf(key, sizeof(key), "x%d/y%d", incx, incy);
    }
    perf_emit("qger", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("qger", key, N, iters, flops, t_subject, t_mg);
    free(A); free(Ai); free(X); free(Y);
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
    for (int xi = 0; xi < n_incx; ++xi) {
        int incx = incxs[xi]; if (incx == 0) continue;
        for (int yi = 0; yi < n_incy; ++yi) {
            int incy = incys[yi]; if (incy == 0) continue;
            for (int i = 0; i < n; ++i) run_one(sizes[i], sizes[i], incx, incy, iters, warmup);
        }
    }
    return 0;
}
