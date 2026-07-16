/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for ygerc (overlay vs migrated).
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


BLAS_EXTERN void ygerc_(const int *, const int *, const C10 *,
    const C10 *, const int *, const C10 *, const int *, C10 *, const int *);
BLAS_EXTERN void ygerc_migrated_(const int *, const int *, const C10 *,
    const C10 *, const int *, const C10 *, const int *, C10 *, const int *);

static void run_one(int M, int N, int incx, int incy, int iters, int warmup) {
    C10 alpha = C10_FROM(0.7, 0.0);
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(M - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(N - 1) * (size_t)absy;
    const size_t MNelt = (size_t)M * (size_t)N;
    C10 *A  = PERF_ALLOC(C10, MNelt);
    C10 *Ai = PERF_ALLOC(C10, MNelt);
    C10 *X  = PERF_ALLOC(C10, lenx);
    C10 *Y  = PERF_ALLOC(C10, leny);
    PERF_FILL_C(C10, Ai, MNelt, 2);
    PERF_FILL_C(C10, X,  lenx, 3);
    PERF_FILL_C(C10, Y,  leny, 4);
    PERF_RESET(A, Ai, MNelt, C10);

    for (int r = 0; r < warmup; ++r) {
        ygerc_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M);          PERF_RESET(A, Ai, MNelt, C10);
        ygerc_migrated_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M); PERF_RESET(A, Ai, MNelt, C10);
    }
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(A, Ai, MNelt, C10), ygerc_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(A, Ai, MNelt, C10), ygerc_migrated_(&M, &N, &alpha, X, &incx, Y, &incy, A, &M));

    double flops = 8.0 * (double)M * (double)N;
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
    PERF_EMIT("ygerc", key, N, iters, flops, t_subject, t_mg);
    free(A); free(Ai); free(X); free(Y);
}

static const int default_sizes[] = {128, 256, 512};
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
