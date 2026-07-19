/* Hand-maintained perf harness (originally generated; the generator was retired).
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
    const size_t NNelt = (size_t)N * (size_t)N;
    X16 *A  = PERF_ALLOC(X16, NNelt);
    X16 *Ai = PERF_ALLOC(X16, NNelt);
    X16 *X  = PERF_ALLOC(X16, lenx);
    PERF_FILL_C(X16, Ai, NNelt, 2);
    PERF_FILL_C(X16, X,  lenx, 3);
    PERF_RESET(A, Ai, NNelt, X16);
    for (int r = 0; r < warmup; ++r) {
        xher_(&uplo, &N, &alpha, X, &incx, A, &N, 1);          PERF_RESET(A, Ai, NNelt, X16);
        xher_migrated_(&uplo, &N, &alpha, X, &incx, A, &N, 1); PERF_RESET(A, Ai, NNelt, X16);
    }
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(A, Ai, NNelt, X16), xher_(&uplo, &N, &alpha, X, &incx, A, &N, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(A, Ai, NNelt, X16), xher_migrated_(&uplo, &N, &alpha, X, &incx, A, &N, 1));
    char key[16];
    if (incx == 1) {
        key[0] = uplo; key[1] = 0;
    } else {
        snprintf(key, sizeof(key), "%c/x%d", uplo, incx);
    }
    PERF_EMIT("xher", key, N, iters, t_subject, t_mg);
    free(A); free(Ai); free(X);
}

static const int default_sizes[] = {128, 256, 512};
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
