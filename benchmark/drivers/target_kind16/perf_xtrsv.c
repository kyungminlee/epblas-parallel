/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for xtrsv (overlay vs migrated).
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


BLAS_EXTERN void xtrsv_(const char *, const char *, const char *, const int *,
    const X16 *, const int *, X16 *, const int *, size_t, size_t, size_t);
BLAS_EXTERN void xtrsv_migrated_(const char *, const char *, const char *, const int *,
    const X16 *, const int *, X16 *, const int *, size_t, size_t, size_t);

static void run_one(char uplo, char trans, char diag, int N, int incx,
                    int iters, int warmup) {
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t NNelt = (size_t)N * (size_t)N;
    X16 *A  = PERF_ALLOC(X16, NNelt);
    X16 *X  = PERF_ALLOC(X16, lenx);
    X16 *Xi = PERF_ALLOC(X16, lenx);
    /* Diagonally dominant for trsv stability */
    PERF_FILL_C(X16, A, NNelt, 2);
    for (int i = 0; i < N; ++i) {
        size_t idx = (size_t)i * N + i;
        A[idx] = Tc_from_d((double)(N + 4));
    }
    PERF_FILL_C(X16, Xi, lenx, 3);
    PERF_RESET(X, Xi, lenx, X16);
    for (int r = 0; r < warmup; ++r) {
        xtrsv_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1);          PERF_RESET(X, Xi, lenx, X16);
        xtrsv_migrated_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1); PERF_RESET(X, Xi, lenx, X16);
    }
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(X, Xi, lenx, X16), xtrsv_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(X, Xi, lenx, X16), xtrsv_migrated_(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1));
    /* Key encodes UPLO/TRANS/DIAG + stride. Examples: "LTN" (incx=1),
     * "LTN/x2" (incx=2), "LTN/x-1" (incx=-1). incx=1 keeps the old
     * key so existing reports stay parseable. */
    char key[16];
    if (incx == 1) {
        key[0] = uplo; key[1] = trans; key[2] = diag; key[3] = 0;
    } else {
        snprintf(key, sizeof(key), "%c%c%c/x%d", uplo, trans, diag, incx);
    }
    PERF_EMIT("xtrsv", key, N, iters, t_subject, t_mg);
    free(A); free(X); free(Xi);
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
    const char transes[] = { 'N','T','C' };
    const char diags[]   = { 'N', 'U' };
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < sizeof(transes); ++t)
    for (size_t d = 0; d < sizeof(diags); ++d) {
        char uplo = (u == 0) ? 'U' : 'L';
        char trans = transes[t];
        char diag = diags[d];
        for (int xi = 0; xi < n_incx; ++xi) {
            int incx = incxs[xi];
            if (incx == 0) continue;
            for (int i = 0; i < n; ++i) run_one(uplo, trans, diag, sizes[i], incx, iters, warmup);
        }
    }
    return 0;
}
