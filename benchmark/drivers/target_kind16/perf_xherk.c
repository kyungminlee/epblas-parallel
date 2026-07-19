/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for xherk (overlay vs migrated).
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


BLAS_EXTERN void xherk_(const char *, const char *, const int *, const int *,
    const Q16 *, const X16 *, const int *,
    const Q16 *, X16 *, const int *, size_t, size_t);
BLAS_EXTERN void xherk_migrated_(const char *, const char *, const int *, const int *,
    const Q16 *, const X16 *, const int *,
    const Q16 *, X16 *, const int *, size_t, size_t);

static void run_one(char uplo, char trans, int N, int K, int iters, int warmup) {
    Q16 alpha = Q16_FROM(0.7), beta = Q16_FROM(0.3);
    int A_rows = (trans == 'N') ? N : K;
    int A_cols = (trans == 'N') ? K : N;
    const size_t AAelt = (size_t)A_rows * (size_t)A_cols;
    const size_t NNelt = (size_t)N * (size_t)N;
    int lda = A_rows, ldc = N;
    X16 *A  = PERF_ALLOC(X16, AAelt);
    X16 *C  = PERF_ALLOC(X16, NNelt);
    X16 *Ci = PERF_ALLOC(X16, NNelt);
    PERF_FILL_C(X16, A,  AAelt, 2);
    PERF_FILL_C(X16, Ci, NNelt, 4);
    PERF_RESET(C, Ci, NNelt, X16);
    for (int r = 0; r < warmup; ++r) {
        xherk_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, NNelt, X16);
        xherk_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, NNelt, X16);
    }
    double t_subject, t_mg;
    PERF_RESET(C, Ci, NNelt, X16);
    PERF_TIME(t_subject, iters, xherk_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, NNelt, X16);
    PERF_TIME(t_mg,      iters, xherk_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1));
    char key[3] = {uplo, trans, 0};
    PERF_EMIT("xherk", key, N, iters, t_subject, t_mg);
    free(A); free(C); free(Ci);
}

static const int default_sizes[] = {64, 128, 256};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  10);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 2);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    const char uplos[] = {'U', 'L'};
    const char transes[] = { 'N', 'C' };
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < 2; ++t)
        for (int i = 0; i < n; ++i)
            run_one(uplos[u], transes[t], sizes[i], sizes[i], iters, warmup);
    return 0;
}
