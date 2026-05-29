/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for ysyr2k (overlay vs migrated).
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


BLAS_EXTERN void ysyr2k_(const char *, const char *, const int *, const int *,
    const C10 *, const C10 *, const int *,
    const C10 *, const int *,
    const C10 *, C10 *, const int *, size_t, size_t);
BLAS_EXTERN void ysyr2k_migrated_(const char *, const char *, const int *, const int *,
    const C10 *, const C10 *, const int *,
    const C10 *, const int *,
    const C10 *, C10 *, const int *, size_t, size_t);

static void run_one(char uplo, char trans, int N, int K, int iters, int warmup) {
    C10 alpha = C10_FROM(0.7, 0.0);
    C10  beta  = C10_FROM(0.3, 0.0);
    int A_rows = (trans == 'N') ? N : K;
    int A_cols = (trans == 'N') ? K : N;
    const size_t AAelt = (size_t)A_rows * (size_t)A_cols;
    const size_t NNelt = (size_t)N * (size_t)N;
    int lda = A_rows, ldb = A_rows, ldc = N;
    C10 *A  = PERF_ALLOC(C10, AAelt);
    C10 *B  = PERF_ALLOC(C10, AAelt);
    C10 *C  = PERF_ALLOC(C10, NNelt);
    C10 *Ci = PERF_ALLOC(C10, NNelt);
    PERF_FILL_C(C10, A,  AAelt, 2);
    PERF_FILL_C(C10, B,  AAelt, 3);
    PERF_FILL_C(C10, Ci, NNelt, 4);
    PERF_RESET(C, Ci, NNelt, C10);
    for (int r = 0; r < warmup; ++r) {
        ysyr2k_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, NNelt, C10);
        ysyr2k_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, NNelt, C10);
    }
    double t_subject, t_mg;
    PERF_RESET(C, Ci, NNelt, C10);
    PERF_TIME(t_subject, iters, ysyr2k_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, NNelt, C10);
    PERF_TIME(t_mg,      iters, ysyr2k_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    double flops = 8.0 * (double)N * (double)N * (double)K;
    char key[3] = {uplo, trans, 0};
    PERF_EMIT("ysyr2k", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(C); free(Ci);
}

static const int default_sizes[] = {64, 128, 256, 512};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  10);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 2);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    const char uplos[] = {'U', 'L'};
    const char transes[] = { 'N', 'T' };
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < 2; ++t)
        for (int i = 0; i < n; ++i)
            run_one(uplos[u], transes[t], sizes[i], sizes[i], iters, warmup);
    return 0;
}
