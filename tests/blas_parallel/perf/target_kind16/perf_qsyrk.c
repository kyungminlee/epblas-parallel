/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for qsyrk (overlay vs migrated).
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


BLAS_EXTERN void qsyrk_(const char *, const char *, const int *, const int *,
    const Q16 *, const Q16 *, const int *,
    const Q16 *, Q16 *, const int *, size_t, size_t);
BLAS_EXTERN void qsyrk_migrated_(const char *, const char *, const int *, const int *,
    const Q16 *, const Q16 *, const int *,
    const Q16 *, Q16 *, const int *, size_t, size_t);

static void run_one(char uplo, char trans, int N, int K, int iters, int warmup) {
    Q16 alpha = Q16_FROM(0.7), beta = Q16_FROM(0.3);
    int A_rows = (trans == 'N') ? N : K;
    int A_cols = (trans == 'N') ? K : N;
    int lda = A_rows, ldc = N;
    Q16 *A  = (Q16 *)perf_aligned_alloc(64, (size_t)A_rows * (size_t)A_cols * sizeof(Q16));
    Q16 *C  = (Q16 *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof(Q16));
    Q16 *Ci = (Q16 *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof(Q16));
    for (size_t i = 0; i < (size_t)A_rows*A_cols; ++i) { int s = 2; A[i] = Q16_FROM(perf_fill_double(i, s)); }
    for (size_t i = 0; i < (size_t)N*N; ++i)           { int s = 4; Ci[i] = Q16_FROM(perf_fill_double(i, s)); }
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof(Q16));
    for (int r = 0; r < warmup; ++r) {
        qsyrk_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)N * (size_t)N * sizeof(Q16));
        qsyrk_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)N * (size_t)N * sizeof(Q16));
    }
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof(Q16));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        qsyrk_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof(Q16));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        qsyrk_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = 1.0 * (double)N * (double)N * (double)K;
    char key[3] = {uplo, trans, 0};
    perf_emit("qsyrk", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("qsyrk", key, N, iters, flops, t_subject, t_mg);
    free(A); free(C); free(Ci);
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
