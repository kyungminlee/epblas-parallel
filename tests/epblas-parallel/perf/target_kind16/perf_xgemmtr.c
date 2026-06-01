/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for xgemmtr (overlay vs migrated).
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


BLAS_EXTERN void xgemmtr_(const char *, const char *, const char *,
    const int *, const int *,
    const X16 *, const X16 *, const int *, const X16 *, const int *,
    const X16 *, X16 *, const int *,
    size_t, size_t, size_t);
BLAS_EXTERN void xgemmtr_migrated_(const char *, const char *, const char *,
    const int *, const int *,
    const X16 *, const X16 *, const int *, const X16 *, const int *,
    const X16 *, X16 *, const int *,
    size_t, size_t, size_t);

static void run_one(char uplo, char ta, char tb, int N, int K, int iters, int warmup) {
    X16 alpha = X16_FROM(0.7, 0.0), beta = X16_FROM(0.3, 0.0);
    int Arows = (ta == 'N') ? N : K;
    int Acols = (ta == 'N') ? K : N;
    int Brows = (tb == 'N') ? K : N;
    int Bcols = (tb == 'N') ? N : K;
    const size_t ABelt = (size_t)Arows * (size_t)Acols;
    const size_t BBelt = (size_t)Brows * (size_t)Bcols;
    const size_t NNelt = (size_t)N * (size_t)N;
    int lda = Arows, ldb = Brows, ldc = N;
    X16 *A  = PERF_ALLOC(X16, ABelt);
    X16 *B  = PERF_ALLOC(X16, BBelt);
    X16 *C  = PERF_ALLOC(X16, NNelt);
    X16 *Ci = PERF_ALLOC(X16, NNelt);
    PERF_FILL_C(X16, A,  ABelt, 2);
    PERF_FILL_C(X16, B,  BBelt, 3);
    PERF_FILL_C(X16, Ci, NNelt, 4);
    PERF_RESET(C, Ci, NNelt, X16);
    for (int r = 0; r < warmup; ++r) {
        xgemmtr_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1);          PERF_RESET(C, Ci, NNelt, X16);
        xgemmtr_migrated_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1); PERF_RESET(C, Ci, NNelt, X16);
    }
    double t_subject, t_mg;
    PERF_RESET(C, Ci, NNelt, X16);
    PERF_TIME(t_subject, iters, xgemmtr_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1));
    PERF_RESET(C, Ci, NNelt, X16);
    PERF_TIME(t_mg,      iters, xgemmtr_migrated_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1));
    double flops = 4.0 * (double)N * (double)N * (double)K;
    char key[4] = {uplo, ta, tb, 0};
    PERF_EMIT("xgemmtr", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(C); free(Ci);
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
    /* Sample full (ta, tb) grid: N/T for real, N/T/C for complex.
     * Trans choice flips the inner walk over A and B; covering all
     * combinations stresses every code path the kernel may take. */
    const char *pairs[] = { "NN", "NT", "NC", "TN", "TT", "TC", "CN", "CT", "CC" };
    for (size_t u = 0; u < 2; ++u)
        for (size_t p = 0; p < sizeof(pairs)/sizeof(pairs[0]); ++p)
            for (int i = 0; i < n; ++i)
                run_one(uplos[u], pairs[p][0], pairs[p][1], sizes[i], sizes[i], iters, warmup);
    return 0;
}
