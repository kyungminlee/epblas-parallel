/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for xhemm (overlay vs migrated).
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


BLAS_EXTERN void xhemm_(const char *, const char *, const int *, const int *,
    const X16 *, const X16 *, const int *, const X16 *, const int *,
    const X16 *, X16 *, const int *, size_t, size_t);
BLAS_EXTERN void xhemm_migrated_(const char *, const char *, const int *, const int *,
    const X16 *, const X16 *, const int *, const X16 *, const int *,
    const X16 *, X16 *, const int *, size_t, size_t);

static void run_one(char side, char uplo, int M, int N, int iters, int warmup) {
    X16 alpha = X16_FROM(0.7, 0.0), beta = X16_FROM(0.3, 0.0);
    int Asz = (side == 'L') ? M : N;
    const size_t AAelt = (size_t)Asz * (size_t)Asz;
    const size_t MNelt = (size_t)M * (size_t)N;
    X16 *A  = PERF_ALLOC(X16, AAelt);
    X16 *B  = PERF_ALLOC(X16, MNelt);
    X16 *C  = PERF_ALLOC(X16, MNelt);
    X16 *Ci = PERF_ALLOC(X16, MNelt);
    int lda = Asz, ldb = M, ldc = M;
    PERF_FILL_C(X16, A,  AAelt, 2);
    PERF_FILL_C(X16, B,  MNelt, 3);
    PERF_FILL_C(X16, Ci, MNelt, 4);
    PERF_RESET(C, Ci, MNelt, X16);
    for (int r = 0; r < warmup; ++r) {
        xhemm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, MNelt, X16);
        xhemm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, MNelt, X16);
    }
    double t_subject, t_mg;
    PERF_RESET(C, Ci, MNelt, X16);
    PERF_TIME(t_subject, iters, xhemm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, MNelt, X16);
    PERF_TIME(t_mg,      iters, xhemm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    double flops = 8.0 * (double)M * (double)M * (double)N;
    char key[3] = {side, uplo, 0};
    PERF_EMIT("xhemm", key, N, iters, flops, t_subject, t_mg);
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
    const char sides[] = {'L', 'R'};
    const char uplos[] = {'U', 'L'};
    for (size_t s = 0; s < 2; ++s) for (size_t u = 0; u < 2; ++u)
        for (int i = 0; i < n; ++i)
            run_one(sides[s], uplos[u], sizes[i], sizes[i], iters, warmup);
    return 0;
}
