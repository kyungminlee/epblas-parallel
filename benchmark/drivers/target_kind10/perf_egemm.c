/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for egemm (overlay vs migrated).
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


BLAS_EXTERN void egemm_(const char *, const char *, const int *, const int *, const int *,
    const R10 *, const R10 *, const int *, const R10 *, const int *,
    const R10 *, R10 *, const int *, size_t, size_t);
BLAS_EXTERN void egemm_migrated_(const char *, const char *, const int *, const int *, const int *,
    const R10 *, const R10 *, const int *, const R10 *, const int *,
    const R10 *, R10 *, const int *, size_t, size_t);

static void run_one(char ta, char tb, int M, int N, int K, int iters, int warmup) {
    R10 alpha = R10_FROM(0.7), beta = R10_FROM(0.3);
    const size_t MKelt = (size_t)M * (size_t)K;
    const size_t KNelt = (size_t)K * (size_t)N;
    const size_t MNelt = (size_t)M * (size_t)N;
    R10 *A  = PERF_ALLOC(R10, MKelt);
    R10 *B  = PERF_ALLOC(R10, KNelt);
    R10 *C  = PERF_ALLOC(R10, MNelt);
    R10 *Ci = PERF_ALLOC(R10, MNelt);
    int lda = M, ldb = K, ldc = M;
    PERF_FILL_R(R10, A,  MKelt, 2);
    PERF_FILL_R(R10, B,  KNelt, 3);
    PERF_FILL_R(R10, Ci, MNelt, 4);
    PERF_RESET(C, Ci, MNelt, R10);

    for (int r = 0; r < warmup; ++r) {
        egemm_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, MNelt, R10);
        egemm_migrated_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, MNelt, R10);
    }
    double t_subject, t_mg;
    PERF_RESET(C, Ci, MNelt, R10);
    PERF_TIME(t_subject, iters, egemm_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, MNelt, R10);
    PERF_TIME(t_mg,      iters, egemm_migrated_(&ta, &tb, &M, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));

    double flops = 2.0 * (double)M * (double)N * (double)K;
    char key[3] = {ta, tb, 0};
    PERF_EMIT("egemm", key, N, iters, flops, t_subject, t_mg);
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
    const char *pairs[] = { "NN","TN","NT","TT" };
    for (size_t p = 0; p < sizeof(pairs)/sizeof(pairs[0]); ++p)
        for (int i = 0; i < n; ++i)
            run_one(pairs[p][0], pairs[p][1], sizes[i], sizes[i], sizes[i], iters, warmup);
    return 0;
}
