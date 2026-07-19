/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for ysymm (overlay vs migrated).
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


BLAS_EXTERN void ysymm_(const char *, const char *, const int *, const int *,
    const C10 *, const C10 *, const int *, const C10 *, const int *,
    const C10 *, C10 *, const int *, size_t, size_t);
BLAS_EXTERN void ysymm_migrated_(const char *, const char *, const int *, const int *,
    const C10 *, const C10 *, const int *, const C10 *, const int *,
    const C10 *, C10 *, const int *, size_t, size_t);

static void run_one(char side, char uplo, int M, int N, int iters, int warmup) {
    C10 alpha = C10_FROM(0.7, 0.0), beta = C10_FROM(0.3, 0.0);
    int Asz = (side == 'L') ? M : N;
    const size_t AAelt = (size_t)Asz * (size_t)Asz;
    const size_t MNelt = (size_t)M * (size_t)N;
    C10 *A  = PERF_ALLOC(C10, AAelt);
    C10 *B  = PERF_ALLOC(C10, MNelt);
    C10 *C  = PERF_ALLOC(C10, MNelt);
    C10 *Ci = PERF_ALLOC(C10, MNelt);
    int lda = Asz, ldb = M, ldc = M;
    PERF_FILL_C(C10, A,  AAelt, 2);
    PERF_FILL_C(C10, B,  MNelt, 3);
    PERF_FILL_C(C10, Ci, MNelt, 4);
    PERF_RESET(C, Ci, MNelt, C10);
    for (int r = 0; r < warmup; ++r) {
        ysymm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, MNelt, C10);
        ysymm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, MNelt, C10);
    }
    double t_subject, t_mg;
    PERF_RESET(C, Ci, MNelt, C10);
    PERF_TIME(t_subject, iters, ysymm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, MNelt, C10);
    PERF_TIME(t_mg,      iters, ysymm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    char key[3] = {side, uplo, 0};
    PERF_EMIT("ysymm", key, N, iters, t_subject, t_mg);
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
    const char sides[] = {'L', 'R'};
    const char uplos[] = {'U', 'L'};
    for (size_t s = 0; s < 2; ++s) for (size_t u = 0; u < 2; ++u)
        for (int i = 0; i < n; ++i)
            run_one(sides[s], uplos[u], sizes[i], sizes[i], iters, warmup);
    return 0;
}
