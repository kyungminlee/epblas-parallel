/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for qsymm (overlay vs migrated).
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


BLAS_EXTERN void qsymm_(const char *, const char *, const int *, const int *,
    const Q16 *, const Q16 *, const int *, const Q16 *, const int *,
    const Q16 *, Q16 *, const int *, size_t, size_t);
BLAS_EXTERN void qsymm_migrated_(const char *, const char *, const int *, const int *,
    const Q16 *, const Q16 *, const int *, const Q16 *, const int *,
    const Q16 *, Q16 *, const int *, size_t, size_t);

static void run_one(char side, char uplo, int M, int N, int iters, int warmup) {
    Q16 alpha = Q16_FROM(0.7), beta = Q16_FROM(0.3);
    int Asz = (side == 'L') ? M : N;
    const size_t AAelt = (size_t)Asz * (size_t)Asz;
    const size_t MNelt = (size_t)M * (size_t)N;
    Q16 *A  = PERF_ALLOC(Q16, AAelt);
    Q16 *B  = PERF_ALLOC(Q16, MNelt);
    Q16 *C  = PERF_ALLOC(Q16, MNelt);
    Q16 *Ci = PERF_ALLOC(Q16, MNelt);
    int lda = Asz, ldb = M, ldc = M;
    PERF_FILL_R(Q16, A,  AAelt, 2);
    PERF_FILL_R(Q16, B,  MNelt, 3);
    PERF_FILL_R(Q16, Ci, MNelt, 4);
    PERF_RESET(C, Ci, MNelt, Q16);
    for (int r = 0; r < warmup; ++r) {
        qsymm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, MNelt, Q16);
        qsymm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, MNelt, Q16);
    }
    double t_subject, t_mg;
    PERF_RESET(C, Ci, MNelt, Q16);
    PERF_TIME(t_subject, iters, qsymm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, MNelt, Q16);
    PERF_TIME(t_mg,      iters, qsymm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    double flops = 2.0 * (double)M * (double)M * (double)N;
    char key[3] = {side, uplo, 0};
    PERF_EMIT("qsymm", key, N, iters, flops, t_subject, t_mg);
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
