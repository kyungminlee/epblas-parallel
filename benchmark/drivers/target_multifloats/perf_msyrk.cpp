/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for msyrk (overlay vs migrated).
 * Built per-executable with -ffunction-sections / --gc-sections.
 */
#include "../perf_common.h"

#ifdef __cplusplus
#define BLAS_EXTERN extern "C"
#else
#define BLAS_EXTERN extern
#endif
typedef struct { double v[2]; } MFR;     /* float64x2 layout (POD) */
typedef struct { MFR r; MFR i; } MFC;    /* complex64x2 layout (POD) */
static inline MFR MFR_FROM(double d) { MFR x; x.v[0] = d; x.v[1] = 0.0; return x; }
static inline MFC MFC_FROM(double re, double im) {
    MFC z;
    z.r.v[0] = re; z.r.v[1] = 0.0;
    z.i.v[0] = im; z.i.v[1] = 0.0;
    return z;
}
static inline MFR Tr_from_d(double d) { return MFR_FROM(d); }
static inline MFC Tc_from_d(double d) { return MFC_FROM(d, 0.0); }


BLAS_EXTERN void msyrk_(const char *, const char *, const int *, const int *,
    const MFR *, const MFR *, const int *,
    const MFR *, MFR *, const int *, size_t, size_t);
BLAS_EXTERN void msyrk_migrated_(const char *, const char *, const int *, const int *,
    const MFR *, const MFR *, const int *,
    const MFR *, MFR *, const int *, size_t, size_t);

static void run_one(char uplo, char trans, int N, int K, int iters, int warmup) {
    MFR alpha = MFR_FROM(0.7), beta = MFR_FROM(0.3);
    int A_rows = (trans == 'N') ? N : K;
    int A_cols = (trans == 'N') ? K : N;
    const size_t AAelt = (size_t)A_rows * (size_t)A_cols;
    const size_t NNelt = (size_t)N * (size_t)N;
    int lda = A_rows, ldc = N;
    MFR *A  = PERF_ALLOC(MFR, AAelt);
    MFR *C  = PERF_ALLOC(MFR, NNelt);
    MFR *Ci = PERF_ALLOC(MFR, NNelt);
    PERF_FILL_R(MFR, A,  AAelt, 2);
    PERF_FILL_R(MFR, Ci, NNelt, 4);
    PERF_RESET(C, Ci, NNelt, MFR);
    for (int r = 0; r < warmup; ++r) {
        msyrk_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, NNelt, MFR);
        msyrk_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, NNelt, MFR);
    }
    double t_subject, t_mg;
    PERF_RESET(C, Ci, NNelt, MFR);
    PERF_TIME(t_subject, iters, msyrk_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, NNelt, MFR);
    PERF_TIME(t_mg,      iters, msyrk_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1));
    char key[3] = {uplo, trans, 0};
    PERF_EMIT("msyrk", key, N, iters, t_subject, t_mg);
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
