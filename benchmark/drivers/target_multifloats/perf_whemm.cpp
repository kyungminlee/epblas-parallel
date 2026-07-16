/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for whemm (overlay vs migrated).
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


BLAS_EXTERN void whemm_(const char *, const char *, const int *, const int *,
    const MFC *, const MFC *, const int *, const MFC *, const int *,
    const MFC *, MFC *, const int *, size_t, size_t);
BLAS_EXTERN void whemm_migrated_(const char *, const char *, const int *, const int *,
    const MFC *, const MFC *, const int *, const MFC *, const int *,
    const MFC *, MFC *, const int *, size_t, size_t);

static void run_one(char side, char uplo, int M, int N, int iters, int warmup) {
    MFC alpha = MFC_FROM(0.7, 0.0), beta = MFC_FROM(0.3, 0.0);
    int Asz = (side == 'L') ? M : N;
    const size_t AAelt = (size_t)Asz * (size_t)Asz;
    const size_t MNelt = (size_t)M * (size_t)N;
    MFC *A  = PERF_ALLOC(MFC, AAelt);
    MFC *B  = PERF_ALLOC(MFC, MNelt);
    MFC *C  = PERF_ALLOC(MFC, MNelt);
    MFC *Ci = PERF_ALLOC(MFC, MNelt);
    int lda = Asz, ldb = M, ldc = M;
    PERF_FILL_C(MFC, A,  AAelt, 2);
    PERF_FILL_C(MFC, B,  MNelt, 3);
    PERF_FILL_C(MFC, Ci, MNelt, 4);
    PERF_RESET(C, Ci, MNelt, MFC);
    for (int r = 0; r < warmup; ++r) {
        whemm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);          PERF_RESET(C, Ci, MNelt, MFC);
        whemm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1); PERF_RESET(C, Ci, MNelt, MFC);
    }
    double t_subject, t_mg;
    PERF_RESET(C, Ci, MNelt, MFC);
    PERF_TIME(t_subject, iters, whemm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    PERF_RESET(C, Ci, MNelt, MFC);
    PERF_TIME(t_mg,      iters, whemm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1));
    double flops = 8.0 * (double)M * (double)M * (double)N;
    char key[3] = {side, uplo, 0};
    PERF_EMIT("whemm", key, N, iters, flops, t_subject, t_mg);
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
