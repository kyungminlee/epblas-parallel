/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for wgemmtr (overlay vs migrated).
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


BLAS_EXTERN void wgemmtr_(const char *, const char *, const char *,
    const int *, const int *,
    const MFC *, const MFC *, const int *, const MFC *, const int *,
    const MFC *, MFC *, const int *,
    size_t, size_t, size_t);
BLAS_EXTERN void wgemmtr_migrated_(const char *, const char *, const char *,
    const int *, const int *,
    const MFC *, const MFC *, const int *, const MFC *, const int *,
    const MFC *, MFC *, const int *,
    size_t, size_t, size_t);

static void run_one(char uplo, char ta, char tb, int N, int K, int iters, int warmup) {
    MFC alpha = MFC_FROM(0.7, 0.0), beta = MFC_FROM(0.3, 0.0);
    int Arows = (ta == 'N') ? N : K;
    int Acols = (ta == 'N') ? K : N;
    int Brows = (tb == 'N') ? K : N;
    int Bcols = (tb == 'N') ? N : K;
    const size_t ABelt = (size_t)Arows * (size_t)Acols;
    const size_t BBelt = (size_t)Brows * (size_t)Bcols;
    const size_t NNelt = (size_t)N * (size_t)N;
    int lda = Arows, ldb = Brows, ldc = N;
    MFC *A  = PERF_ALLOC(MFC, ABelt);
    MFC *B  = PERF_ALLOC(MFC, BBelt);
    MFC *C  = PERF_ALLOC(MFC, NNelt);
    MFC *Ci = PERF_ALLOC(MFC, NNelt);
    PERF_FILL_C(MFC, A,  ABelt, 2);
    PERF_FILL_C(MFC, B,  BBelt, 3);
    PERF_FILL_C(MFC, Ci, NNelt, 4);
    PERF_RESET(C, Ci, NNelt, MFC);
    for (int r = 0; r < warmup; ++r) {
        wgemmtr_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1);          PERF_RESET(C, Ci, NNelt, MFC);
        wgemmtr_migrated_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1); PERF_RESET(C, Ci, NNelt, MFC);
    }
    double t_subject, t_mg;
    PERF_RESET(C, Ci, NNelt, MFC);
    PERF_TIME(t_subject, iters, wgemmtr_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1));
    PERF_RESET(C, Ci, NNelt, MFC);
    PERF_TIME(t_mg,      iters, wgemmtr_migrated_(&uplo, &ta, &tb, &N, &K, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1, 1));
    double flops = 4.0 * (double)N * (double)N * (double)K;
    char key[4] = {uplo, ta, tb, 0};
    PERF_EMIT("wgemmtr", key, N, iters, flops, t_subject, t_mg);
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
