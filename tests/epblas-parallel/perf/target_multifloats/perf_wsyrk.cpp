/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for wsyrk (overlay vs migrated).
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


BLAS_EXTERN void wsyrk_(const char *, const char *, const int *, const int *,
    const MFC *, const MFC *, const int *,
    const MFC *, MFC *, const int *, size_t, size_t);
BLAS_EXTERN void wsyrk_migrated_(const char *, const char *, const int *, const int *,
    const MFC *, const MFC *, const int *,
    const MFC *, MFC *, const int *, size_t, size_t);

static void run_one(char uplo, char trans, int N, int K, int iters, int warmup) {
    MFC alpha = MFC_FROM(0.7, 0.0), beta = MFC_FROM(0.3, 0.0);
    int A_rows = (trans == 'N') ? N : K;
    int A_cols = (trans == 'N') ? K : N;
    int lda = A_rows, ldc = N;
    MFC *A  = (MFC *)perf_aligned_alloc(64, (size_t)A_rows * (size_t)A_cols * sizeof(MFC));
    MFC *C  = (MFC *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof(MFC));
    MFC *Ci = (MFC *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof(MFC));
    for (size_t i = 0; i < (size_t)A_rows*A_cols; ++i) { int s = 2; A[i] = MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < (size_t)N*N; ++i)           { int s = 4; Ci[i] = MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof(MFC));
    for (int r = 0; r < warmup; ++r) {
        wsyrk_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)N * (size_t)N * sizeof(MFC));
        wsyrk_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)N * (size_t)N * sizeof(MFC));
    }
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof(MFC));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        wsyrk_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(C, Ci, (size_t)N * (size_t)N * sizeof(MFC));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        wsyrk_migrated_(&uplo, &trans, &N, &K, &alpha, A, &lda, &beta, C, &ldc, 1, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = 4.0 * (double)N * (double)N * (double)K;
    char key[3] = {uplo, trans, 0};
    perf_emit("wsyrk", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("wsyrk", key, N, iters, flops, t_subject, t_mg);
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
