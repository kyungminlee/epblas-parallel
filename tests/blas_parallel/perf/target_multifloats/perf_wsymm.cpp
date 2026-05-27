/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for wsymm (overlay vs migrated).
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


BLAS_EXTERN void wsymm_(const char *, const char *, const int *, const int *,
    const MFC *, const MFC *, const int *, const MFC *, const int *,
    const MFC *, MFC *, const int *, size_t, size_t);
BLAS_EXTERN void wsymm_migrated_(const char *, const char *, const int *, const int *,
    const MFC *, const MFC *, const int *, const MFC *, const int *,
    const MFC *, MFC *, const int *, size_t, size_t);

static void run_one(char side, char uplo, int M, int N, int iters, int warmup) {
    MFC alpha = MFC_FROM(0.7, 0.0), beta = MFC_FROM(0.3, 0.0);
    int Asz = (side == 'L') ? M : N;
    MFC *A  = (MFC *)perf_aligned_alloc(64, (size_t)Asz * (size_t)Asz * sizeof(MFC));
    MFC *B  = (MFC *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(MFC));
    MFC *C  = (MFC *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(MFC));
    MFC *Ci = (MFC *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(MFC));
    int lda = Asz, ldb = M, ldc = M;
    for (size_t i = 0; i < (size_t)Asz*Asz; ++i) { int s = 2; A[i] = MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < (size_t)M*N; ++i)     { int s = 3; B[i] = MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < (size_t)M*N; ++i)     { int s = 4; Ci[i] = MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof(MFC));
    for (int r = 0; r < warmup; ++r) {
        wsymm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)M * (size_t)N * sizeof(MFC));
        wsymm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)M * (size_t)N * sizeof(MFC));
    }
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof(MFC));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        wsymm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof(MFC));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        wsymm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = 8.0 * (double)M * (double)M * (double)N;
    char key[3] = {side, uplo, 0};
    perf_emit("wsymm", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("wsymm", key, N, iters, flops, t_subject, t_mg);
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
