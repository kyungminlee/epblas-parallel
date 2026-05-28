/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
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
    C10 *A  = (C10 *)perf_aligned_alloc(64, (size_t)Asz * (size_t)Asz * sizeof(C10));
    C10 *B  = (C10 *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(C10));
    C10 *C  = (C10 *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(C10));
    C10 *Ci = (C10 *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(C10));
    int lda = Asz, ldb = M, ldc = M;
    for (size_t i = 0; i < (size_t)Asz*Asz; ++i) { int s = 2; A[i] = C10_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < (size_t)M*N; ++i)     { int s = 3; B[i] = C10_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < (size_t)M*N; ++i)     { int s = 4; Ci[i] = C10_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof(C10));
    for (int r = 0; r < warmup; ++r) {
        ysymm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)M * (size_t)N * sizeof(C10));
        ysymm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
        memcpy(C, Ci, (size_t)M * (size_t)N * sizeof(C10));
    }
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof(C10));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        ysymm_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(C, Ci, (size_t)M * (size_t)N * sizeof(C10));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it)
        ysymm_migrated_(&side, &uplo, &M, &N, &alpha, A, &lda, B, &ldb, &beta, C, &ldc, 1, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = 8.0 * (double)M * (double)M * (double)N;
    char key[3] = {side, uplo, 0};
    perf_emit("ysymm", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("ysymm", key, N, iters, flops, t_subject, t_mg);
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
