/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for ytrmm (overlay vs migrated).
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


BLAS_EXTERN void ytrmm_(const char *, const char *, const char *, const char *,
    const int *, const int *, const C10 *,
    const C10 *, const int *, C10 *, const int *,
    size_t, size_t, size_t, size_t);
BLAS_EXTERN void ytrmm_migrated_(const char *, const char *, const char *, const char *,
    const int *, const int *, const C10 *,
    const C10 *, const int *, C10 *, const int *,
    size_t, size_t, size_t, size_t);

static void run_one(char side, char uplo, char trans, char diag,
                    int M, int N, int iters, int warmup) {
    C10 alpha = C10_FROM(0.7, 0.0);
    int Asz = (side == 'L') ? M : N;
    int lda = Asz, ldb = M;
    C10 *A  = (C10 *)perf_aligned_alloc(64, (size_t)Asz * (size_t)Asz * sizeof(C10));
    C10 *B  = (C10 *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(C10));
    C10 *Bi = (C10 *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(C10));
    for (size_t i = 0; i < (size_t)Asz*Asz; ++i) { int s = 2; A[i] = C10_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    /* diagonal dominance for trsm */
    for (int i = 0; i < Asz; ++i) A[(size_t)i * lda + i] = Tc_from_d((double)(Asz + 4));
    for (size_t i = 0; i < (size_t)M*N; ++i) { int s = 4; Bi[i] = C10_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    memcpy(B, Bi, (size_t)M * (size_t)N * sizeof(C10));
    for (int r = 0; r < warmup; ++r) {
        ytrmm_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof(C10));
        ytrmm_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof(C10));
    }
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        ytrmm_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof(C10));
    }
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        ytrmm_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof(C10));
    }
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = 4.0 * (double)M * (double)N * (double)M;
    char key[5] = {side, uplo, trans, diag, 0};
    perf_emit("ytrmm", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("ytrmm", key, N, iters, flops, t_subject, t_mg);
    free(A); free(B); free(Bi);
}

static const int default_sizes[] = {64, 128, 256, 512};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  10);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 2);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    /* Sample over (side, uplo, trans, diag) — diag=N/U so the unit-diag
     * branch of trmm/trsm is exercised; full grid omits no categorical. */
    const char sides[] = {'L', 'R'};
    const char uplos[] = {'U', 'L'};
    const char transes[] = { 'N', 'T', 'C' };
    const char diags[]   = { 'N', 'U' };
    for (size_t s = 0; s < 2; ++s) for (size_t u = 0; u < 2; ++u)
      for (size_t t = 0; t < sizeof(transes); ++t)
        for (size_t d = 0; d < sizeof(diags); ++d)
          for (int i = 0; i < n; ++i)
              run_one(sides[s], uplos[u], transes[t], diags[d], sizes[i], sizes[i], iters, warmup);
    return 0;
}
