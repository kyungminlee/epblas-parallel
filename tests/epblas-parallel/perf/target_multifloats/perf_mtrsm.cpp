/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for mtrsm (overlay vs migrated).
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


BLAS_EXTERN void mtrsm_(const char *, const char *, const char *, const char *,
    const int *, const int *, const MFR *,
    const MFR *, const int *, MFR *, const int *,
    size_t, size_t, size_t, size_t);
BLAS_EXTERN void mtrsm_migrated_(const char *, const char *, const char *, const char *,
    const int *, const int *, const MFR *,
    const MFR *, const int *, MFR *, const int *,
    size_t, size_t, size_t, size_t);

static void run_one(char side, char uplo, char trans, char diag,
                    int M, int N, int iters, int warmup) {
    MFR alpha = MFR_FROM(0.7);
    int Asz = (side == 'L') ? M : N;
    int lda = Asz, ldb = M;
    MFR *A  = (MFR *)perf_aligned_alloc(64, (size_t)Asz * (size_t)Asz * sizeof(MFR));
    MFR *B  = (MFR *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(MFR));
    MFR *Bi = (MFR *)perf_aligned_alloc(64, (size_t)M * (size_t)N * sizeof(MFR));
    for (size_t i = 0; i < (size_t)Asz*Asz; ++i) { int s = 2; A[i] = MFR_FROM(perf_fill_double(i, s)); }
    /* diagonal dominance for trsm */
    for (int i = 0; i < Asz; ++i) A[(size_t)i * lda + i] = Tr_from_d((double)(Asz + 4));
    for (size_t i = 0; i < (size_t)M*N; ++i) { int s = 4; Bi[i] = MFR_FROM(perf_fill_double(i, s)); }
    memcpy(B, Bi, (size_t)M * (size_t)N * sizeof(MFR));
    for (int r = 0; r < warmup; ++r) {
        mtrsm_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof(MFR));
        mtrsm_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof(MFR));
    }
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        mtrsm_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof(MFR));
    }
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        mtrsm_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(B, Bi, (size_t)M * (size_t)N * sizeof(MFR));
    }
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = 1.0 * (double)M * (double)N * (double)M;
    char key[5] = {side, uplo, trans, diag, 0};
    perf_emit("mtrsm", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("mtrsm", key, N, iters, flops, t_subject, t_mg);
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
    const char transes[] = { 'N', 'T' };
    const char diags[]   = { 'N', 'U' };
    for (size_t s = 0; s < 2; ++s) for (size_t u = 0; u < 2; ++u)
      for (size_t t = 0; t < sizeof(transes); ++t)
        for (size_t d = 0; d < sizeof(diags); ++d)
          for (int i = 0; i < n; ++i)
              run_one(sides[s], uplos[u], transes[t], diags[d], sizes[i], sizes[i], iters, warmup);
    return 0;
}
