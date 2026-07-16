/* Hand-maintained perf harness (originally generated; the generator was retired).
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
    const size_t AAelt = (size_t)Asz * (size_t)Asz;
    const size_t MNelt = (size_t)M * (size_t)N;
    int lda = Asz, ldb = M;
    MFR *A  = PERF_ALLOC(MFR, AAelt);
    MFR *B  = PERF_ALLOC(MFR, MNelt);
    MFR *Bi = PERF_ALLOC(MFR, MNelt);
    PERF_FILL_R(MFR, A,  AAelt, 2);
    /* diagonal dominance for trsm */
    for (int i = 0; i < Asz; ++i) A[(size_t)i * lda + i] = Tr_from_d((double)(Asz + 4));
    PERF_FILL_R(MFR, Bi, MNelt, 4);
    PERF_RESET(B, Bi, MNelt, MFR);
    for (int r = 0; r < warmup; ++r) {
        mtrsm_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1);          PERF_RESET(B, Bi, MNelt, MFR);
        mtrsm_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1); PERF_RESET(B, Bi, MNelt, MFR);
    }
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(B, Bi, MNelt, MFR),
        mtrsm_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(B, Bi, MNelt, MFR),
        mtrsm_migrated_(&side, &uplo, &trans, &diag, &M, &N, &alpha, A, &lda, B, &ldb, 1, 1, 1, 1));
    double flops = 1.0 * (double)M * (double)N * (double)M;
    char key[5] = {side, uplo, trans, diag, 0};
    PERF_EMIT("mtrsm", key, N, iters, flops, t_subject, t_mg);
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
