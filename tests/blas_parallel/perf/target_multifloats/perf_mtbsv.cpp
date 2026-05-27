/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for mtbsv (overlay vs migrated).
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


BLAS_EXTERN void mtbsv_(const char *, const char *, const char *, const int *, const int *,
    const MFR *, const int *, MFR *, const int *, size_t, size_t, size_t);
BLAS_EXTERN void mtbsv_migrated_(const char *, const char *, const char *, const int *, const int *,
    const MFR *, const int *, MFR *, const int *, size_t, size_t, size_t);

static void run_one(char uplo, char trans, char diag, int N, int K, int incx,
                    int iters, int warmup) {
    int LDA = K + 1;
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    MFR *A  = (MFR *)perf_aligned_alloc(64, (size_t)LDA * (size_t)N * sizeof(MFR));
    MFR *X  = (MFR *)perf_aligned_alloc(64, lenx * sizeof(MFR));
    MFR *Xi = (MFR *)perf_aligned_alloc(64, lenx * sizeof(MFR));
    for (size_t i = 0; i < (size_t)LDA*N; ++i) { int s = 2; A[i] = MFR_FROM(perf_fill_double(i, s)); }
    /* diagonal at known row of band — large to stabilize tbsv */
    int diag_row = (uplo == 'U') ? K : 0;
    for (int j = 0; j < N; ++j) A[(size_t)j * LDA + diag_row] = Tr_from_d((double)(K + 4));
    for (size_t i = 0; i < lenx; ++i) { int s = 3; Xi[i] = MFR_FROM(perf_fill_double(i, s)); }
    memcpy(X, Xi, lenx * sizeof(MFR));
    for (int r = 0; r < warmup; ++r) {
        mtbsv_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1);
        memcpy(X, Xi, lenx * sizeof(MFR));
        mtbsv_migrated_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1);
        memcpy(X, Xi, lenx * sizeof(MFR));
    }
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        mtbsv_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, lenx * sizeof(MFR));
    }
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        mtbsv_migrated_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, lenx * sizeof(MFR));
    }
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = 1.0 * (double)(2*K+1) * (double)N;
    char key[16];
    if (incx == 1) {
        key[0] = uplo; key[1] = trans; key[2] = diag; key[3] = 0;
    } else {
        snprintf(key, sizeof(key), "%c%c%c/x%d", uplo, trans, diag, incx);
    }
    perf_emit("mtbsv", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("mtbsv", key, N, iters, flops, t_subject, t_mg);
    free(A); free(X); free(Xi);
}

static const int default_sizes[] = {128, 256, 512, 1024};
static const int default_incxs[] = {1, 2, -1};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    int incxs[8];
    int n_incx = perf_parse_int_list("BLAS_PERF_INCX", default_incxs,
        (int)(sizeof(default_incxs)/sizeof(default_incxs[0])), incxs, 8);
    perf_print_header();
    const char transes[] = { 'N','T' };
    const char diags[]   = { 'N', 'U' };
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < sizeof(transes); ++t)
    for (size_t d = 0; d < sizeof(diags); ++d) {
        char uplo = (u == 0) ? 'U' : 'L';
        char trans = transes[t];
        char diag = diags[d];
        for (int xi = 0; xi < n_incx; ++xi) {
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int i = 0; i < n; ++i) run_one(uplo, trans, diag, sizes[i], 16, incx, iters, warmup);
        }
    }
    return 0;
}
