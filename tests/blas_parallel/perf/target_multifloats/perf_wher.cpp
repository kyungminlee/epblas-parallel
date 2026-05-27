/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for wher (overlay vs migrated).
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


BLAS_EXTERN void wher_(const char *, const int *, const MFR *,
    const MFC *, const int *, MFC *, const int *, size_t);
BLAS_EXTERN void wher_migrated_(const char *, const int *, const MFR *,
    const MFC *, const int *, MFC *, const int *, size_t);

static void run_one(char uplo, int N, int incx, int iters, int warmup) {
    MFR alpha = MFR_FROM(0.7);
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    MFC *A  = (MFC *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof(MFC));
    MFC *Ai = (MFC *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof(MFC));
    MFC *X  = (MFC *)perf_aligned_alloc(64, lenx * sizeof(MFC));
    for (size_t i = 0; i < (size_t)N*N; ++i) { int s = 2; Ai[i] = MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < lenx; ++i)      { int s = 3; X[i] = MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    memcpy(A, Ai, (size_t)N * (size_t)N * sizeof(MFC));
    for (int r = 0; r < warmup; ++r) {
        wher_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof(MFC));
        wher_migrated_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof(MFC));
    }
    /* Per-call kernel-only timing — keep memcpy reset out of timed window. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        wher_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof(MFC));
    }
    double t_subject = t_sum / (iters ? iters : 1);
    t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        wher_migrated_(&uplo, &N, &alpha, X, &incx, A, &N, 1);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(A, Ai, (size_t)N * (size_t)N * sizeof(MFC));
    }
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = 4.0 * (double)N * (double)N;
    char key[16];
    if (incx == 1) {
        key[0] = uplo; key[1] = 0;
    } else {
        snprintf(key, sizeof(key), "%c/x%d", uplo, incx);
    }
    perf_emit("wher", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("wher", key, N, iters, flops, t_subject, t_mg);
    free(A); free(Ai); free(X);
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
    for (size_t u = 0; u < 2; ++u) {
        char uplo = (u == 0) ? 'U' : 'L';
        for (int xi = 0; xi < n_incx; ++xi) {
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int i = 0; i < n; ++i) run_one(uplo, sizes[i], incx, iters, warmup);
        }
    }
    return 0;
}
