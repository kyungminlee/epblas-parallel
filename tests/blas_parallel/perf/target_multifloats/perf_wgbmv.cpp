/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for wgbmv (overlay vs migrated).
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


BLAS_EXTERN void wgbmv_(const char *, const int *, const int *, const int *, const int *,
    const MFC *, const MFC *, const int *, const MFC *, const int *,
    const MFC *, MFC *, const int *, size_t);
BLAS_EXTERN void wgbmv_migrated_(const char *, const int *, const int *, const int *, const int *,
    const MFC *, const MFC *, const int *, const MFC *, const int *,
    const MFC *, MFC *, const int *, size_t);

static void run_one(char trans, int M, int N, int KL, int KU,
                    int incx, int incy, int iters, int warmup) {
    MFC alpha = MFC_FROM(0.7, 0.0), beta = MFC_FROM(0.3, 0.0);
    int LDA = KL + KU + 1;
    MFC *A  = (MFC *)perf_aligned_alloc(64, (size_t)LDA * (size_t)N * sizeof(MFC));
    const int XL = (trans == 'N') ? N : M;
    const int YL = (trans == 'N') ? M : N;
    const int absx = incx < 0 ? -incx : incx;
    const int absy = incy < 0 ? -incy : incy;
    const size_t lenx = (size_t)1 + (size_t)(XL - 1) * (size_t)absx;
    const size_t leny = (size_t)1 + (size_t)(YL - 1) * (size_t)absy;
    MFC *X  = (MFC *)perf_aligned_alloc(64, lenx * sizeof(MFC));
    MFC *Y  = (MFC *)perf_aligned_alloc(64, leny * sizeof(MFC));
    MFC *Yi = (MFC *)perf_aligned_alloc(64, leny * sizeof(MFC));
    for (size_t i = 0; i < (size_t)LDA*N; ++i) { int s = 2; A[i] = MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < lenx; ++i) { int s = 3; X[i] = MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    for (size_t i = 0; i < leny; ++i) { int s = 4; Yi[i] = MFC_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131)); }
    memcpy(Y, Yi, leny * sizeof(MFC));
    for (int r = 0; r < warmup; ++r) {
        wgbmv_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof(MFC));
        wgbmv_migrated_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
        memcpy(Y, Yi, leny * sizeof(MFC));
    }
    memcpy(Y, Yi, leny * sizeof(MFC));
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) wgbmv_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
    double t1 = perf_now_s();
    double t_subject = (t1 - t0) / (iters ? iters : 1);
    memcpy(Y, Yi, leny * sizeof(MFC));
    t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) wgbmv_migrated_(&trans, &M, &N, &KL, &KU, &alpha, A, &LDA, X, &incx, &beta, Y, &incy, 1);
    t1 = perf_now_s();
    double t_mg = (t1 - t0) / (iters ? iters : 1);
    double flops = 8.0 * (double)(KL+KU+1) * (double)N;
    char key[24];
    if (incx == 1 && incy == 1) {
        key[0] = trans; key[1] = 0;
    } else if (incy == 1) {
        snprintf(key, sizeof(key), "%c/x%d", trans, incx);
    } else if (incx == 1) {
        snprintf(key, sizeof(key), "%c/y%d", trans, incy);
    } else {
        snprintf(key, sizeof(key), "%c/x%d/y%d", trans, incx, incy);
    }
    perf_emit("wgbmv", key, N, iters, flops, t_subject, t_mg);
    perf_emit_json("wgbmv", key, N, iters, flops, t_subject, t_mg);
    free(A); free(X); free(Y); free(Yi);
}

static const int default_sizes[] = {128, 256, 512, 1024};
static const int default_incxs[] = {1, 2, -1};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    int incxs[8], incys[8];
    int n_incx = perf_parse_int_list("BLAS_PERF_INCX", default_incxs,
        (int)(sizeof(default_incxs)/sizeof(default_incxs[0])), incxs, 8);
    int n_incy = perf_parse_int_list("BLAS_PERF_INCY", incxs, n_incx, incys, 8);
    perf_print_header();
    const char transes[] = { 'N','T','C' };
    for (size_t t = 0; t < sizeof(transes); ++t)
        for (int xi = 0; xi < n_incx; ++xi) {
            int incx = incxs[xi]; if (incx == 0) continue;
            for (int yi = 0; yi < n_incy; ++yi) {
                int incy = incys[yi]; if (incy == 0) continue;
                for (int i = 0; i < n; ++i)
                    run_one(transes[t], sizes[i], sizes[i], 16, 16, incx, incy, iters, warmup);
            }
        }
    return 0;
}
