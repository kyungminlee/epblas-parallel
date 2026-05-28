/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for mrot (overlay vs migrated).
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


BLAS_EXTERN void mrot_(const int *, MFR *, const int *, MFR *, const int *,
    const MFR *, const MFR *);
BLAS_EXTERN void mrot_migrated_(const int *, MFR *, const int *, MFR *, const int *,
    const MFR *, const MFR *);

static void run_one(int N, int iters, int warmup) {
    int one = 1;
    MFR c_ = MFR_FROM(0.7), s_ = MFR_FROM(0.3);
    MFR *X = (MFR *)perf_aligned_alloc(64, (size_t)N * sizeof(MFR));
    MFR *Y = (MFR *)perf_aligned_alloc(64, (size_t)N * sizeof(MFR));
    MFR *Xi = (MFR *)perf_aligned_alloc(64, (size_t)N * sizeof(MFR));
    MFR *Yi = (MFR *)perf_aligned_alloc(64, (size_t)N * sizeof(MFR));
    for (int i = 0; i < N; ++i) { int s = 0; Xi[i] = MFR_FROM(perf_fill_double(i, s)); }
    for (int i = 0; i < N; ++i) { int s = 1; Yi[i] = MFR_FROM(perf_fill_double(i, s)); }
    memcpy(X, Xi, (size_t)N * sizeof(MFR));
    memcpy(Y, Yi, (size_t)N * sizeof(MFR));
    for (int r = 0; r < warmup; ++r) {
        mrot_(&N, X, &one, Y, &one, &c_, &s_);
        memcpy(X, Xi, (size_t)N * sizeof(MFR)); memcpy(Y, Yi, (size_t)N * sizeof(MFR));
        mrot_migrated_(&N, X, &one, Y, &one, &c_, &s_);
        memcpy(X, Xi, (size_t)N * sizeof(MFR)); memcpy(Y, Yi, (size_t)N * sizeof(MFR));
    }
    /* Per-call kernel-only timing — keep memcpy resets out of the
     * timed window so they don't Amdahl-mask MT scaling. */
    double t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        mrot_(&N, X, &one, Y, &one, &c_, &s_);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, (size_t)N * sizeof(MFR)); memcpy(Y, Yi, (size_t)N * sizeof(MFR));
    }
    double t_subject = t_sum / (iters ? iters : 1);

    t_sum = 0;
    for (int it = 0; it < iters; ++it) {
        double a = perf_now_s();
        mrot_migrated_(&N, X, &one, Y, &one, &c_, &s_);
        double b = perf_now_s();
        t_sum += (b - a);
        memcpy(X, Xi, (size_t)N * sizeof(MFR)); memcpy(Y, Yi, (size_t)N * sizeof(MFR));
    }
    double t_mg = t_sum / (iters ? iters : 1);
    double flops = 6.0 * (double)N;
    perf_emit("mrot", "-", N, iters, flops, t_subject, t_mg);
    perf_emit_json("mrot", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Y); free(Xi); free(Yi);
}

static const int default_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_one(sizes[i], iters, warmup);
    return 0;
}
