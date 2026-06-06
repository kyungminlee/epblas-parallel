/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for mnrm2 (overlay vs migrated).
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


BLAS_EXTERN MFR mnrm2_(const int *, const MFR *, const int *);
BLAS_EXTERN MFR mnrm2_migrated_(const int *, const MFR *, const int *);

static void run_one(int N, int iters, int warmup) {
    int one = 1;
    MFR r;
    MFR *X = PERF_ALLOC(MFR, N);
    PERF_FILL_R(MFR, X, N, 0);
    for (int r2 = 0; r2 < warmup; ++r2) {
        r = mnrm2_(&N, X, &one);
        r = mnrm2_migrated_(&N, X, &one);
    }
    double t_subject, t_mg;
    PERF_TIME(t_subject, iters, r = mnrm2_(&N, X, &one));
    PERF_TIME(t_mg,      iters, r = mnrm2_migrated_(&N, X, &one));
    double flops = 2.0 * (double)N;
    PERF_EMIT("mnrm2", "-", N, iters, flops, t_subject, t_mg);
    if ((double)(*((double*)&r)) == -123e30) { free(X); return; }
    free(X);
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
