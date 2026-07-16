/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for mscal (overlay vs migrated).
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


BLAS_EXTERN void mscal_(const int *, const MFR *, MFR *, const int *);
BLAS_EXTERN void mscal_migrated_(const int *, const MFR *, MFR *, const int *);

static void run_mscal(int N, int iters, int warmup) {
    int one = 1;
    MFR alpha = MFR_FROM(0.7);
    MFR *X  = PERF_ALLOC(MFR, N);
    MFR *Xi = PERF_ALLOC(MFR, N);
    PERF_FILL_R(MFR, Xi, N, 0);
    PERF_RESET(X, Xi, N, MFR);
    for (int r = 0; r < warmup; ++r) {
        mscal_(&N, &alpha, X, &one);          PERF_RESET(X, Xi, N, MFR);
        mscal_migrated_(&N, &alpha, X, &one); PERF_RESET(X, Xi, N, MFR);
    }
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(X, Xi, N, MFR), mscal_(&N, &alpha, X, &one));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(X, Xi, N, MFR), mscal_migrated_(&N, &alpha, X, &one));
    double flops = 1.0 * (double)N;
    PERF_EMIT("mscal", "-", N, iters, flops, t_subject, t_mg);
    free(X); free(Xi);
}

static const int default_sizes[] = {64, 128, 256, 512, 1024, 2048, 4096, 16384, 65536};
int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  200);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 20);
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    perf_print_header();
    for (int i = 0; i < n; ++i) run_mscal(sizes[i], iters, warmup);
    return 0;
}
