/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for wrotg (overlay vs migrated).
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


BLAS_EXTERN void wrotg_(MFC *, MFC *, MFR *, MFC *);
BLAS_EXTERN void wrotg_migrated_(MFC *, MFC *, MFR *, MFC *);

static void run_one(int iters, int warmup) {
    MFC A = MFC_FROM(0.7, 0.0), B = MFC_FROM(0.7, 0.0);
    MFR C = Tr_from_d(0.0);
    MFC S = Tc_from_d(0.0);
    /* per call: regenerate fresh A, B inputs */
    for (int r = 0; r < warmup; ++r) {
        MFC a = A, b = B; wrotg_(&a, &b, &C, &S);
        a = A; b = B; wrotg_migrated_(&a, &b, &C, &S);
    }
    double t_subject, t_mg;
    PERF_TIME(t_subject, iters, MFC a = A, b = B; wrotg_(&a, &b, &C, &S));
    PERF_TIME(t_mg,      iters, MFC a = A, b = B; wrotg_migrated_(&a, &b, &C, &S));
    /* report time per call as "flops" abuse: per-call flop count ~10. */
    double flops = 10.0;
    PERF_EMIT("wrotg", "-", iters, iters, flops, t_subject, t_mg);
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}
