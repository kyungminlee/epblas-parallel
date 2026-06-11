/* GENERATED-BY-gen_perf_harnesses — do not edit by hand; regenerate via
 *   python3 scripts/gen_perf_harnesses.py
 *
 * Kernel-isolated C perf harness for mrotmg (overlay vs migrated).
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


BLAS_EXTERN void mrotmg_(MFR *, MFR *, MFR *, const MFR *, MFR *);
BLAS_EXTERN void mrotmg_migrated_(MFR *, MFR *, MFR *, const MFR *, MFR *);

static void run_one(int iters, int warmup) {
    MFR D1 = MFR_FROM(0.7), D2 = MFR_FROM(0.7), X1 = MFR_FROM(0.7), Y1 = MFR_FROM(0.7);
    MFR PARAM[5];
    for (int r = 0; r < warmup; ++r) {
        MFR d1 = D1, d2 = D2, x1 = X1;
        mrotmg_(&d1, &d2, &x1, &Y1, PARAM);
        d1 = D1; d2 = D2; x1 = X1;
        mrotmg_migrated_(&d1, &d2, &x1, &Y1, PARAM);
    }
    double t_subject, t_mg;
    PERF_TIME(t_subject, iters, MFR d1 = D1, d2 = D2, x1 = X1; mrotmg_(&d1, &d2, &x1, &Y1, PARAM));
    PERF_TIME(t_mg,      iters, MFR d1 = D1, d2 = D2, x1 = X1; mrotmg_migrated_(&d1, &d2, &x1, &Y1, PARAM));
    double flops = 20.0;
    PERF_EMIT("mrotmg", "-", iters, iters, flops, t_subject, t_mg);
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS", 100000);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 1000);
    perf_print_header();
    run_one(iters, warmup);
    return 0;
}
