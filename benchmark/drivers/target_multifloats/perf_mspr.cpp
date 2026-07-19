/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for mspr (overlay vs migrated).
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


BLAS_EXTERN void mspr_(const char *, const int *, const MFR *,
    const MFR *, const int *, MFR *, size_t);
BLAS_EXTERN void mspr_migrated_(const char *, const int *, const MFR *,
    const MFR *, const int *, MFR *, size_t);

static void run_one(char uplo, int N, int incx, int iters, int warmup) {
    MFR alpha = MFR_FROM(0.7);
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    size_t AP_LEN = (size_t)N * (size_t)(N + 1) / 2;
    MFR *AP  = PERF_ALLOC(MFR, AP_LEN);
    MFR *APi = PERF_ALLOC(MFR, AP_LEN);
    MFR *X   = PERF_ALLOC(MFR, lenx);
    PERF_FILL_R(MFR, APi, AP_LEN, 2);
    PERF_FILL_R(MFR, X,   lenx, 3);
    PERF_RESET(AP, APi, AP_LEN, MFR);
    for (int r = 0; r < warmup; ++r) {
        mspr_(&uplo, &N, &alpha, X, &incx, AP, 1);          PERF_RESET(AP, APi, AP_LEN, MFR);
        mspr_migrated_(&uplo, &N, &alpha, X, &incx, AP, 1); PERF_RESET(AP, APi, AP_LEN, MFR);
    }
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(AP, APi, AP_LEN, MFR), mspr_(&uplo, &N, &alpha, X, &incx, AP, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(AP, APi, AP_LEN, MFR), mspr_migrated_(&uplo, &N, &alpha, X, &incx, AP, 1));
    char key[16];
    if (incx == 1) {
        key[0] = uplo; key[1] = 0;
    } else {
        snprintf(key, sizeof(key), "%c/x%d", uplo, incx);
    }
    PERF_EMIT("mspr", key, N, iters, t_subject, t_mg);
    free(AP); free(APi); free(X);
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
