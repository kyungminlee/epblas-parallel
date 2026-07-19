/* Hand-maintained perf harness (originally generated; the generator was retired).
 *
 * Kernel-isolated C perf harness for qtbmv (overlay vs migrated).
 * Built per-executable with -ffunction-sections / --gc-sections.
 */
#include "../perf_common.h"

#include <quadmath.h>
#ifdef __cplusplus
#define BLAS_EXTERN extern "C"
#else
#define BLAS_EXTERN extern
#endif
typedef __float128 Q16;
typedef _Complex float __attribute__((mode(TC))) X16;
#define Q16_FROM(d) ((Q16)(double)(d))
#define X16_FROM(re, im) ((X16)((Q16)(double)(re) + 1.0i * (Q16)(double)(im)))
static inline Q16 Tr_from_d(double d) { return (Q16)d; }
static inline X16 Tc_from_d(double d) { return (X16)((Q16)d); }


BLAS_EXTERN void qtbmv_(const char *, const char *, const char *, const int *, const int *,
    const Q16 *, const int *, Q16 *, const int *, size_t, size_t, size_t);
BLAS_EXTERN void qtbmv_migrated_(const char *, const char *, const char *, const int *, const int *,
    const Q16 *, const int *, Q16 *, const int *, size_t, size_t, size_t);

static void run_one(char uplo, char trans, char diag, int N, int K, int incx,
                    int iters, int warmup) {
    int LDA = K + 1;
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    const size_t Aelt = (size_t)LDA * (size_t)N;
    Q16 *A  = PERF_ALLOC(Q16, Aelt);
    Q16 *X  = PERF_ALLOC(Q16, lenx);
    Q16 *Xi = PERF_ALLOC(Q16, lenx);
    PERF_FILL_R(Q16, A, Aelt, 2);
    /* diagonal at known row of band — large to stabilize tbsv */
    int diag_row = (uplo == 'U') ? K : 0;
    for (int j = 0; j < N; ++j) A[(size_t)j * LDA + diag_row] = Tr_from_d((double)(K + 4));
    PERF_FILL_R(Q16, Xi, lenx, 3);
    PERF_RESET(X, Xi, lenx, Q16);
    for (int r = 0; r < warmup; ++r) {
        qtbmv_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1);          PERF_RESET(X, Xi, lenx, Q16);
        qtbmv_migrated_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1); PERF_RESET(X, Xi, lenx, Q16);
    }
    /* Per-call timing (reset out of the timed window). */
    double t_subject, t_mg;
    PERF_TIME_PER_CALL(t_subject, iters, PERF_RESET(X, Xi, lenx, Q16), qtbmv_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1));
    PERF_TIME_PER_CALL(t_mg,      iters, PERF_RESET(X, Xi, lenx, Q16), qtbmv_migrated_(&uplo, &trans, &diag, &N, &K, A, &LDA, X, &incx, 1, 1, 1));
    char key[16];
    if (incx == 1) {
        key[0] = uplo; key[1] = trans; key[2] = diag; key[3] = 0;
    } else {
        snprintf(key, sizeof(key), "%c%c%c/x%d", uplo, trans, diag, incx);
    }
    PERF_EMIT("qtbmv", key, N, iters, t_subject, t_mg);
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
