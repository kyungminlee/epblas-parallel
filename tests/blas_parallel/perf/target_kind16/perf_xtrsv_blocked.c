/* 5-way timing + fuzz harness for xtrsv (kind16 complex).
 *
 * Compares:
 *   (1) overlay xtrsv_ at OMP=1
 *   (2) overlay xtrsv_ at OMP=4
 *   (3) overlay xtrsv_blocked_ at OMP=1
 *   (4) overlay xtrsv_blocked_ at OMP=4
 *   (5) migrated xtrsv_migrated_ at OMP=1
 *
 * Also performs a correctness check by comparing xtrsv_blocked_ output
 * to xtrsv_migrated_ output element-wise.
 *
 * Env knobs:
 *   BLAS_PERF_ITERS    timed iters per (shape, size)   (default 50)
 *   BLAS_PERF_WARMUP   warmup calls                    (default 5)
 *   BLAS_PERF_SIZES    comma-separated sizes           (default 256,512,1024)
 *   BLAS_PERF_INCX     comma-separated strides         (default 1)
 *   XTRSV_FUZZ_TOL     abs tol for blocked-vs-migrated (default 1e-25)
 *   XTRSV_NB           block size                      (default 64)
 */

#include "../perf_common.h"

#include <quadmath.h>
#include <math.h>
#include <omp.h>

#ifdef __cplusplus
#define BLAS_EXTERN extern "C"
#else
#define BLAS_EXTERN extern
#endif

typedef __float128 Q16;
typedef _Complex float __attribute__((mode(TC))) X16;
#define X16_FROM(re, im) ((X16)((Q16)(double)(re) + 1.0i * (Q16)(double)(im)))
static inline X16 Tc_from_d(double d) { return (X16)((Q16)d); }

BLAS_EXTERN void xtrsv_(const char *, const char *, const char *, const int *,
    const X16 *, const int *, X16 *, const int *, size_t, size_t, size_t);
BLAS_EXTERN void xtrsv_blocked_(const char *, const char *, const char *, const int *,
    const X16 *, const int *, X16 *, const int *, size_t, size_t, size_t);
BLAS_EXTERN void xtrsv_migrated_(const char *, const char *, const char *, const int *,
    const X16 *, const int *, X16 *, const int *, size_t, size_t, size_t);

typedef void (*xtrsv_fn)(const char *, const char *, const char *, const int *,
    const X16 *, const int *, X16 *, const int *, size_t, size_t, size_t);

static Q16 q16_abs(Q16 z) { return z < 0 ? -z : z; }
static Q16 cabsq16(X16 z) {
    Q16 r = __real__ z, i = __imag__ z;
    Q16 ar = q16_abs(r), ai = q16_abs(i);
    if (ar < ai) { Q16 t = ar; ar = ai; ai = t; }
    if (ar == 0) return 0;
    Q16 q = ai / ar;
    return ar * sqrtq(1.0Q + q * q);
}

/* Element-wise relative-error check. Triangular systems generated from
 * uniform diagonals and small random off-diagonals can produce solution
 * values of arbitrary magnitude — comparing absolute differences is
 * meaningless, so we compare diff/|ref|. */
static int fuzz_check(const X16 *X_ref, const X16 *X_test, int N, int incx,
                      Q16 tol, const char *label) {
    int bad = 0;
    Q16 maxrelerr = 0;
    for (int i = 0; i < N; ++i) {
        int idx = (incx > 0) ? i * incx : -(N - 1 - i) * incx;
        X16 d = X_ref[idx] - X_test[idx];
        Q16 ad = cabsq16(d);
        Q16 mag = cabsq16(X_ref[idx]);
        Q16 denom = (mag > 1) ? mag : (Q16)1.0Q;
        Q16 rel = ad / denom;
        if (rel > maxrelerr) maxrelerr = rel;
        if (rel > tol) bad++;
    }
    if (bad > 0) {
        char buf[64];
        quadmath_snprintf(buf, sizeof(buf), "%.6Qg", maxrelerr);
        fprintf(stderr,
                "[FUZZ FAIL] %s: %d/%d elems exceed tol; max relerr=%s\n",
                label, bad, N, buf);
    }
    return bad;
}

static double time_kernel(xtrsv_fn fn, int nthreads,
                          char uplo, char trans, char diag, int N, int incx,
                          const X16 *A, const X16 *Xi, X16 *X,
                          size_t lenx, int iters, int warmup) {
    omp_set_num_threads(nthreads);
    for (int r = 0; r < warmup; ++r) {
        memcpy(X, Xi, lenx * sizeof(X16));
        fn(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1);
    }
    double t0 = perf_now_s();
    for (int it = 0; it < iters; ++it) {
        memcpy(X, Xi, lenx * sizeof(X16));
        fn(&uplo, &trans, &diag, &N, A, &N, X, &incx, 1, 1, 1);
    }
    double t1 = perf_now_s();
    return (t1 - t0) / (iters ? iters : 1);
}

static void run_one(char uplo, char trans, char diag, int N, int incx,
                    int iters, int warmup, Q16 tol) {
    const int absx = incx < 0 ? -incx : incx;
    const size_t lenx = (size_t)1 + (size_t)(N - 1) * (size_t)absx;
    X16 *A   = (X16 *)perf_aligned_alloc(64, (size_t)N * (size_t)N * sizeof(X16));
    X16 *X   = (X16 *)perf_aligned_alloc(64, lenx * sizeof(X16));
    X16 *Xi  = (X16 *)perf_aligned_alloc(64, lenx * sizeof(X16));
    X16 *Xref = (X16 *)perf_aligned_alloc(64, lenx * sizeof(X16));
    X16 *Xtst = (X16 *)perf_aligned_alloc(64, lenx * sizeof(X16));

    for (size_t i = 0; i < (size_t)N * N; ++i) {
        int s = 2;
        A[i] = X16_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131));
    }
    /* Diagonally dominant (matches existing perf_xtrsv harness). */
    for (int i = 0; i < N; ++i) {
        A[(size_t)i * N + i] = Tc_from_d((double)(N + 4));
    }
    for (size_t i = 0; i < lenx; ++i) {
        int s = 3;
        Xi[i] = X16_FROM(perf_fill_double(i, s), perf_fill_double(i, s + 131));
    }

    /* Correctness: blocked vs migrated. */
    memcpy(Xref, Xi, lenx * sizeof(X16));
    omp_set_num_threads(1);
    xtrsv_migrated_(&uplo, &trans, &diag, &N, A, &N, Xref, &incx, 1, 1, 1);
    memcpy(Xtst, Xi, lenx * sizeof(X16));
    omp_set_num_threads(4);
    xtrsv_blocked_(&uplo, &trans, &diag, &N, A, &N, Xtst, &incx, 1, 1, 1);
    char label[64];
    snprintf(label, sizeof(label), "xtrsv blk %c%c%c N=%d incx=%d", uplo, trans, diag, N, incx);
    fuzz_check(Xref, Xtst, N, incx, tol, label);

    /* Timing: 5 configurations. */
    double t_cur_1 = time_kernel(xtrsv_,         1, uplo, trans, diag, N, incx,
                                  A, Xi, X, lenx, iters, warmup);
    double t_cur_4 = time_kernel(xtrsv_,         4, uplo, trans, diag, N, incx,
                                  A, Xi, X, lenx, iters, warmup);
    double t_blk_1 = time_kernel(xtrsv_blocked_, 1, uplo, trans, diag, N, incx,
                                  A, Xi, X, lenx, iters, warmup);
    double t_blk_4 = time_kernel(xtrsv_blocked_, 4, uplo, trans, diag, N, incx,
                                  A, Xi, X, lenx, iters, warmup);
    double t_mig_1 = time_kernel(xtrsv_migrated_, 1, uplo, trans, diag, N, incx,
                                  A, Xi, X, lenx, iters, warmup);

    double flops = 4.0 * (double)N * (double)N;
    char key[16];
    if (incx == 1) {
        key[0] = uplo; key[1] = trans; key[2] = diag; key[3] = 0;
    } else {
        snprintf(key, sizeof(key), "%c%c%c/x%d", uplo, trans, diag, incx);
    }

    double g_cur_1 = flops / t_cur_1 / 1e9;
    double g_cur_4 = flops / t_cur_4 / 1e9;
    double g_blk_1 = flops / t_blk_1 / 1e9;
    double g_blk_4 = flops / t_blk_4 / 1e9;
    double g_mig_1 = flops / t_mig_1 / 1e9;

    printf("%-10s  %-7s  %5d  %4d  %8.4f  %8.4f  %8.4f  %8.4f  %8.4f  "
           "%6.2fx %6.2fx %6.2fx %6.2fx\n",
           "xtrsv", key, N, iters,
           g_cur_1, g_cur_4, g_blk_1, g_blk_4, g_mig_1,
           g_cur_4 / g_cur_1,   /* scaling of current */
           g_blk_1 / g_cur_1,   /* blocked-OMP1 vs current-OMP1 */
           g_blk_4 / g_cur_4,   /* blocked-OMP4 vs current-OMP4 */
           g_blk_4 / g_mig_1);  /* blocked-OMP4 vs migrated */
    fflush(stdout);

    free(A); free(X); free(Xi); free(Xref); free(Xtst);
}

int main(void) {
    int iters  = perf_env_int("BLAS_PERF_ITERS",  50);
    int warmup = perf_env_int("BLAS_PERF_WARMUP", 5);
    const char *tol_s = getenv("XTRSV_FUZZ_TOL");
    /* Relative-error tolerance — well above quad-prec eps (1.93e-34)
     * but tight enough to catch real algorithmic bugs. */
    Q16 tol = (tol_s && *tol_s) ? strtoflt128(tol_s, NULL) : 1.0e-28Q;

    static const int default_sizes[] = {128, 256, 512, 1024};
    static const int default_incxs[] = {1};
    int sizes[32];
    int n = perf_parse_sizes(default_sizes,
        (int)(sizeof(default_sizes)/sizeof(default_sizes[0])), sizes, 32);
    int incxs[8];
    int n_incx = perf_parse_int_list("BLAS_PERF_INCX", default_incxs,
        (int)(sizeof(default_incxs)/sizeof(default_incxs[0])), incxs, 8);

    printf("# columns: cur_1, cur_4, blk_1, blk_4, mig_1 (all in GFlop/s)\n");
    printf("# ratios:  cur4/cur1  blk1/cur1  blk4/cur4  blk4/mig1\n");
    printf("# routine    key      N      it   cur_1     cur_4     blk_1     blk_4     mig_1     ratios\n");

    const char transes[] = { 'N', 'T', 'C' };
    const char diags[]   = { 'N', 'U' };
    for (size_t u = 0; u < 2; ++u) for (size_t t = 0; t < sizeof(transes); ++t)
    for (size_t d = 0; d < sizeof(diags); ++d) {
        char uplo  = (u == 0) ? 'U' : 'L';
        char trans = transes[t];
        char diag  = diags[d];
        for (int xi = 0; xi < n_incx; ++xi) {
            int incx = incxs[xi];
            if (incx == 0) continue;
            for (int i = 0; i < n; ++i)
                run_one(uplo, trans, diag, sizes[i], incx, iters, warmup, tol);
        }
    }
    return 0;
}
